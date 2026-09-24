import os
import numpy as np
import pandas as pd
from configuration import *
from pv_profile import build_pv_for_window, capacity_per_grid
from wind_profile import build_wind_for_window
from operation import run_interconnected, run_isolated   # the two operating modes: isolated fleet and interconnected cluster
from reporting import (plot_battery_comparison, plot_battery_distribution,
                              plot_lcoe_components, plot_lcoe_breakdown, plot_soc_lowest_week,
                              print_comparison, print_generation_split, print_cable_breakeven,
                              print_grid_comparison,
                              print_carbon_tax_comparison, plot_carbon_tax_comparison)
from load_profile import prepare_model_loads, MODEL_BUILDINGS


# ============================================================================
# run_model.py
# Press run. All choices live in configuration.py; nothing here needs editing.
# This file only orchestrates: load inputs, pick the configuration, call the
# scenarios, hand results to the reporters in reporting.py.
#
# What runs depends on CONFIGURATION (set in configuration.py):
#   CONFIG_PV_ONLY / CONFIG_PV_DIESEL / CONFIG_PV_WIND
#       greenfield comparison: isolated vs interconnected, pooled battery
#       re-optimised, metric = LCOE. This is the main SQ2 result.
#
# Wind: when CONFIGURATION uses wind, PV and wind are summed into one renewable
# supply for dispatch, while each source's gross generation is reported on its
# own. With wind off, every run is identical to the PV-only / PV+diesel case.
#
# Optional overlays (configuration.py): carbon tax (Apply_carbon_tax) and
# the interconnection cable cost (CABLING_COSTS_ON).
# ============================================================================

def print_capacity_allocation(pv_sizes, base_loads, wind_sizes=None):
    """Print each lodge's PV (and wind) capacity next to its share of the load.

    A quick sanity check on the ALLOCATION split: the per-lodge sizes should
    track the load shares, and the totals should match the real installed system
    (PV_capacity_kW * N, and Wind_capacity_kW * N when wind is on).
    """
    hours_per_step = Time_step_min / 60
    annual_kwh = base_loads.sum(axis=0) * hours_per_step        # per-lodge consumption over the window [kWh]
    share = annual_kwh / annual_kwh.sum() * 100                 # each lodge's % of total load

    print("\n" + "=" * 60)
    print(f"Capacity allocation  (ALLOCATION = {ALLOCATION!r})")
    print("=" * 60)
    for h, name in enumerate(MODEL_BUILDINGS):
        extra = f", wind {wind_sizes[h]:.3f} kW" if wind_sizes is not None else ""
        print(f"  {name:<14}: PV {pv_sizes[h]:.3f} kWp{extra}   "
              f"(load {annual_kwh[h]:.0f} kWh, {share[h]:.1f}% of cluster)")
    extra_tot = f", wind {wind_sizes.sum():.3f} kW" if wind_sizes is not None else ""
    print("  " + "-" * 56)
    print(f"  {'Total':<14}: PV {pv_sizes.sum():.3f} kWp{extra_tot}   "
          f"(load {annual_kwh.sum():.0f} kWh)")
    print("=" * 60)

def load_inputs():                                              # Load the real lodge loads + aligned PV (and wind, if used) once

    base_loads, time_index = prepare_model_loads()              # (n_steps, n) real lodges, 15-min
    n_households = base_loads.shape[1]

    # Per-lodge sizes for this ALLOCATION, printed so you can check the split.
    pv_sizes = capacity_per_grid(base_loads, PV_capacity_kW, ALLOCATION)
    wind_sizes = None
    if CONFIGURATION["use_wind"]:
        wind_alloc = ALLOCATION if isinstance(ALLOCATION, str) else "load"   # a PV kW list is not a wind list; fall back to load-share
        wind_sizes = capacity_per_grid(base_loads, Wind_capacity_kW, wind_alloc)
    print_capacity_allocation(pv_sizes, base_loads, wind_sizes)

    pv_cluster = build_pv_for_window(time_index, n_households, SOLAR_FILE, base_loads)
    wind_cluster = build_wind_for_window(time_index, n_households, SOLAR_FILE, base_loads) if CONFIGURATION["use_wind"] else None
    days = len(time_index) * Time_step_min / 60 / 24
    horizon_note = (f"Real lodges: {', '.join(MODEL_BUILDINGS)} | "
                    f"{time_index[0].date()} to {time_index[-1].date()} "
                    f"(~{days:.0f} days, NOT a full year)")
    return base_loads, pv_cluster, wind_cluster, n_households, horizon_note, time_index


def run_one(base_loads, pv_cluster, wind_cluster):             # Run both scenarios; returns the two results + the interconnected sweep
    loads = base_loads                                          # real metered load
    gen = pv_cluster if wind_cluster is None else pv_cluster + wind_cluster   # PV+wind combined for dispatch
    inter_sweep, inter_optimal, _ = run_interconnected(gen, loads, config=CONFIGURATION)
    iso_result, _                 = run_isolated(gen, loads, config=CONFIGURATION)
    return iso_result, inter_optimal, inter_sweep

def carbon_tax_comparison(base_loads, pv_cluster, wind_cluster):
    # Carbon-tax overlay (diesel config only). Runs the SAME configuration twice,
    # once with the tax off (the base case) and once with the EU ETS price from
    # configuration.py, for both topologies, so the change in the benefit and in the
    # renewable fraction can be measured against a clean baseline. Called from main()
    # only when Apply_carbon_tax is on, so a normal run without the tax is unaffected.
    loads = base_loads
    gen = pv_cluster if wind_cluster is None else pv_cluster + wind_cluster

    off = {"Apply_carbon_tax": False}                     # base case: tax forced off
    on  = {"Apply_carbon_tax": True}                      # +carbon: price taken from configuration.py

    _, inter_base,   _ = run_interconnected(gen, loads, config=CONFIGURATION, overrides=off)
    iso_base,        _ = run_isolated(gen, loads, config=CONFIGURATION, overrides=off)
    _, inter_carbon, _ = run_interconnected(gen, loads, config=CONFIGURATION, overrides=on)
    iso_carbon,      _ = run_isolated(gen, loads, config=CONFIGURATION, overrides=on)

    print_carbon_tax_comparison(iso_base, inter_base, iso_carbon, inter_carbon, CONFIGURATION)
    plot_carbon_tax_comparison(iso_base, inter_base, iso_carbon, inter_carbon, CONFIGURATION)
    
def main():

    os.makedirs("outputs", exist_ok=True)
    base_loads, pv_cluster, wind_cluster, n_households, horizon_note, time_index = load_inputs()

    # Greenfield comparison: isolated vs interconnected, pooled battery re-optimised, metric = LCOE.
    iso_result, inter_optimal, inter_sweep = run_one(base_loads, pv_cluster, wind_cluster)
    print_comparison(iso_result, inter_optimal, n_households, CONFIGURATION, horizon_note)
    print_cable_breakeven(iso_result, inter_optimal, CONFIGURATION, Breakeven_horizon_years)
    print_grid_comparison(inter_optimal, Grid_price_eur_kWh, Grid_project_lifetime_years)
    if CONFIGURATION["use_wind"]:
        print_generation_split(pv_cluster, wind_cluster, n_households)
    plot_battery_comparison(iso_result, inter_optimal, n_households)
    plot_lcoe_breakdown(iso_result, inter_optimal, CONFIGURATION, n_households)   # stacked bars: what each LCOE is made of
    plot_battery_distribution(iso_result, inter_optimal, n_households, names=MODEL_BUILDINGS)
    plot_lcoe_components(inter_sweep, inter_optimal, CONFIGURATION, n_households)
    gen_for_dispatch = pv_cluster if wind_cluster is None else pv_cluster + wind_cluster   # same supply the dispatch used
    plot_soc_lowest_week(gen_for_dispatch, base_loads, inter_optimal, time_index, CONFIGURATION)
    # Carbon-tax overlay: only when the toggle is on and the config burns diesel.
    if Apply_carbon_tax and CONFIGURATION["use_diesel"]:
        carbon_tax_comparison(base_loads, pv_cluster, wind_cluster)
    return base_loads, pv_cluster, iso_result, inter_optimal   # returned so they appear in the variable explorer


    
if __name__ == "__main__":
    base_loads, pv_cluster, iso_result, inter_optimal = main()         # results in the variable explorer