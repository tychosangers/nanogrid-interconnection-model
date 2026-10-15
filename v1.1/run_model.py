# ============================================================================
#  run_model.py
#  Main run: compares isolated and interconnected operation for the
#  configuration and modelling window set in configuration.py, prints the
#  results and saves the figures in outputs/.
#  With wind enabled, PV and wind are summed into one generation series for
#  dispatch; their generation is reported separately.
# ============================================================================

import os
from configuration import *
from pv_profile import build_pv_for_window, capacity_per_grid
from wind_profile import build_wind_for_window
from operation import run_interconnected, run_isolated
from reporting import (plot_battery_distribution,
                       plot_lcoe_components, plot_lcoe_breakdown, plot_soc_lowest_week,
                       print_comparison, print_generation_split, print_cable_breakeven,
                       print_grid_comparison,
                       print_carbon_tax_comparison, plot_carbon_tax_comparison)
from load_profile import prepare_model_loads


def print_capacity_allocation(pv_sizes, base_loads, wind_sizes=None):
    """Print the PV (and wind) capacity of each nanogrid next to its share of the load."""
    hours_per_step = TIME_STEP_MIN / 60
    annual_kwh = base_loads.sum(axis=0) * hours_per_step
    share = annual_kwh / annual_kwh.sum() * 100

    print("\n" + "=" * 60)
    print(f"Capacity allocation  (ALLOCATION = {ALLOCATION!r})")
    print("=" * 60)
    for h, name in enumerate(NANOGRID_NAMES):
        extra = f", wind {wind_sizes[h]:.3f} kW" if wind_sizes is not None else ""
        print(f"  {name:<14}: PV {pv_sizes[h]:.3f} kWp{extra}   "
              f"(load {annual_kwh[h]:.0f} kWh, {share[h]:.1f}% of cluster)")
    extra_tot = f", wind {wind_sizes.sum():.3f} kW" if wind_sizes is not None else ""
    print("  " + "-" * 56)
    print(f"  {'Total':<14}: PV {pv_sizes.sum():.3f} kWp{extra_tot}   "
          f"(load {annual_kwh.sum():.0f} kWh)")
    print("=" * 60)


def load_inputs():
    """Load the nanogrid loads and the matching PV (and wind) generation."""
    base_loads, time_index = prepare_model_loads()
    n_nanogrids = base_loads.shape[1]

    pv_sizes = capacity_per_grid(base_loads, PV_CAPACITY_KW, ALLOCATION)
    wind_sizes = None
    if CONFIGURATION["use_wind"]:
        # An explicit kW list applies to PV only; wind then follows the load shares.
        wind_alloc = ALLOCATION if isinstance(ALLOCATION, str) else "load"
        wind_sizes = capacity_per_grid(base_loads, WIND_CAPACITY_KW, wind_alloc)
    print_capacity_allocation(pv_sizes, base_loads, wind_sizes)

    pv_cluster = build_pv_for_window(time_index, n_nanogrids, SOLAR_FILE, base_loads)
    wind_cluster = (build_wind_for_window(time_index, n_nanogrids, SOLAR_FILE, base_loads)
                    if CONFIGURATION["use_wind"] else None)
    days = len(time_index) * TIME_STEP_MIN / 60 / 24
    horizon_note = (f"Nanogrids: {', '.join(NANOGRID_NAMES)} | "
                    f"{time_index[0].date()} to {time_index[-1].date()} "
                    f"({days:.0f} days)")
    return base_loads, pv_cluster, wind_cluster, n_nanogrids, horizon_note, time_index


def run_one(base_loads, pv_cluster, wind_cluster):
    """Run both topologies. Returns the isolated result, the interconnected optimum and sweep."""
    loads = base_loads
    gen = pv_cluster if wind_cluster is None else pv_cluster + wind_cluster
    inter_sweep, inter_optimal, _ = run_interconnected(gen, loads, config=CONFIGURATION)
    iso_result, _                 = run_isolated(gen, loads, config=CONFIGURATION)
    return iso_result, inter_optimal, inter_sweep


def carbon_tax_comparison(base_loads, pv_cluster, wind_cluster):
    """Run both topologies without and with the carbon price and compare them."""
    loads = base_loads
    gen = pv_cluster if wind_cluster is None else pv_cluster + wind_cluster

    off = {"APPLY_CARBON_TAX": False}
    on  = {"APPLY_CARBON_TAX": True}

    _, inter_base,   _ = run_interconnected(gen, loads, config=CONFIGURATION, overrides=off)
    iso_base,        _ = run_isolated(gen, loads, config=CONFIGURATION, overrides=off)
    _, inter_carbon, _ = run_interconnected(gen, loads, config=CONFIGURATION, overrides=on)
    iso_carbon,      _ = run_isolated(gen, loads, config=CONFIGURATION, overrides=on)

    print_carbon_tax_comparison(iso_base, inter_base, iso_carbon, inter_carbon, CONFIGURATION)
    plot_carbon_tax_comparison(iso_base, inter_base, iso_carbon, inter_carbon, CONFIGURATION)


def main():
    os.makedirs("outputs", exist_ok=True)
    base_loads, pv_cluster, wind_cluster, n_nanogrids, horizon_note, time_index = load_inputs()

    iso_result, inter_optimal, inter_sweep = run_one(base_loads, pv_cluster, wind_cluster)
    print_comparison(iso_result, inter_optimal, n_nanogrids, CONFIGURATION, horizon_note)
    print_cable_breakeven(iso_result, inter_optimal, CONFIGURATION, BREAKEVEN_HORIZON_YEARS)
    print_grid_comparison(inter_optimal, GRID_PRICE_EUR_KWH, GRID_PROJECT_LIFETIME_YEARS)
    if CONFIGURATION["use_wind"]:
        print_generation_split(pv_cluster, wind_cluster, n_nanogrids)
    plot_lcoe_breakdown(iso_result, inter_optimal, CONFIGURATION, n_nanogrids)
    plot_battery_distribution(iso_result, inter_optimal, n_nanogrids, names=NANOGRID_NAMES)
    plot_lcoe_components(inter_sweep, inter_optimal, CONFIGURATION, n_nanogrids)
    gen_for_dispatch = pv_cluster if wind_cluster is None else pv_cluster + wind_cluster
    plot_soc_lowest_week(gen_for_dispatch, base_loads, inter_optimal, time_index, CONFIGURATION)
    if APPLY_CARBON_TAX and CONFIGURATION["use_diesel"]:
        carbon_tax_comparison(base_loads, pv_cluster, wind_cluster)
    return base_loads, pv_cluster, iso_result, inter_optimal


if __name__ == "__main__":
    base_loads, pv_cluster, iso_result, inter_optimal = main()
