# ============================================================================
#  operation.py
#  The two ways the same set of nanogrids can be operated, side by side so the
#  single difference between them is easy to see:
#
#    run_isolated       - every lodge stands alone. Each sizes and runs its own
#                         battery on its own load, and the fleet battery is just
#                         the sum of those independent optima.
#    run_interconnected - the lodges share one pooled battery. Local PV serves
#                         local load first (pool_cluster_residual), then the
#                         leftovers are pooled across the cluster and served by a
#                         single shared battery that is sized once.
#
#  Both return the same result dictionary, so the reporting and analysis code
#  treats them interchangeably. This file was previously split across
#  scenario_isolated.py and scenario_interconnected.py.
# ============================================================================

import numpy as np
import pandas as pd
from configuration import *
from simulation import run_simulation
from optimiser import run_battery_sweep, calculate_lcoe
from cabling import cable_length_routed


# ----------------------------------------------------------------------------
#  Isolated operation: each lodge on its own
# ----------------------------------------------------------------------------


def run_isolated(pv_kw, load_kw, config=DEFAULT_CONFIG, overrides=None, battery_sizes=None):  # N isolated nanogrids: each its own PV, diesel and individually-sized battery

    n_steps, n_households = load_kw.shape
    hours_per_step = Time_step_min / 60

    optimal_batteries = np.zeros(n_households)                   # Each household's own optimal battery size [kWh]
    total_diesel_fuel_L  = 0.0                                   # Fleet diesel fuel, summed over households [L]
    total_diesel_kWh     = 0.0                                   # Fleet diesel energy [kWh]
    total_unmet_kWh      = 0.0
    total_curtailed_kWh  = 0.0                                   # Fleet curtailed PV [kWh]
    total_emissions_kg   = 0.0                                   # Fleet CO2 emissions [kg]

    print(f"Running isolated scenario ({config['name']}): {n_households} household sweeps...")
    print(f"      [window] {n_steps} steps, "
      f"lodge 2 annual load {load_kw[:, 1].sum() * Time_step_min / 60:.0f} kWh")

    for h in range(n_households):                               # Optimise each household on its own load column

        _, optimal_h = run_battery_sweep(pv_kw[:, h], load_kw[:, h], verbose=False, config=config, overrides=overrides, battery_sizes=battery_sizes)   # Each household swept on its own load (battery_sizes forwarded so a coarser grid can be passed)
        optimal_batteries[h] = optimal_h["battery_kWh"]

        sim_h = run_simulation(pv_kw[:, h], load_kw[:, h], optimal_batteries[h], use_diesel=config["use_diesel"])  # Re-run at its optimum to get its energy flows
        total_diesel_fuel_L += sim_h["diesel_fuel_L"]
        total_diesel_kWh    += sim_h["annual_diesel_kWh"]
        total_unmet_kWh     += sim_h["annual_unmet_kWh"]
        total_curtailed_kWh += sim_h["annual_curtailed_kWh"]
        total_emissions_kg  += sim_h["diesel_emissions_kg"]
        print(f"      household {h+1:>2}/{n_households}: {optimal_batteries[h]:.1f} kWh")   # One tidy line per household

    battery_kWh_total = optimal_batteries.sum()                  # Sum of isolated optima = fleet battery
    cluster_load_kWh  = load_kw.sum() * hours_per_step           # True total cluster load [kWh/year]
    diesel_capacity_total = load_kw.max(axis=0).sum()           # Each genset sized to its own house's peak load [kW]

    fleet = {"diesel_fuel_L": total_diesel_fuel_L}               # Wrap fleet fuel so calculate_lcoe can read it like a results dict
    lcoe, ann_capex, ann_opex = calculate_lcoe(                  # One cluster LCOE, same function as the interconnected scenario
        fleet, battery_kWh_total, n_units=n_households,
        battery_kWh_total=battery_kWh_total, annual_load_override=cluster_load_kWh,
        diesel_capacity_kW_total=diesel_capacity_total, config=config, overrides=overrides
    )
    renewable_fraction = (cluster_load_kWh - total_diesel_kWh - total_unmet_kWh) / cluster_load_kWh   # Share served by PV and battery

    result = {                                                  # Single summary dict (no sweep table: each household already swept internally)
        "battery_kWh_total":    battery_kWh_total,
        "optimal_batteries":    optimal_batteries,
        "lcoe_eur_kWh":         lcoe,
        "annualised_capex_eur": ann_capex,
        "annual_opex_eur":      ann_opex,
        "cluster_load_kWh":     cluster_load_kWh,
        "annual_diesel_kWh":    total_diesel_kWh,
        "annual_curtailed_kWh": total_curtailed_kWh,
        "diesel_fuel_L":        total_diesel_fuel_L,
        "diesel_emissions_kg":  total_emissions_kg,
        "renewable_fraction":   renewable_fraction,
        # Sizing inputs kept alongside the results so the LCOE can be rebuilt term by term
        # later (lcoe_breakdown in optimiser.py) without re-deriving them from the load array.
        "n_units":              n_households,
        "diesel_capacity_kW":   diesel_capacity_total,
        "cable_length_m":       0.0,                     # isolated fleet: no interconnection cable
    }

    return result, n_households

# ----------------------------------------------------------------------------
#  Interconnected operation: one pooled battery for the whole cluster
# ----------------------------------------------------------------------------


def pool_cluster_residual(pv_kw, load_kw):                         # Local PV serves local load first, then pool the leftovers across the cluster

    direct = np.minimum(pv_kw, load_kw)                            # Per-household PV used directly on own load [kW], shape (n_steps, N)
    surplus_pv    = pv_kw   - direct                              # Per-household PV left after serving own load [kW]
    residual_load = load_kw - direct                              # Per-household load left after own PV [kW]

    pooled_pv   = surplus_pv.sum(axis=1)                          # Sum surplus PV across households -> 1-D series [kW]
    pooled_load = residual_load.sum(axis=1)                       # Sum residual load across households -> 1-D series [kW]

    return pooled_pv, pooled_load


def run_interconnected(pv_kw, load_kw, config=DEFAULT_CONFIG, overrides=None, battery_sizes=None):   # One shared battery; PV and diesel stay individual per household

    n_households   = load_kw.shape[1]
    hours_per_step = Time_step_min / 60
    cluster_load_kWh = load_kw.sum() * hours_per_step             # True total cluster load [kWh/year]
    diesel_capacity_total = load_kw.max(axis=0).sum()           # Per-house gensets, each sized to its own peak load [kW]
    cable_length_m = cable_length_routed(Tinos_coordinates) * CABLING_COSTS_ON      # Routed MST interconnection cable for the cluster [m]

    pooled_pv, pooled_load = pool_cluster_residual(pv_kw, load_kw)  # Reduce the cluster to a single residual nanogrid

    # battery_sizes lets a caller (e.g. the PV sensitivity) pass a coarser grid to run
    # faster. Left as None it falls back to the configuration.py sweep, unchanged.
    if battery_sizes is None:
        battery_sizes = np.arange(Battery_min_kWh, Battery_max_kWh + Battery_step_kWh, Battery_step_kWh)

    rows = []
    print(f"Running interconnected sweep ({config['name']}): {len(battery_sizes)} simulations for N = {n_households}...")

    for battery_kWh in battery_sizes:

        sim = run_simulation(pooled_pv, pooled_load, battery_kWh, use_diesel=config["use_diesel"])   # config decides whether diesel may close gaps
        lcoe, ann_capex, ann_opex = calculate_lcoe(                            # Rank by true cluster LCOE: N PV+diesel sets, one shared battery
            sim, battery_kWh, n_units=n_households,
            battery_kWh_total=battery_kWh, annual_load_override=cluster_load_kWh,
            diesel_capacity_kW_total=diesel_capacity_total,
            cable_length_m=cable_length_m, config=config, overrides=overrides
        )
        renewable_fraction = (cluster_load_kWh - sim["annual_diesel_kWh"] - sim["annual_unmet_kWh"]) / cluster_load_kWh   # Share served by PV and battery

        row = {
            "battery_kWh_total":    battery_kWh,                              # Shared battery = total fleet battery
            "lcoe_eur_kWh":         lcoe,
            "annualised_capex_eur": ann_capex,
            "annual_opex_eur":      ann_opex,
            "cluster_load_kWh":     cluster_load_kWh,
            "annual_diesel_kWh":    sim["annual_diesel_kWh"],
            "annual_curtailed_kWh": sim["annual_curtailed_kWh"],
            "diesel_fuel_L":        sim["diesel_fuel_L"],
            "annual_unmet_kWh":     sim["annual_unmet_kWh"],
            "diesel_emissions_kg":  sim["diesel_emissions_kg"],
            "renewable_fraction":   renewable_fraction,
            # Sizing inputs kept alongside the results so the LCOE can be rebuilt term by term
            # later (lcoe_breakdown in optimiser.py) without re-deriving them from the load array.
            "n_units":              n_households,
            "diesel_capacity_kW":   diesel_capacity_total,
            "cable_length_m":       cable_length_m,      # already zero when CABLING_COSTS_ON is False
        }
        rows.append(row)

    df_sweep = pd.DataFrame(rows)
    feasible = df_sweep[df_sweep["annual_unmet_kWh"] <= 1e-6]             # sizes that serve all pooled load (all of them when diesel is on)
    if len(feasible) > 0:
        optimal = feasible.loc[feasible["lcoe_eur_kWh"].idxmin()]
    else:
        optimal = df_sweep.loc[df_sweep["annual_unmet_kWh"].idxmin()]
        print("      WARNING: no shared battery size serves all load without diesel; picked the lowest-unmet size")

    if optimal["battery_kWh_total"] >= battery_sizes.max():                    # Optimum at the boundary means the true optimum is likely beyond the sweep
        print(f"      WARNING: interconnected optimum hit the sweep ceiling "
              f"({Battery_max_kWh} kWh). Raise Battery_max_kWh in config.py.")

    return df_sweep, optimal, n_households