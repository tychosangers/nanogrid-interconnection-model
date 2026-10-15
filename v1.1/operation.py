# ============================================================================
#  operation.py
#  The two operating modes of the same set of nanogrids:
#    run_isolated:       each nanogrid sizes and operates its own battery.
#    run_interconnected: each nanogrid serves its own load with its own
#                        generation first; the remaining surpluses and deficits
#                        are pooled and served by a shared battery capacity.
#  Both return result dictionaries with the same keys.
# ============================================================================

import numpy as np
import pandas as pd
from configuration import *
from simulation import run_simulation
from optimiser import run_battery_sweep, calculate_lcoe
from cabling import cable_length_routed


# ----------------------------------------------------------------------------
#  Isolated operation
# ----------------------------------------------------------------------------

def run_isolated(pv_kw, load_kw, config=DEFAULT_CONFIG, overrides=None, battery_sizes=None):
    """Size and simulate every nanogrid on its own.

    pv_kw and load_kw have shape (n_steps, n_nanogrids). The cluster battery
    is the sum of the individual optima, and the LCOE is calculated for the
    cluster as a whole. Returns the result dictionary and the number of nanogrids.
    """
    n_steps, n_nanogrids = load_kw.shape
    hours_per_step = TIME_STEP_MIN / 60

    optimal_batteries   = np.zeros(n_nanogrids)     # optimal battery per nanogrid [kWh]
    total_diesel_fuel_L = 0.0
    total_diesel_kWh    = 0.0
    total_unmet_kWh     = 0.0
    total_curtailed_kWh = 0.0
    total_emissions_kg  = 0.0

    print(f"Running isolated scenario ({config['name']}): {n_nanogrids} nanogrid sweeps...")

    for h in range(n_nanogrids):
        _, optimal_h = run_battery_sweep(pv_kw[:, h], load_kw[:, h], verbose=False, config=config,
                                         overrides=overrides, battery_sizes=battery_sizes)
        optimal_batteries[h] = optimal_h["battery_kWh"]

        # Simulate again at the optimum to collect the energy flows.
        sim_h = run_simulation(pv_kw[:, h], load_kw[:, h], optimal_batteries[h], use_diesel=config["use_diesel"])
        total_diesel_fuel_L += sim_h["diesel_fuel_L"]
        total_diesel_kWh    += sim_h["annual_diesel_kWh"]
        total_unmet_kWh     += sim_h["annual_unmet_kWh"]
        total_curtailed_kWh += sim_h["annual_curtailed_kWh"]
        total_emissions_kg  += sim_h["diesel_emissions_kg"]
        print(f"      nanogrid {h+1:>2}/{n_nanogrids}: {optimal_batteries[h]:.1f} kWh")

    battery_kWh_total = optimal_batteries.sum()
    cluster_load_kWh  = load_kw.sum() * hours_per_step
    diesel_capacity_total = load_kw.max(axis=0).sum()   # one generator per nanogrid, sized to its peak load [kW]

    fleet = {"diesel_fuel_L": total_diesel_fuel_L}
    lcoe, ann_capex, ann_opex = calculate_lcoe(
        fleet, battery_kWh_total, n_units=n_nanogrids,
        battery_kWh_total=battery_kWh_total, annual_load_override=cluster_load_kWh,
        diesel_capacity_kW_total=diesel_capacity_total, config=config, overrides=overrides
    )
    renewable_fraction = (cluster_load_kWh - total_diesel_kWh - total_unmet_kWh) / cluster_load_kWh

    result = {
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
        # Sizing inputs for lcoe_breakdown()
        "n_units":              n_nanogrids,
        "diesel_capacity_kW":   diesel_capacity_total,
        "cable_length_m":       0.0,
    }

    return result, n_nanogrids


# ----------------------------------------------------------------------------
#  Interconnected operation
# ----------------------------------------------------------------------------

def pool_cluster_residual(pv_kw, load_kw):
    """Pool the surpluses and deficits that remain after local self-consumption.

    Each nanogrid first serves its own load with its own generation. Returns the
    pooled surplus and the pooled residual load as 1-D series [kW].
    """
    direct = np.minimum(pv_kw, load_kw)
    surplus_pv    = pv_kw   - direct
    residual_load = load_kw - direct

    pooled_pv   = surplus_pv.sum(axis=1)
    pooled_load = residual_load.sum(axis=1)

    return pooled_pv, pooled_load


def run_interconnected(pv_kw, load_kw, config=DEFAULT_CONFIG, overrides=None, battery_sizes=None):
    """Size and simulate the interconnected cluster.

    Generation and diesel generators stay per nanogrid; only the battery is
    shared. Sweeps the shared battery size and selects the lowest LCOE at full
    reliability. Returns the sweep DataFrame, the optimal row and the number
    of nanogrids.
    """
    n_nanogrids = load_kw.shape[1]
    hours_per_step = TIME_STEP_MIN / 60
    cluster_load_kWh = load_kw.sum() * hours_per_step
    diesel_capacity_total = load_kw.max(axis=0).sum()   # one generator per nanogrid, sized to its peak load [kW]
    cable_length_m = cable_length_routed(NANOGRID_COORDINATES) * CABLING_COSTS_ON

    pooled_pv, pooled_load = pool_cluster_residual(pv_kw, load_kw)

    if battery_sizes is None:
        battery_sizes = np.arange(BATTERY_MIN_KWH, BATTERY_MAX_KWH + BATTERY_STEP_KWH, BATTERY_STEP_KWH)

    rows = []
    print(f"Running interconnected sweep ({config['name']}): {len(battery_sizes)} simulations for N = {n_nanogrids}...")

    for battery_kWh in battery_sizes:
        sim = run_simulation(pooled_pv, pooled_load, battery_kWh, use_diesel=config["use_diesel"])
        lcoe, ann_capex, ann_opex = calculate_lcoe(
            sim, battery_kWh, n_units=n_nanogrids,
            battery_kWh_total=battery_kWh, annual_load_override=cluster_load_kWh,
            diesel_capacity_kW_total=diesel_capacity_total,
            cable_length_m=cable_length_m, config=config, overrides=overrides
        )
        renewable_fraction = (cluster_load_kWh - sim["annual_diesel_kWh"] - sim["annual_unmet_kWh"]) / cluster_load_kWh

        row = {
            "battery_kWh_total":    battery_kWh,
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
            # Sizing inputs for lcoe_breakdown()
            "n_units":              n_nanogrids,
            "diesel_capacity_kW":   diesel_capacity_total,
            "cable_length_m":       cable_length_m,
        }
        rows.append(row)

    df_sweep = pd.DataFrame(rows)

    # Full reliability: only sizes without unmet load are eligible.
    feasible = df_sweep[df_sweep["annual_unmet_kWh"] <= 1e-6]
    if len(feasible) > 0:
        optimal = feasible.loc[feasible["lcoe_eur_kWh"].idxmin()]
    else:
        optimal = df_sweep.loc[df_sweep["annual_unmet_kWh"].idxmin()]
        print("      WARNING: no shared battery size serves all load; selected the size with the least unmet load")

    if optimal["battery_kWh_total"] >= battery_sizes.max():
        print(f"      WARNING: the optimum lies at the upper bound of the sweep "
              f"({BATTERY_MAX_KWH} kWh). Raise BATTERY_MAX_KWH in configuration.py.")

    return df_sweep, optimal, n_nanogrids
