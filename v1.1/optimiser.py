# ============================================================================
#  optimiser.py
#  Cost model and battery sizing: the LCOE and its cost terms, the battery
#  sweep, and the decomposition of the LCOE gap between the two topologies.
# ============================================================================

import numpy as np
import pandas as pd
from configuration import *
from simulation import run_simulation
from cabling import cable_length_routed


def capital_recovery_factor(discount_rate, lifetime):
    """Factor that converts an investment into an equal annual payment."""
    i = discount_rate
    N = lifetime
    return (i * (1 + i) ** N) / ((1 + i) ** N - 1)


def annual_cost_terms(diesel_fuel_L, battery_kWh, n_units=1, diesel_kW=0.0,
                      cable_length_m=0.0, config=DEFAULT_CONFIG, overrides=None):
    """Annual cost of the system per cost term [EUR/year].

    Returns two dictionaries: annualised investment costs (capex_terms) and
    running costs (opex_terms). Terms of technologies that are switched off in
    the configuration are zero.

    overrides is an optional dictionary that replaces parameter values from
    configuration.py, for example {"DISCOUNT_RATE": 0.05}.
    """
    diesel_on = config["use_diesel"]
    wind_on   = config["use_wind"]

    overrides = overrides or {}
    discount_rate       = overrides.get("DISCOUNT_RATE",         DISCOUNT_RATE)
    pv_capex_eur_kW     = overrides.get("PV_CAPEX_EUR_KW",       PV_CAPEX_EUR_KW)
    battery_capex_kWh   = overrides.get("BATTERY_CAPEX_EUR_KWH", BATTERY_CAPEX_EUR_KWH)
    diesel_capex_eur_kW = overrides.get("DIESEL_CAPEX_EUR_KW",   DIESEL_CAPEX_EUR_KW)
    diesel_price_eur_L  = overrides.get("DIESEL_PRICE_EUR_L",    DIESEL_PRICE_EUR_L)
    carbon_price_eur_kg = overrides.get("CARBON_PRICE_EUR_KG",   CARBON_PRICE_EUR_KG)
    wind_capex_eur_kW   = overrides.get("WIND_CAPEX_EUR_KW",     WIND_CAPEX_EUR_KW)
    wind_opex_eur_kW_yr = overrides.get("WIND_OPEX_EUR_KW_YR",   WIND_OPEX_EUR_KW_YR)
    cable_capex_eur_m   = overrides.get("CABLE_CAPEX_EUR_M",     CABLE_CAPEX_EUR_M)
    apply_carbon_tax    = overrides.get("APPLY_CARBON_TAX",      APPLY_CARBON_TAX)
    pv_capacity_kW      = overrides.get("PV_CAPACITY_KW",        PV_CAPACITY_KW)

    crf_pv      = capital_recovery_factor(discount_rate, PV_LIFETIME)
    crf_battery = capital_recovery_factor(discount_rate, BATTERY_LIFETIME)
    crf_diesel  = capital_recovery_factor(discount_rate, DIESEL_LIFETIME)
    crf_cable   = capital_recovery_factor(discount_rate, CABLE_LIFETIME)
    crf_wind    = capital_recovery_factor(discount_rate, WIND_LIFETIME)

    # PV and wind are per nanogrid (n_units); battery and diesel are passed in as totals.
    capex_terms = {
        "PV capex":      crf_pv      * pv_capex_eur_kW     * pv_capacity_kW   * n_units,
        "Wind capex":    crf_wind    * wind_capex_eur_kW   * WIND_CAPACITY_KW * n_units * wind_on,
        "Diesel capex":  crf_diesel  * diesel_capex_eur_kW * diesel_kW * diesel_on,
        "Cable capex":   crf_cable   * cable_capex_eur_m   * cable_length_m,
        "Battery capex": crf_battery * battery_capex_kWh   * battery_kWh,
    }

    carbon_on = apply_carbon_tax and diesel_on

    opex_terms = {
        "PV O&M":      PV_OPEX_EUR_KW_YR      * pv_capacity_kW   * n_units,
        "Wind O&M":    wind_opex_eur_kW_yr    * WIND_CAPACITY_KW * n_units * wind_on,
        "Diesel O&M":  DIESEL_OPEX_EUR_KW_YR  * diesel_kW * diesel_on,
        "Diesel fuel": diesel_fuel_L * diesel_price_eur_L * diesel_on,
        "Carbon tax":  diesel_fuel_L * DIESEL_EMISSIONS_L * carbon_price_eur_kg * carbon_on,
        "Battery O&M": BATTERY_OPEX_EUR_KWH_YR * battery_kWh,
    }

    return capex_terms, opex_terms


def lcoe_breakdown(result, config, overrides=None):
    """LCOE of a scenario result split into its cost terms [EUR/kWh].

    The terms sum to result["lcoe_eur_kWh"] and are returned in stacking
    order for a bar chart, from PV capex at the bottom to battery capex on top.
    """
    capex_terms, opex_terms = annual_cost_terms(
        result["diesel_fuel_L"], result["battery_kWh_total"],
        n_units=result["n_units"], diesel_kW=result["diesel_capacity_kW"],
        cable_length_m=result["cable_length_m"], config=config, overrides=overrides
    )

    all_terms = {}
    all_terms.update(capex_terms)
    all_terms.update(opex_terms)

    load_kWh = result["cluster_load_kWh"]
    stack_order = ["PV capex", "PV O&M",
                   "Wind capex", "Wind O&M",
                   "Diesel capex", "Diesel O&M", "Diesel fuel", "Carbon tax",
                   "Cable capex",
                   "Battery O&M", "Battery capex"]

    return {name: all_terms[name] / load_kWh for name in stack_order}


def calculate_lcoe(results, battery_capacity_kWh,
                   n_units=1, battery_kWh_total=None, annual_load_override=None,
                   diesel_capacity_kW_total=None, cable_length_m=0.0, config=DEFAULT_CONFIG,
                   overrides=None):
    """LCOE = (annualised capex + annual opex) / annual load [EUR/kWh].

    n_units:                  number of nanogrids, each with its own PV (and wind)
    battery_kWh_total:        battery size to cost, instead of battery_capacity_kWh
    annual_load_override:     load to divide by, instead of results["annual_load_kWh"]
    diesel_capacity_kW_total: total generator capacity to cost; default DIESEL_CAPACITY_KW per unit
    cable_length_m:           interconnection cable length, zero for isolated nanogrids

    Returns the LCOE, the annualised capex and the annual opex.
    """
    battery_kWh = battery_kWh_total if battery_kWh_total is not None else battery_capacity_kWh
    diesel_kW = diesel_capacity_kW_total if diesel_capacity_kW_total is not None else DIESEL_CAPACITY_KW * n_units

    capex_terms, opex_terms = annual_cost_terms(
        results["diesel_fuel_L"], battery_kWh, n_units=n_units, diesel_kW=diesel_kW,
        cable_length_m=cable_length_m, config=config, overrides=overrides
    )
    annualised_capex = sum(capex_terms.values())     # [EUR/year]
    annual_opex      = sum(opex_terms.values())      # [EUR/year]

    annual_load = annual_load_override if annual_load_override is not None else results["annual_load_kWh"]

    lcoe = (annualised_capex + annual_opex) / annual_load

    return lcoe, annualised_capex, annual_opex


def run_battery_sweep(pv_kw, load_kw, verbose=True, config=DEFAULT_CONFIG, overrides=None, battery_sizes=None):
    """Simulate one nanogrid for each battery size and select the optimum.

    The optimum is the size with the lowest LCOE among the sizes that serve
    all load. battery_sizes defaults to the sweep set in configuration.py.
    The diesel generator is sized to the peak load of this nanogrid.

    Returns the full sweep as a DataFrame and the optimal row.
    """
    if battery_sizes is None:
        battery_sizes = np.arange(BATTERY_MIN_KWH, BATTERY_MAX_KWH + BATTERY_STEP_KWH, BATTERY_STEP_KWH)

    diesel_capacity_kW = load_kw.max()

    rows = []

    if verbose:
        print(f"Running battery sweep: {len(battery_sizes)} simulations...")

    for battery_kWh in battery_sizes:
        sim = run_simulation(pv_kw, load_kw, battery_kWh, use_diesel=config["use_diesel"])
        lcoe, ann_capex, ann_opex = calculate_lcoe(sim, battery_kWh, diesel_capacity_kW_total=diesel_capacity_kW,
                                                   config=config, overrides=overrides)
        renewable_fraction = ((sim["annual_pv_to_load_kWh"] + sim["annual_batt_to_load_kWh"]) / sim["annual_load_kWh"])

        row = {
            "battery_kWh":          battery_kWh,
            "lcoe_eur_kWh":         lcoe,
            "annualised_capex_eur": ann_capex,
            "annual_opex_eur":      ann_opex,
            "annual_load_kWh":      sim["annual_load_kWh"],
            "annual_diesel_kWh":    sim["annual_diesel_kWh"],
            "annual_unmet_kWh":     sim["annual_unmet_kWh"],
            "annual_curtailed_kWh": sim["annual_curtailed_kWh"],
            "diesel_fuel_L":        sim["diesel_fuel_L"],
            "diesel_emissions_kg":  sim["diesel_emissions_kg"],
            "renewable_fraction":   renewable_fraction,
        }
        rows.append(row)

    df_sweep = pd.DataFrame(rows)

    # Full reliability: only sizes without unmet load are eligible.
    feasible = df_sweep[df_sweep["annual_unmet_kWh"] <= 1e-6]
    if len(feasible) > 0:
        optimal_row = feasible.loc[feasible["lcoe_eur_kWh"].idxmin()]
    else:
        optimal_row = df_sweep.loc[df_sweep["annual_unmet_kWh"].idxmin()]
        if verbose:
            print("      WARNING: no battery size serves all load; selected the size with the least unmet load")

    return df_sweep, optimal_row


def decompose_lcoe_gap(iso_result, inter_optimal, config):
    """Split the LCOE gap between isolated and interconnected operation [EUR/kWh].

    Only cost terms that differ between the topologies contribute: the battery,
    diesel fuel, the cable (if CABLING_COSTS_ON) and the carbon price (if
    APPLY_CARBON_TAX). Returns the battery, fuel, cable and carbon parts and
    the total gap.
    """
    load_kWh = iso_result["cluster_load_kWh"]

    crf_batt = capital_recovery_factor(DISCOUNT_RATE, BATTERY_LIFETIME)
    batt_cost = lambda kWh: crf_batt * BATTERY_CAPEX_EUR_KWH * kWh + BATTERY_OPEX_EUR_KWH_YR * kWh
    batt_diff = batt_cost(iso_result["battery_kWh_total"]) - batt_cost(inter_optimal["battery_kWh_total"])

    fuel_l_diff = (iso_result["diesel_fuel_L"] - inter_optimal["diesel_fuel_L"]) * config["use_diesel"]
    fuel_diff   = fuel_l_diff * DIESEL_PRICE_EUR_L

    # The cable is a cost of the interconnected cluster only.
    crf_cable    = capital_recovery_factor(DISCOUNT_RATE, CABLE_LIFETIME)
    cable_annual = crf_cable * CABLE_CAPEX_EUR_M * cable_length_routed(NANOGRID_COORDINATES) * CABLING_COSTS_ON
    cable_diff   = 0.0 - cable_annual

    carbon_diff = fuel_l_diff * DIESEL_EMISSIONS_L * CARBON_PRICE_EUR_KG * APPLY_CARBON_TAX

    batt_lcoe_part   = batt_diff   / load_kWh
    fuel_lcoe_part   = fuel_diff   / load_kWh
    cable_lcoe_part  = cable_diff  / load_kWh
    carbon_lcoe_part = carbon_diff / load_kWh
    total_gap        = iso_result["lcoe_eur_kWh"] - inter_optimal["lcoe_eur_kWh"]

    return batt_lcoe_part, fuel_lcoe_part, cable_lcoe_part, carbon_lcoe_part, total_gap


def cable_breakeven_length(iso_result, inter_optimal, config, horizon_years):
    """Longest cable for which interconnection still breaks even within horizon_years.

    Returns the annual saving before cable costs [EUR/year], the break-even
    length with discounting and the simple-payback length [m]. The lengths are
    None when interconnection gives no saving.
    """
    load_kWh = iso_result["cluster_load_kWh"]
    batt_part, fuel_part, cable_part, carbon_part, _ = decompose_lcoe_gap(iso_result, inter_optimal, config)
    annual_benefit = (batt_part + fuel_part + carbon_part) * load_kWh

    if annual_benefit <= 0:
        return annual_benefit, None, None

    crf_h = capital_recovery_factor(DISCOUNT_RATE, horizon_years)
    length_discounted = annual_benefit / (crf_h * CABLE_CAPEX_EUR_M)
    length_simple     = horizon_years * annual_benefit / CABLE_CAPEX_EUR_M
    return annual_benefit, length_discounted, length_simple


def grid_extension_breakeven(result, grid_price, lifetime_years):
    """Maximum grid extension cost at which the off-grid cluster remains cheaper.

    Compares the lifetime cost of serving the load at the scenario LCOE with
    buying the same energy from the grid at grid_price. Returns both lifetime
    costs and their difference [EUR]; a negative difference means the
    off-grid cluster is cheaper even without extension costs.
    """
    load_kWh = result["cluster_load_kWh"]
    lcoe     = result["lcoe_eur_kWh"]

    nanogrid_lifetime_cost = lcoe       * load_kWh * lifetime_years
    grid_energy_lifetime   = grid_price * load_kWh * lifetime_years
    breakeven_extension_capex = nanogrid_lifetime_cost - grid_energy_lifetime
    return nanogrid_lifetime_cost, grid_energy_lifetime, breakeven_extension_capex


def sweep_cost_components(df_sweep, config):
    """Battery, fuel and carbon contributions to the LCOE per row of a sweep [EUR/kWh].

    The remaining part of the LCOE (PV, wind, diesel O&M and cable) does not
    depend on battery size.
    """
    crf_batt = capital_recovery_factor(DISCOUNT_RATE, BATTERY_LIFETIME)
    load = df_sweep["cluster_load_kWh"]

    batt_eur_kWh = (crf_batt * BATTERY_CAPEX_EUR_KWH * df_sweep["battery_kWh_total"]
                    + BATTERY_OPEX_EUR_KWH_YR * df_sweep["battery_kWh_total"]) / load
    fuel_eur_kWh = (df_sweep["diesel_fuel_L"] * DIESEL_PRICE_EUR_L * config["use_diesel"]) / load
    carbon_on    = APPLY_CARBON_TAX and config["use_diesel"]
    carbon_eur_kWh = (df_sweep["diesel_fuel_L"] * DIESEL_EMISSIONS_L * CARBON_PRICE_EUR_KG * carbon_on) / load
    return batt_eur_kWh, fuel_eur_kWh, carbon_eur_kWh
