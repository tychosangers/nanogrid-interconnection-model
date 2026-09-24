import numpy as np
import pandas as pd
from configuration import *
from simulation import run_simulation
from cabling import cable_length_routed


def capital_recovery_factor(discount_rate, lifetime):                   # Annualises investment over lifetime 

    i = discount_rate
    N = lifetime
    
    return (i * (1 + i) ** N) / ((1 + i) ** N - 1)

def annual_cost_terms(diesel_fuel_L, battery_kWh, n_units=1, diesel_kW=0.0,      # Every cost that enters the LCOE, one entry per cost term [€/year]
                      cable_length_m=0.0, config=DEFAULT_CONFIG, overrides=None):
    """Return the yearly cost of the system, split by technology, in EUR per year.

    This is the single place where the cost arithmetic lives. calculate_lcoe()
    simply adds these terms up and divides by the annual load, and the LCOE
    breakdown plot reads exactly the same terms, so the bars in that plot can
    never disagree with the LCOE number printed in the summary table.

    Two dictionaries are returned. A dictionary is a {name: value} lookup table:
    capex_terms holds the investment costs already spread over their lifetime
    with the capital recovery factor, and opex_terms holds the running costs.
    Terms that this configuration does not pay for (wind when wind is off,
    diesel when diesel is off, the cable in the isolated case) simply come out
    as zero, because their switch multiplies them by False, which counts as 0.
    """
    diesel_on = config["use_diesel"]                                    # If False, all diesel cost terms drop to zero
    wind_on   = config["use_wind"]                                      # If False, all wind cost terms drop to zero

    # --- optional parameter overrides (used by scenario_sensitivity.py) ---
    # Each local below falls back to its configuration.py constant unless the
    # caller passes a replacement in the overrides dict. dict.get(key, default)
    # returns the stored value when the key is present, otherwise the default.
    # With overrides left as None every local is the normal constant, so all
    # existing runs stay byte for byte identical.
    overrides = overrides or {}
    discount_rate       = overrides.get("Discount_rate",         Discount_rate)
    pv_capex_eur_kW     = overrides.get("PV_capex_eur_kW",       PV_capex_eur_kW)
    battery_capex_kWh   = overrides.get("Battery_capex_eur_kWh", Battery_capex_eur_kWh)
    diesel_capex_eur_kW = overrides.get("Diesel_capex_eur_kW",   Diesel_capex_eur_kW)
    diesel_price_eur_L  = overrides.get("Diesel_price_eur_L",    Diesel_price_eur_L)
    carbon_price_eur_kg = overrides.get("Carbon_price_eur_kg",   Carbon_price_eur_kg)
    wind_capex_eur_kW   = overrides.get("Wind_capex_eur_kW",     Wind_capex_eur_kW)
    wind_opex_eur_kW_yr = overrides.get("Wind_opex_eur_kW_yr",   Wind_opex_eur_kW_yr)
    cable_capex_eur_m   = overrides.get("Cable_capex_eur_m",     Cable_capex_eur_m)
    apply_carbon_tax    = overrides.get("Apply_carbon_tax",      Apply_carbon_tax)
    pv_capacity_kW      = overrides.get("PV_capacity_kW",        PV_capacity_kW)   # lets the PV sensitivity move PV capacity on the cost side

    # Capital recovery factors per technology
    crf_pv      = capital_recovery_factor(discount_rate, PV_Lifetime)
    crf_battery = capital_recovery_factor(discount_rate, Battery_Lifetime)
    crf_diesel  = capital_recovery_factor(discount_rate, Diesel_Lifetime)
    crf_cable   = capital_recovery_factor(discount_rate, Cable_Lifetime)
    crf_wind    = capital_recovery_factor(discount_rate, Wind_Lifetime)

    # Annualised CAPEX per technology (PV and wind scale with n_units; battery and diesel are passed in as totals)
    capex_terms = {
        "PV capex":      crf_pv      * pv_capex_eur_kW     * pv_capacity_kW   * n_units,             # € /yr
        "Wind capex":    crf_wind    * wind_capex_eur_kW   * Wind_capacity_kW * n_units * wind_on,   # € /yr (one turbine per nanogrid; zero if no wind)
        "Diesel capex":  crf_diesel  * diesel_capex_eur_kW * diesel_kW * diesel_on,                  # € /yr (zero if no diesel)
        "Cable capex":   crf_cable   * cable_capex_eur_m   * cable_length_m,                         # € /yr (zero in the isolated case)
        "Battery capex": crf_battery * battery_capex_kWh   * battery_kWh,                            # € /yr
    }

    carbon_on = apply_carbon_tax and diesel_on                          # carbon tax only bites when diesel is actually costed

    # Annual OPEX per technology
    opex_terms = {
        "PV O&M":      PV_opex_eur_kW_yr      * pv_capacity_kW   * n_units,                          # € /yr
        "Wind O&M":    wind_opex_eur_kW_yr    * Wind_capacity_kW * n_units * wind_on,                # € /yr (zero if no wind)
        "Diesel O&M":  Diesel_opex_eur_kW_yr  * diesel_kW * diesel_on,                               # € /yr (zero if no diesel)
        "Diesel fuel": diesel_fuel_L * diesel_price_eur_L * diesel_on,                               # € /yr (zero if no diesel)
        "Carbon tax":  diesel_fuel_L * Diesel_Emissions_L * carbon_price_eur_kg * carbon_on,         # € /yr, EU ETS-style tax on diesel CO2
        "Battery O&M": Battery_opex_eur_kWh_yr * battery_kWh,                                        # € /yr
    }

    return capex_terms, opex_terms


def lcoe_breakdown(result, config, overrides=None):                     # The LCOE of one scenario, split into its cost terms [€/kWh]
    """Split a finished scenario result into the cost terms that make up its LCOE.

    Feeds the stacked-bar figure. The values are in EUR/kWh and they add up to
    result["lcoe_eur_kWh"], because they are the same terms calculate_lcoe()
    summed, divided by the same annual load. The returned dictionary is built in
    a fixed stacking order (bottom of the bar first, battery capex last, so it
    ends up on top): from Python 3.7 onwards a dictionary keeps the order in
    which its keys were inserted, so the plot can just loop over it.

    result must carry the three sizing keys the scenarios now store: n_units,
    diesel_capacity_kW and cable_length_m (zero for the isolated fleet).
    """
    capex_terms, opex_terms = annual_cost_terms(
        result["diesel_fuel_L"], result["battery_kWh_total"],
        n_units=result["n_units"], diesel_kW=result["diesel_capacity_kW"],
        cable_length_m=result["cable_length_m"], config=config, overrides=overrides
    )

    all_terms = {}                                                      # merge the two dictionaries into one lookup table
    all_terms.update(capex_terms)                                       # .update() copies every {name: value} pair into all_terms
    all_terms.update(opex_terms)

    load_kWh = result["cluster_load_kWh"]                               # the same denominator calculate_lcoe used
    stack_order = ["PV capex", "PV O&M",                                # bottom of the bar
                   "Wind capex", "Wind O&M",
                   "Diesel capex", "Diesel O&M", "Diesel fuel", "Carbon tax",
                   "Cable capex",
                   "Battery O&M", "Battery capex"]                      # top of the bar

    return {name: all_terms[name] / load_kWh for name in stack_order}   # dictionary comprehension: same keys, values converted to €/kWh


def calculate_lcoe(results, battery_capacity_kWh,                       #  LCOE = (sum of annualised CAPEX + annual OPEX) / annual load served
                   n_units=1, battery_kWh_total=None, annual_load_override=None,
                   diesel_capacity_kW_total=None, cable_length_m=0.0, config=DEFAULT_CONFIG,
                   overrides=None):
    # n_units                  : number of PV+diesel sets (1 = standalone, N = cluster with per-household assets)
    # battery_kWh_total        : if given, use this as the battery size instead of battery_capacity_kWh (e.g. one shared cluster battery)
    # annual_load_override     : if given, divide by this load instead of results["annual_load_kWh"] (e.g. cluster total load)
    # diesel_capacity_kW_total : if given, total genset capacity to cost (e.g. sum of per-house peak loads); else falls back to the flat nameplate
    # cable_length_m           : interconnection cable length [m]; zero in the isolated case (no cable links the nanogrids)
    # config                   : configuration bundle (e.g. CONFIG_PV_DIESEL or CONFIG_PV_ONLY); switches diesel and wind on/off

    battery_kWh = battery_kWh_total if battery_kWh_total is not None else battery_capacity_kWh   # Shared battery total, else the swept single size
    diesel_kW   = diesel_capacity_kW_total if diesel_capacity_kW_total is not None else Diesel_capacity_kW * n_units   # Peak-sized when given, else flat nameplate

    # All the cost arithmetic now lives in annual_cost_terms() below, so the LCOE and the
    # LCOE breakdown plot can never drift apart. sum(d.values()) adds up every value in a
    # dictionary, so these two lines reproduce the old annualised_capex / annual_opex exactly.
    capex_terms, opex_terms = annual_cost_terms(
        results["diesel_fuel_L"], battery_kWh, n_units=n_units, diesel_kW=diesel_kW,
        cable_length_m=cable_length_m, config=config, overrides=overrides
    )
    annualised_capex = sum(capex_terms.values())                        # €/year
    annual_opex      = sum(opex_terms.values())                         # €/year

    # Annual load (demand). Unmet load is zero at any feasible optimum, and always with diesel on.
    annual_load = annual_load_override if annual_load_override is not None else results["annual_load_kWh"]

    # LCOE
    lcoe = (annualised_capex + annual_opex) / annual_load               # €/kWh

    return lcoe, annualised_capex, annual_opex


def run_battery_sweep(pv_kw, load_kw, verbose=True, config=DEFAULT_CONFIG, overrides=None, battery_sizes=None):    # config selects the system bundle (diesel on/off etc.)

    # battery_sizes lets a caller (e.g. the PV sensitivity) pass a coarser grid to run
    # faster. Left as None it falls back to the configuration.py sweep, so every existing
    # call behaves exactly as before.
    if battery_sizes is None:
        battery_sizes = np.arange(Battery_min_kWh, Battery_max_kWh + Battery_step_kWh, Battery_step_kWh)    # Ensures last step is included

    diesel_capacity_kW = load_kw.max()                                 # Genset sized to this house's own peak load [kW]

    rows = []                                                           # An empty list to collect results

    if verbose:
        print(f"Running battery sweep: {len(battery_sizes)} simulations...")

    for battery_kWh in battery_sizes:

        sim = run_simulation(pv_kw, load_kw, battery_kWh, use_diesel=config["use_diesel"])   # config decides whether diesel may close gaps
        lcoe, ann_capex, ann_opex = calculate_lcoe(sim, battery_kWh, diesel_capacity_kW_total=diesel_capacity_kW, config=config, overrides=overrides)    # Diesel sized to the house peak; config switches diesel cost on/off
        renewable_fraction = ((sim["annual_pv_to_load_kWh"] + sim["annual_batt_to_load_kWh"])/ sim["annual_load_kWh"])

        # Collect all key outputs for this battery size into one dictionary
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

    # Convert the list of dictionaries into a pandas DataFrame.Each dictionary becomes one row; the keys become column names.
    df_sweep = pd.DataFrame(rows)

    # Keep only battery sizes that serve all the load. With diesel on every size does, so this is a no-op;
    # with diesel off it stops the sweep choosing a cheap size that just leaves load unmet.
    feasible = df_sweep[df_sweep["annual_unmet_kWh"] <= 1e-6]
    if len(feasible) > 0:
        optimal_row = feasible.loc[feasible["lcoe_eur_kWh"].idxmin()]
    else:
        optimal_row = df_sweep.loc[df_sweep["annual_unmet_kWh"].idxmin()]
        if verbose:
            print("      WARNING: no battery size serves all load without diesel; picked the lowest-unmet size")

    return df_sweep, optimal_row


def decompose_lcoe_gap(iso_result, inter_optimal, config):      # Split the LCOE reduction into battery, fuel, cable and carbon parts
    """The LCOE gap between scenarios comes only from terms that DIFFER.
    PV and wind (N units each), and diesel fixed O&M (N gensets), are identical
    in both topologies, so they cancel. What differs: battery size (CAPEX+OPEX),
    diesel fuel burnt, the interconnection cable (interconnected only, and only
    if CABLING_COSTS_ON), and, if the toggle is on, the carbon tax on that fuel.
    """
    load_kWh = iso_result["cluster_load_kWh"]                   # same load in both scenarios

    # Battery annual cost difference (isolated minus interconnected)
    crf_batt = capital_recovery_factor(Discount_rate, Battery_Lifetime)
    batt_cost = lambda kWh: crf_batt * Battery_capex_eur_kWh * kWh + Battery_opex_eur_kWh_yr * kWh   # €/yr for a given total kWh
    batt_diff = batt_cost(iso_result["battery_kWh_total"]) - batt_cost(inter_optimal["battery_kWh_total"])

    # Diesel fuel annual cost difference (isolated minus interconnected)
    fuel_l_diff = (iso_result["diesel_fuel_L"] - inter_optimal["diesel_fuel_L"]) * config["use_diesel"]
    fuel_diff   = fuel_l_diff * Diesel_price_eur_L

    # Cable annual cost difference: the isolated fleet has no cable, the interconnected one does, so this term is purely a
    # cost on the interconnected side (it works against interconnection). Zero when CABLING_COSTS_ON is False.
    crf_cable    = capital_recovery_factor(Discount_rate, Cable_Lifetime)
    cable_annual = crf_cable * Cable_capex_eur_m * cable_length_routed(Tinos_coordinates) * CABLING_COSTS_ON
    cable_diff   = 0.0 - cable_annual

    # Carbon tax annual cost difference (isolated minus interconnected); zero unless the toggle is on.
    carbon_diff = fuel_l_diff * Diesel_Emissions_L * Carbon_price_eur_kg * Apply_carbon_tax

    # Convert each annual-€ difference into €/kWh (LCOE units)
    batt_lcoe_part   = batt_diff   / load_kWh
    fuel_lcoe_part   = fuel_diff   / load_kWh
    cable_lcoe_part  = cable_diff  / load_kWh
    carbon_lcoe_part = carbon_diff / load_kWh
    total_gap        = iso_result["lcoe_eur_kWh"] - inter_optimal["lcoe_eur_kWh"]   # what we should reproduce

    return batt_lcoe_part, fuel_lcoe_part, cable_lcoe_part, carbon_lcoe_part, total_gap


def cable_breakeven_length(iso_result, inter_optimal, config, horizon_years):   # Longest cable that still breaks even within the horizon
    load_kWh = iso_result["cluster_load_kWh"]
    batt_part, fuel_part, cable_part, carbon_part, _ = decompose_lcoe_gap(iso_result, inter_optimal, config)
    annual_benefit = (batt_part + fuel_part + carbon_part) * load_kWh           # €/yr that interconnecting saves, before paying for any cable

    if annual_benefit <= 0:
        return annual_benefit, None, None                                       # no benefit even with a free cable -> no break-even length

    crf_h = capital_recovery_factor(Discount_rate, horizon_years)               # spreads a one-off cable cost over the horizon at the discount rate
    length_discounted = annual_benefit / (crf_h * Cable_capex_eur_m)            # cable amortised over the horizon
    length_simple     = horizon_years * annual_benefit / Cable_capex_eur_m      # undiscounted payback view (cumulative saving = cable cost)
    return annual_benefit, length_discounted, length_simple


def grid_extension_breakeven(result, grid_price, lifetime_years):   # Most a main-grid extension could cost while the nanogrid still wins
    # Compares serving the load off-grid (at this scenario's LCOE) against extending the main grid and buying
    # electricity at grid_price. Over the project lifetime the two break even when the extension's one-off capex
    # equals the energy-cost gap, so a costlier extension means the nanogrid is the cheaper choice.
    load_kWh = result["cluster_load_kWh"]
    lcoe     = result["lcoe_eur_kWh"]

    nanogrid_lifetime_cost = lcoe       * load_kWh * lifetime_years   # € to serve the load off-grid over the lifetime
    grid_energy_lifetime   = grid_price * load_kWh * lifetime_years   # € of grid electricity for the same load
    breakeven_extension_capex = nanogrid_lifetime_cost - grid_energy_lifetime   # headroom for the one-off extension (negative = nanogrid wins outright)
    return nanogrid_lifetime_cost, grid_energy_lifetime, breakeven_extension_capex


def sweep_cost_components(df_sweep, config):   # Battery and diesel-fuel contributions to LCOE along a battery sweep [€/kWh]
    # Mirrors the battery, fuel and carbon terms in calculate_lcoe, per row of the sweep. The remaining
    # "fixed" part (PV, diesel fixed O&M, cable, wind) is whatever is left of the LCOE after these, and it
    # does not change with battery size, so a plot can derive it as lcoe - batt - fuel - carbon.
    crf_batt = capital_recovery_factor(Discount_rate, Battery_Lifetime)
    load = df_sweep["cluster_load_kWh"]

    batt_eur_kWh = (crf_batt * Battery_capex_eur_kWh * df_sweep["battery_kWh_total"]
                    + Battery_opex_eur_kWh_yr * df_sweep["battery_kWh_total"]) / load
    fuel_eur_kWh = (df_sweep["diesel_fuel_L"] * Diesel_price_eur_L * config["use_diesel"]) / load
    carbon_on    = Apply_carbon_tax and config["use_diesel"]
    carbon_eur_kWh = (df_sweep["diesel_fuel_L"] * Diesel_Emissions_L * Carbon_price_eur_kg * carbon_on) / load
    return batt_eur_kWh, fuel_eur_kWh, carbon_eur_kWh