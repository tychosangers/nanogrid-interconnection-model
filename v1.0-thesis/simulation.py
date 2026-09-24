import numpy as np
import pandas as pd
from configuration import *


def run_simulation(pv_kw, load_kw, battery_capacity_kWh, use_diesel=True):   # one-year, 15-min dispatch simulation for single nanogrid
    # use_diesel: True  -> diesel closes any gap, so load is always served (PV+diesel config)
    #             False -> no generator, so an uncovered gap is recorded as unmet load (PV-only config)

    n_steps = len(load_kw)
    hours_per_step = Time_step_min / 60                         # converts timestep from minutes to hours (0.25 h)

    # Usable battery window
    soc_min_kWh = SOC_min * battery_capacity_kWh
    soc_max_kWh = SOC_max * battery_capacity_kWh

    # Pre-allocating output arrays
    soc_kWh       = np.zeros(n_steps)                           # State of Charge at end of each timestep [kWh]
    pv_to_load    = np.zeros(n_steps)                           # PV power directly used to serve load [kW]
    pv_to_batt    = np.zeros(n_steps)                           # PV power used to charge battery [kW]
    curtailed     = np.zeros(n_steps)                           # PV power wasted (surplus after charging) [kW]
    batt_to_load  = np.zeros(n_steps)                           # Battery power discharged to serve load [kW]
    diesel_kw     = np.zeros(n_steps)                           # Diesel generator output to serve load [kW]
    unmet_kw      = np.zeros(n_steps)                           # Load that nothing could serve this step [kW]

    # Battery starts at midpoint of usable window
    current_soc = 0.5 * battery_capacity_kWh

    # Main dispatch loop
    for t in range(n_steps):

        pv    = pv_kw[t]
        load  = load_kw[t]

        # 1: PV serves load directly
        direct = min(pv, load)                                 # If PV produces more than load, load is picked
        pv_to_load[t] = direct
        residual_load = load - direct                          # remaining load after PV [kW]
        surplus_pv    = pv - direct                            # remaining PV after serving load [kW]

        # 2: Surplus PV charges the battery
        if surplus_pv > 0:
            headroom_kWh = soc_max_kWh - current_soc           # space remaining in battery [kWh]
            # PV side limited by headroom / charge efficiency, since some PV is lost as heat
            charge_kWh   = min(surplus_pv * hours_per_step, headroom_kWh / Battery_charge_eff)
            charge_kw    = charge_kWh / hours_per_step
            pv_to_batt[t] = charge_kw
            current_soc  += charge_kWh * Battery_charge_eff    # what actually lands in the battery
            curtailed[t] = surplus_pv - charge_kw              # Any surplus beyond battery capacity is curtailed [kW]

        # 3: Battery covers remaining load deficit
        if residual_load > 0:
            available_kWh = (current_soc - soc_min_kWh) * Battery_discharge_eff
            discharge_kWh = min(residual_load * hours_per_step, available_kWh)
            discharge_kw  = discharge_kWh / hours_per_step
            batt_to_load[t] = discharge_kw

            current_soc -= discharge_kWh / Battery_discharge_eff   # actual energy drawn from SoC
            residual_load -= discharge_kw

        # 4: Close the final gap. With diesel it is always covered; without diesel it stays unmet
        if residual_load > 0:
            if use_diesel:
                diesel_kw[t] = residual_load                   # generator fills the gap -> LOLP stays 0
            else:
                unmet_kw[t]  = residual_load                   # no generator -> this load is lost
            residual_load = 0.0

        soc_kWh[t] = current_soc                               # Save state of charge for this timestep

    # Total energy flows over the year [kWh]
    annual_pv_to_load   = pv_to_load.sum()   * hours_per_step
    annual_pv_to_batt   = pv_to_batt.sum()   * hours_per_step
    annual_curtailed    = curtailed.sum()    * hours_per_step
    annual_batt_to_load = batt_to_load.sum() * hours_per_step
    annual_diesel_kWh   = diesel_kw.sum()    * hours_per_step
    annual_unmet_kWh    = unmet_kw.sum()     * hours_per_step
    annual_load_kWh     = load_kw.sum()      * hours_per_step

    # Reliability: LOLP = share of timesteps that had any unmet load (0 means fully reliable)
    lolp = (unmet_kw > 0).mean()

    # Diesel fuel and emissions
    diesel_fuel_L      = annual_diesel_kWh / (Diesel_kWh_L * Diesel_efficiency)
    diesel_emissions   = diesel_fuel_L * Diesel_Emissions_L     # kg CO2

    results = {
        # Time-series arrays
        "soc_kWh":       soc_kWh,
        "pv_to_load":    pv_to_load,
        "pv_to_batt":    pv_to_batt,
        "curtailed":     curtailed,
        "batt_to_load":  batt_to_load,
        "diesel_kw":     diesel_kw,
        "unmet_kw":      unmet_kw,

        # Annual energy totals [kWh]
        "annual_pv_to_load_kWh":   annual_pv_to_load,
        "annual_pv_to_batt_kWh":   annual_pv_to_batt,
        "annual_curtailed_kWh":    annual_curtailed,
        "annual_batt_to_load_kWh": annual_batt_to_load,
        "annual_diesel_kWh":       annual_diesel_kWh,
        "annual_unmet_kWh":        annual_unmet_kWh,
        "annual_load_kWh":         annual_load_kWh,

        # Reliability
        "lolp": lolp,

        # Fuel and emissions
        "diesel_fuel_L":     diesel_fuel_L,
        "diesel_emissions_kg": diesel_emissions,
    }

    return results
