# ============================================================================
#  simulation.py
#  Rule-based dispatch of one nanogrid over the modelling window, for a given
#  battery capacity. Used for both the isolated nanogrids and the pooled cluster.
# ============================================================================

import numpy as np
from configuration import *


def run_simulation(pv_kw, load_kw, battery_capacity_kWh, use_diesel=True):
    """Simulate one nanogrid in time steps of TIME_STEP_MIN.

    Dispatch order per step: generation to load, surplus to the battery
    (excess is curtailed), battery to the remaining load, then diesel. Without
    diesel, load that cannot be served is recorded as unmet.

    pv_kw and load_kw are arrays in kW; returns time series and annual totals.
    """
    n_steps = len(load_kw)
    hours_per_step = TIME_STEP_MIN / 60

    soc_min_kWh = SOC_MIN * battery_capacity_kWh
    soc_max_kWh = SOC_MAX * battery_capacity_kWh

    soc_kWh      = np.zeros(n_steps)          # state of charge at the end of each step [kWh]
    pv_to_load   = np.zeros(n_steps)          # [kW]
    pv_to_batt   = np.zeros(n_steps)          # [kW]
    curtailed    = np.zeros(n_steps)          # [kW]
    batt_to_load = np.zeros(n_steps)          # [kW]
    diesel_kw    = np.zeros(n_steps)          # [kW]
    unmet_kw     = np.zeros(n_steps)          # [kW]

    current_soc = 0.5 * battery_capacity_kWh  # initial state of charge [kWh]

    for t in range(n_steps):
        pv = pv_kw[t]
        load = load_kw[t]

        # 1. Generation serves the load directly
        direct = min(pv, load)
        pv_to_load[t] = direct
        residual_load = load - direct
        surplus_pv = pv - direct

        # 2. Surplus charges the battery; the rest is curtailed
        if surplus_pv > 0:
            headroom_kWh = soc_max_kWh - current_soc
            charge_kWh = min(surplus_pv * hours_per_step, headroom_kWh / BATTERY_CHARGE_EFF)
            charge_kw = charge_kWh / hours_per_step
            pv_to_batt[t] = charge_kw
            current_soc += charge_kWh * BATTERY_CHARGE_EFF
            curtailed[t] = surplus_pv - charge_kw

        # 3. Battery serves the remaining load
        if residual_load > 0:
            available_kWh = (current_soc - soc_min_kWh) * BATTERY_DISCHARGE_EFF
            discharge_kWh = min(residual_load * hours_per_step, available_kWh)
            discharge_kw = discharge_kWh / hours_per_step
            batt_to_load[t] = discharge_kw
            current_soc -= discharge_kWh / BATTERY_DISCHARGE_EFF
            residual_load -= discharge_kw

        # 4. Diesel serves what is left; without diesel it is unmet
        if residual_load > 0:
            if use_diesel:
                diesel_kw[t] = residual_load
            else:
                unmet_kw[t] = residual_load
            residual_load = 0.0

        soc_kWh[t] = current_soc

    # Annual energy totals [kWh]
    annual_pv_to_load   = pv_to_load.sum()   * hours_per_step
    annual_pv_to_batt   = pv_to_batt.sum()   * hours_per_step
    annual_curtailed    = curtailed.sum()    * hours_per_step
    annual_batt_to_load = batt_to_load.sum() * hours_per_step
    annual_diesel_kWh   = diesel_kw.sum()    * hours_per_step
    annual_unmet_kWh    = unmet_kw.sum()     * hours_per_step
    annual_load_kWh     = load_kw.sum()      * hours_per_step

    lolp = (unmet_kw > 0).mean()              # loss of load probability: share of steps with unmet load

    diesel_fuel_L = annual_diesel_kWh / (DIESEL_KWH_L * DIESEL_EFFICIENCY)
    diesel_emissions = diesel_fuel_L * DIESEL_EMISSIONS_L      # [kg CO2]

    results = {
        # Time series
        "soc_kWh":       soc_kWh,
        "pv_to_load":    pv_to_load,
        "pv_to_batt":    pv_to_batt,
        "curtailed":     curtailed,
        "batt_to_load":  batt_to_load,
        "diesel_kw":     diesel_kw,
        "unmet_kw":      unmet_kw,

        # Annual totals [kWh]
        "annual_pv_to_load_kWh":   annual_pv_to_load,
        "annual_pv_to_batt_kWh":   annual_pv_to_batt,
        "annual_curtailed_kWh":    annual_curtailed,
        "annual_batt_to_load_kWh": annual_batt_to_load,
        "annual_diesel_kWh":       annual_diesel_kWh,
        "annual_unmet_kWh":        annual_unmet_kWh,
        "annual_load_kWh":         annual_load_kWh,

        "lolp": lolp,
        "diesel_fuel_L":       diesel_fuel_L,
        "diesel_emissions_kg": diesel_emissions,
    }

    return results
