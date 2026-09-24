# ============================================================================
#  scenario_carbon_tax.py
#  Carbon-price sensitivity figure for the diesel configuration (PV-Diesel-BESS).
#
#  WHY THIS FILE EXISTS
#  Diesel is the only modelled generator that emits at the point of use, so an
#  ETS-style carbon price only touches the PV-Diesel-BESS configuration. Because
#  interconnection lets the four lodges share storage and run their gensets less,
#  the isolated fleet burns more diesel than the interconnected cluster, so a
#  carbon price falls more heavily on the isolated case. The expectation is that
#  the interconnection benefit WIDENS as the carbon price rises. This file sweeps
#  the price and draws that, in two panels:
#     left  : both LCOEs (isolated vs interconnected) with the benefit shaded
#             between them, so the widening gap is visible directly.
#     right : the headline, the benefit as a percentage LCOE reduction, one line.
#
#  UNITS
#  The model prices carbon per kilogram (Carbon_price_eur_kg in configuration.py).
#  The public ETS debate is per tonne, so this figure sweeps in EUR/tonne on the
#  x-axis and converts with price_per_tonne / 1000 before handing it to the model.
#  For reference, 80 EUR/tonne = 0.08 EUR/kg, the value set in configuration.py.
#
#  TOGGLE
#  As requested, the figure is only built when Apply_carbon_tax is True in
#  configuration.py. Flip that toggle on and run this file. The sweep still forces
#  the tax on inside its own runs, so the x-axis always spans the full range; the
#  toggle only decides whether the file produces the figure at all. If you would
#  rather this file always run, delete the guard in the __main__ block at the end.
#
#  BASE CASE
#  Every non-carbon parameter takes its configuration.py value (discount rate 8%,
#  etc.), so this figure sits on the same base case as the sensitivity appendix.
# ============================================================================

import os
import matplotlib.pyplot as plt

from configuration import *                                  # constants: CONFIG_PV_DIESEL, COLOUR_*, Carbon_price_eur_kg, MODEL_WINDOW_*, SOLAR_FILE
from pv_profile import build_pv_for_window
from operation import run_isolated, run_interconnected       # the two operating modes: isolated fleet and interconnected cluster
from load_profile import prepare_model_loads


# ============================================================================
#  Sweep settings
# ============================================================================
CARBON_SWEEP_EUR_TONNE = (65.0, 95.0)      # x-axis span, EUR/tonne CO2 (0 = no tax)
CARBON_SWEEP_POINTS    = 7                # carbon prices sampled; 16 gives a clean 10 EUR/tonne step. Lower it if a run feels slow.
CARBON_REFERENCE_LINES = {                 # vertical markers on the figure, EUR/tonne CO2
    "ETS2 launch ~45": 45.0,
    f"base {Carbon_price_eur_kg * 1000:.0f}": Carbon_price_eur_kg * 1000.0,   # the value set in configuration.py, shown per tonne
}


def run_point(config, gen, loads, overrides):
    # One isolated-vs-interconnected comparison at a single set of overrides.
    # Mirrors run_point in scenario_sensitivity.py, trimmed to the three numbers
    # this figure needs. run_interconnected returns three things and we keep only
    # the middle one (the interconnected result); run_isolated returns two and we
    # keep the first. The leading and trailing "_" are throwaway names for the
    # parts we do not use.
    _, inter, _ = run_interconnected(gen, loads, config=config, overrides=overrides)
    iso, _      = run_isolated(gen, loads, config=config, overrides=overrides)

    lcoe_iso   = iso["lcoe_eur_kWh"]
    lcoe_inter = inter["lcoe_eur_kWh"]
    lcoe_red   = (1 - lcoe_inter / lcoe_iso) * 100           # percentage LCOE reduction from interconnecting
    return {"lcoe_iso": lcoe_iso, "lcoe_inter": lcoe_inter, "lcoe_red": lcoe_red}


def plot_carbon_price_sweep(pv_cluster, loads, n_households, window_label):
    # Sweep the carbon price for the diesel configuration and draw the two-panel
    # figure. Only PV-Diesel-BESS burns fuel, so this runs that config directly.
    config = CONFIG_PV_DIESEL
    gen    = pv_cluster                       # the diesel config uses no wind, so generation is PV only

    lo, hi = CARBON_SWEEP_EUR_TONNE
    # Evenly spaced carbon prices from lo to hi (EUR/tonne). The step is
    # (hi - lo) / (points - 1), so the first point is exactly lo and the last hi.
    prices_tonne = [lo + (hi - lo) * i / (CARBON_SWEEP_POINTS - 1)
                    for i in range(CARBON_SWEEP_POINTS)]

    lcoe_iso, lcoe_inter, reduction = [], [], []
    for price_tonne in prices_tonne:
        # Only the two carbon keys are overridden; every other parameter keeps its
        # configuration.py base value. Apply_carbon_tax is forced True so the tax
        # bites at every point, making the x-axis a real sweep even if the global
        # toggle were off.
        ov = {"Apply_carbon_tax":    True,
              "Carbon_price_eur_kg":  price_tonne / 1000.0}   # 80 EUR/tonne = 0.08 EUR/kg
        point = run_point(config, gen, loads, ov)
        lcoe_iso.append(point["lcoe_iso"])
        lcoe_inter.append(point["lcoe_inter"])
        reduction.append(point["lcoe_red"])

    # ---- figure: two panels sharing the carbon-price x-axis ----
    fig, (ax_lcoe, ax_benefit) = plt.subplots(1, 2, figsize=(13, 5.5))

    # panel 1: the two LCOE trajectories, benefit shaded between them
    ax_lcoe.plot(prices_tonne, lcoe_iso,   marker="o", color=COLOUR_ISOLATED,
                 label="Isolated fleet")
    ax_lcoe.plot(prices_tonne, lcoe_inter, marker="o", color=COLOUR_INTERCONNECTED,
                 label="Interconnected cluster")
    # fill_between shades the vertical gap between the two lines at every x. That
    # shaded area IS the interconnection benefit, and its widening left to right is
    # the finding. alpha=0.15 keeps it faint so the lines stay readable on top.
    ax_lcoe.fill_between(prices_tonne, lcoe_inter, lcoe_iso,
                         color="indianred", alpha=0.15, label="Interconnection benefit")
    ax_lcoe.set_xlabel("Carbon price (EUR/tonne CO2)")
    ax_lcoe.set_ylabel("LCOE (EUR/kWh)")
    ax_lcoe.set_title("Diesel-config LCOE under a carbon price")
    ax_lcoe.grid(True, linestyle="--", alpha=0.5)
    ax_lcoe.legend(fontsize=8)

    # panel 2: the headline, the benefit as an LCOE reduction, one rising line
    ax_benefit.plot(prices_tonne, reduction, marker="o", color="indianred")
    ax_benefit.set_xlabel("Carbon price (EUR/tonne CO2)")
    ax_benefit.set_ylabel("LCOE reduction from interconnecting (%)")
    ax_benefit.set_title("Interconnection benefit vs carbon price")
    ax_benefit.grid(True, linestyle="--", alpha=0.5)

    # reference carbon prices marked on BOTH panels (ETS2 launch anchor and the
    # value set in configuration.py), so the reader can place the policy on the axis
    for ax in (ax_lcoe, ax_benefit):
        for text, price in CARBON_REFERENCE_LINES.items():
            if lo <= price <= hi:                            # only draw a marker that falls inside the swept range
                ax.axvline(price, color="grey", linewidth=0.8, linestyle=":")
                # annotate puts a label at a point. Here x is read in DATA units (the
                # carbon price) and y in AXES-FRACTION units (0.98 = near the top),
                # so the label rides at the top of whichever panel it is in.
                ax.annotate(text, xy=(price, 0.98), xycoords=("data", "axes fraction"),
                            rotation=90, va="top", ha="right", fontsize=8, color="grey")

    fig.suptitle(f"Carbon price sensitivity: {config['name']} "
                 f"(N = {n_households})   {window_label}", fontsize=11)

    plt.tight_layout()
    fname = f"outputs/sensitivity_carbon_price_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()

    # short table to the console, matching the style of the other scenario files
    print("\n  carbon price sweep (PV-Diesel-BESS):")
    print(f"    {'EUR/tonne':>10}{'iso LCOE':>12}{'inter LCOE':>12}{'reduction':>12}")
    for p, li, lin, r in zip(prices_tonne, lcoe_iso, lcoe_inter, reduction):
        print(f"    {p:>10.0f}{li:>12.4f}{lin:>12.4f}{r:>11.2f}%")
    print(f"Saved: {fname}")


# ============================================================================
#  Run
# ============================================================================
if __name__ == "__main__":

    # Gate: only build the figure when the carbon tax is switched on in
    # configuration.py. Delete this guard if you want the file to always run.
    if not Apply_carbon_tax:
        print("Apply_carbon_tax is False in configuration.py, so the carbon figure is skipped.")
        print("Set Apply_carbon_tax = True there and rerun this file to build it.")
    else:
        os.makedirs("outputs", exist_ok=True)

        base_loads, time_index = prepare_model_loads()
        n_households = base_loads.shape[1]
        loads = base_loads

        pv_cluster   = build_pv_for_window(time_index, n_households, SOLAR_FILE, base_loads)
        window_label = f"{MODEL_WINDOW_START} to {MODEL_WINDOW_END}"

        print("\n" + "=" * 78)
        print(f"  CARBON PRICE SENSITIVITY (PV-Diesel-BESS)  ({window_label})")
        print("=" * 78)
        plot_carbon_price_sweep(pv_cluster, loads, n_households, window_label)