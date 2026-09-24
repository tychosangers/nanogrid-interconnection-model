# ============================================================================
#  scenario_grid_extension.py
#  Appendix C support: is extending the main low-voltage grid to the site
#  cheaper than the interconnected off-grid cluster?
#
#  A one-off grid-connection cost (the DEDDIE participation amount) is turned
#  into a levelised cost of electricity so it can be read on the same EUR/kWh
#  axis as the nanogrid results:
#
#     LCOE_grid = grid_price + CRF(discount_rate, lifetime) * connection_cost
#                                                            / annual_load
#
#  The connection is annualised with the capital recovery factor, exactly like
#  every other capex term in optimiser.py, and the site still buys its energy
#  from the grid at the retail price on top.
#
#  The script compares the interconnected PV-BESS cluster against grid extension
#  at three connection sizes (the participation amounts DEDDIE returns for each),
#  so the comparison is shown as a small sensitivity rather than a single number.
#  The 8 kVA connection is the like-for-like size, matched to the cluster peak;
#  the larger sizes show that oversizing the connection only widens the gap.
#
#  Self-contained press-run script. The cluster annual load and the interconnected
#  PV-BESS LCOE come from ONE model run (PV-BESS only), so they match the window
#  and settings in configuration.py. Set GET_LOAD_FROM_MODEL = False to skip that
#  run and type the two numbers in by hand instead.
# ============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch          # a plain coloured rectangle, used to build the legend by hand

from configuration import *
from optimiser import capital_recovery_factor


# ============================================================================
#  TOGGLES
# ============================================================================
# Connection sizes and their one-off DEDDIE participation amounts [EUR].
# The first entry is the headline size, matched to the cluster's coincident peak.
CONNECTION_OPTIONS = [
    ("8 kVA\nsingle-phase",  12617.98),
    ("15 kVA\nthree-phase",  56249.24),
]

LIFETIME_YEARS     = 100                    # annualise the one-off cost over this many years
DISCOUNT_RATE      = Discount_rate         # cost of capital, defaults to the model's 8%
GRID_PRICE_EUR_KWH = Grid_price_eur_kWh    # retail grid electricity price [EUR/kWh]

GET_LOAD_FROM_MODEL = True     # True: one PV-BESS model run for the cluster load and its interconnected LCOE
CLUSTER_LOAD_KWH    = 2520.0   # used only if GET_LOAD_FROM_MODEL is False [kWh/yr]
PVBESS_LCOE         = 0.478    # used only if GET_LOAD_FROM_MODEL is False [EUR/kWh], interconnected PV-BESS

SHOW_FIGURE = True

# Colours kept close to the other figures: blue for the PV-BESS cluster, red for
# the grid-connection cost, grey for the grid energy the site still buys.
COLOUR_CLUSTER     = "#4c72b0"
COLOUR_GRID_CONN   = "#c44e52"
COLOUR_GRID_ENERGY = "#bdbdbd"


# ============================================================================
#  Grid-extension LCOE (same annuity convention as optimiser.py).
# ============================================================================
def grid_extension_lcoe(connection_cost_eur, lifetime_years, discount_rate,
                        grid_price_eur_kWh, cluster_load_kWh):
    """Levelised cost of extending the grid and then buying energy from it [EUR/kWh].

    The one-off connection cost is annualised with the capital recovery factor
    (the same function the nanogrid capex uses), divided by the annual load to put
    it in EUR/kWh, and the retail grid price is added because the site still pays
    for every kWh it draws. Returns the total plus its two parts (energy,
    connection) so the figure can stack them.
    """
    crf = capital_recovery_factor(discount_rate, lifetime_years)      # annualises the one-off cost [1/year]
    connection_part = crf * connection_cost_eur / cluster_load_kWh    # EUR/kWh
    return grid_price_eur_kWh + connection_part, grid_price_eur_kWh, connection_part


def grid_extension_floor(connection_cost_eur, discount_rate,
                         grid_price_eur_kWh, cluster_load_kWh):
    """The lowest LCOE the extension can ever reach, at an infinite lifetime [EUR/kWh].

    As the lifetime grows the capital recovery factor falls towards the discount
    rate, so the LCOE flattens to this floor. If even this floor is above the
    cluster LCOE, no lifetime can make the extension competitive.
    """
    return grid_price_eur_kWh + discount_rate * connection_cost_eur / cluster_load_kWh


# ============================================================================
#  Figure: interconnected PV-BESS cluster vs grid extension at three sizes.
#  One blue bar for the cluster, then one stacked bar per connection size
#  (grey grid energy at the bottom, red annualised connection on top).
# ============================================================================
def plot_grid_vs_cluster(options, lifetime_years, discount_rate, grid_price,
                         cluster_lcoe, cluster_load_kWh):

    labels = ["Interconnected\nPV-BESS cluster"] + [lab for lab, _ in options]
    x = np.arange(len(labels))                                       # bar positions 0, 1, 2, 3

    fig, ax = plt.subplots(figsize=(8, 5))

    # Bar 0: the off-grid cluster, a single solid blue bar.
    ax.bar(0, cluster_lcoe, width=0.6, color=COLOUR_CLUSTER)
    ax.text(0, cluster_lcoe, f"{cluster_lcoe:.2f}", ha="center", va="bottom", fontsize=9)

    # Bars 1..: grid extension, energy stacked below the annualised connection.
    for i, (lab, cost) in enumerate(options, start=1):
        total, energy, conn = grid_extension_lcoe(cost, lifetime_years, discount_rate,
                                                  grid_price, cluster_load_kWh)
        ax.bar(i, energy, width=0.6, color=COLOUR_GRID_ENERGY)                    # grid energy at the bottom
        ax.bar(i, conn, width=0.6, bottom=energy, color=COLOUR_GRID_CONN)         # connection stacked on top (bottom= sits it above)
        ax.text(i, total, f"{total:.2f}", ha="center", va="bottom", fontsize=9)

    # Dashed threshold line at the cluster LCOE: anything above it is where the grid loses.
    ax.axhline(cluster_lcoe, color=COLOUR_CLUSTER, ls="--", lw=1.2, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("LCOE (EUR/kWh)")

    # Legend built by hand from coloured patches (no title on the figure).
    handles = [
        Patch(color=COLOUR_CLUSTER,     label="Interconnected PV-BESS cluster"),
        Patch(color=COLOUR_GRID_CONN,   label="Grid connection (annualised)"),
        Patch(color=COLOUR_GRID_ENERGY, label="Grid energy"),
    ]
    ax.legend(handles=handles, frameon=False)
    ax.grid(True, axis="y", ls="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    fname = f"outputs/grid_extension_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Run
# ============================================================================
if __name__ == "__main__":

    # --- cluster load and the interconnected PV-BESS LCOE ---
    if GET_LOAD_FROM_MODEL:
        # Imported here so the by-hand path does not pull in the whole model chain.
        from pv_profile import build_pv_for_window
        from load_profile import prepare_model_loads
        from operation import run_interconnected

        base_loads, time_index = prepare_model_loads()
        n_households = base_loads.shape[1]
        pv_cluster   = build_pv_for_window(time_index, n_households, SOLAR_FILE, base_loads)   # PV-BESS: PV only, no wind
        _, inter_optimal, _ = run_interconnected(pv_cluster, base_loads, config=CONFIG_PV_ONLY)
        cluster_load_kWh = inter_optimal["cluster_load_kWh"]
        cluster_lcoe     = inter_optimal["lcoe_eur_kWh"]                # interconnected PV-BESS cluster LCOE
    else:
        cluster_load_kWh = CLUSTER_LOAD_KWH
        cluster_lcoe     = PVBESS_LCOE

    # --- console table: grid LCOE by connection size ---
    print("\n" + "=" * 70)
    print("  GRID EXTENSION vs INTERCONNECTED PV-BESS CLUSTER")
    print(f"  lifetime {LIFETIME_YEARS} yr | discount {DISCOUNT_RATE:.0%} | "
          f"grid {GRID_PRICE_EUR_KWH:.2f} EUR/kWh | load {cluster_load_kWh:,.0f} kWh/yr | "
          f"cable {'on' if CABLING_COSTS_ON else 'off'}")
    print(f"  interconnected PV-BESS cluster LCOE: {cluster_lcoe:.3f} EUR/kWh")
    print("=" * 70)
    print(f"  {'connection':>22}{'cost EUR':>12}{'LCOE':>10}{'floor':>10}")
    print("-" * 70)
    for lab, cost in CONNECTION_OPTIONS:
        total, _, _ = grid_extension_lcoe(cost, LIFETIME_YEARS, DISCOUNT_RATE,
                                          GRID_PRICE_EUR_KWH, cluster_load_kWh)
        floor = grid_extension_floor(cost, DISCOUNT_RATE, GRID_PRICE_EUR_KWH, cluster_load_kWh)
        print(f"  {lab.replace(chr(10), ' '):>22}{cost:>12,.0f}{total:>10.3f}{floor:>10.3f}")
    print("=" * 70)
    print("  'floor' is the LCOE at an infinite lifetime: even then the grid stays above the cluster.")

    # --- figure ---
    if SHOW_FIGURE:
        plot_grid_vs_cluster(CONNECTION_OPTIONS, LIFETIME_YEARS, DISCOUNT_RATE,
                             GRID_PRICE_EUR_KWH, cluster_lcoe, cluster_load_kWh)