# ============================================================================
#  scenario_grid_extension.py
#  Compares the interconnected PV-BESS cluster with extending the main
#  low-voltage grid to the site. The one-off connection cost (HEDNO
#  participation amount) is annualised with the capital recovery factor:
#
#     LCOE_grid = grid_price + CRF(discount_rate, lifetime) * connection_cost / annual_load
#
#  The cluster load and LCOE come from a PV-BESS model run; with
#  GET_LOAD_FROM_MODEL = False the values below are used instead.
# ============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from configuration import *
from optimiser import capital_recovery_factor


# ============================================================================
#  Settings
# ============================================================================
# Connection sizes and their one-off HEDNO participation amounts [EUR].
# 8 kVA matches the peak load of the cluster.
CONNECTION_OPTIONS = [
    ("8 kVA\nsingle-phase",  12617.98),
    ("15 kVA\nthree-phase",  56249.24),
]

LIFETIME_YEARS     = 100

GET_LOAD_FROM_MODEL = True
CLUSTER_LOAD_KWH    = 1303.0      # used when GET_LOAD_FROM_MODEL is False
PVBESS_LCOE         = 0.478

SHOW_FIGURE = True

COLOUR_CLUSTER     = "#4c72b0"
COLOUR_GRID_CONN   = "#c44e52"
COLOUR_GRID_ENERGY = "#bdbdbd"


# ============================================================================
#  Grid-extension LCOE
# ============================================================================
def grid_extension_lcoe(connection_cost_eur, lifetime_years, discount_rate,
                        grid_price_eur_kWh, cluster_load_kWh):
    """LCOE of a grid extension [EUR/kWh]: grid price plus the annualised connection cost.

    Returns the total, the energy part and the connection part.
    """
    crf = capital_recovery_factor(discount_rate, lifetime_years)      # annualises the one-off cost [1/year]
    connection_part = crf * connection_cost_eur / cluster_load_kWh    # EUR/kWh
    return grid_price_eur_kWh + connection_part, grid_price_eur_kWh, connection_part


def grid_extension_floor(connection_cost_eur, discount_rate,
                         grid_price_eur_kWh, cluster_load_kWh):
    """LCOE of a grid extension for an infinite lifetime, where the CRF equals the discount rate [EUR/kWh]."""
    return grid_price_eur_kWh + discount_rate * connection_cost_eur / cluster_load_kWh


# ============================================================================
#  Figure: cluster LCOE next to the grid extension LCOE per connection size
#  (grid energy and annualised connection stacked).
# ============================================================================
def plot_grid_vs_cluster(options, lifetime_years, discount_rate, grid_price,
                         cluster_lcoe, cluster_load_kWh):

    labels = ["Interconnected\nPV-BESS cluster"] + [lab for lab, _ in options]
    x = np.arange(len(labels))                                       # bar positions 0, 1, 2, 3

    fig, ax = plt.subplots(figsize=(8, 5))

    # Interconnected cluster
    ax.bar(0, cluster_lcoe, width=0.6, color=COLOUR_CLUSTER)
    ax.text(0, cluster_lcoe, f"{cluster_lcoe:.2f}", ha="center", va="bottom", fontsize=9)

    # Grid extension per connection size
    for i, (lab, cost) in enumerate(options, start=1):
        total, energy, conn = grid_extension_lcoe(cost, lifetime_years, discount_rate,
                                                  grid_price, cluster_load_kWh)
        ax.bar(i, energy, width=0.6, color=COLOUR_GRID_ENERGY)                    # grid energy at the bottom
        ax.bar(i, conn, width=0.6, bottom=energy, color=COLOUR_GRID_CONN)
        ax.text(i, total, f"{total:.2f}", ha="center", va="bottom", fontsize=9)

    # Reference line at the cluster LCOE
    ax.axhline(cluster_lcoe, color=COLOUR_CLUSTER, ls="--", lw=1.2, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("LCOE (EUR/kWh)")

    handles = [
        Patch(color=COLOUR_CLUSTER,     label="Interconnected PV-BESS cluster"),
        Patch(color=COLOUR_GRID_CONN,   label="Grid connection (annualised)"),
        Patch(color=COLOUR_GRID_ENERGY, label="Grid energy"),
    ]
    ax.legend(handles=handles, frameon=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.10)          # 10% extra headroom so the 3.69 label fits

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    fname = f"outputs/grid_extension_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.png"
    save_figure(plt.gcf(), fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Run
# ============================================================================
if __name__ == "__main__":

    # Cluster load and interconnected PV-BESS LCOE
    if GET_LOAD_FROM_MODEL:
        from pv_profile import build_pv_for_window
        from load_profile import prepare_model_loads
        from operation import run_interconnected

        base_loads, time_index = prepare_model_loads()
        n_nanogrids = base_loads.shape[1]
        pv_cluster   = build_pv_for_window(time_index, n_nanogrids, SOLAR_FILE, base_loads)   # PV-BESS: PV only, no wind
        _, inter_optimal, _ = run_interconnected(pv_cluster, base_loads, config=CONFIG_PV_BESS)
        cluster_load_kWh = inter_optimal["cluster_load_kWh"]
        cluster_lcoe     = inter_optimal["lcoe_eur_kWh"]                # interconnected PV-BESS cluster LCOE
    else:
        cluster_load_kWh = CLUSTER_LOAD_KWH
        cluster_lcoe     = PVBESS_LCOE

    # Grid extension LCOE per connection size
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

    if SHOW_FIGURE:
        plot_grid_vs_cluster(CONNECTION_OPTIONS, LIFETIME_YEARS, DISCOUNT_RATE,
                             GRID_PRICE_EUR_KWH, cluster_lcoe, cluster_load_kWh)
