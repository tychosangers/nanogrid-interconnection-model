# ============================================================================
#  scenario_comparison.py  (SQ1)
#  Isolated versus interconnected operation for the three technological
#  configurations side by side: battery capacity per nanogrid, LCOE
#  composition, and a table of the LCOE, battery and CO2 reductions.
#  Uses the modelling window in configuration.py (USE_2025_2026_WINDOW selects
#  the basic or heavy load scenario); output filenames include the window.
#  Run with CABLING_COSTS_ON = False and APPLY_CARBON_TAX = False.
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from configuration import *
from pv_profile import build_pv_for_window
from wind_profile import build_wind_for_window
from operation import run_isolated, run_interconnected
from optimiser import lcoe_breakdown
from load_profile import prepare_model_loads


# ============================================================================
#  Settings
# ============================================================================
CONFIGS_TO_COMPARE = [CONFIG_PV_BESS, CONFIG_PV_WIND_BESS, CONFIG_PV_DIESEL_BESS]   # order shown left to right
ANNOTATE_REDUCTION = True
WRITE_TABLE_CSV    = True     # also save the quantitative table as a CSV


# Smallest bar segment that gets a value label, as a fraction of the axis height
LABEL_MIN_FRAC_BATTERY = 0.02
LABEL_MIN_FRAC_LCOE    = 0.03


# ============================================================================
#  Feasibility of isolated operation (used by scenario_cabling.py)
# ============================================================================
INFEASIBLE_BATTERY_KWH = 100.0
FEASIBILITY_TOL        = 0.999


def isolated_is_feasible(iso_result, config):
    """Whether isolated operation is a valid reference: fully renewable without
    diesel, and no nanogrid battery above INFEASIBLE_BATTERY_KWH.
    """
    if not config["use_diesel"] and iso_result["renewable_fraction"] < FEASIBILITY_TOL:
        return False
    if iso_result["optimal_batteries"].max() > INFEASIBLE_BATTERY_KWH:
        return False
    return True


# ============================================================================
#  Model runs for all configurations
# ============================================================================
def collect_results():

    base_loads, time_index = prepare_model_loads()
    n_nanogrids = base_loads.shape[1]
    loads = base_loads

    pv_cluster   = build_pv_for_window(time_index, n_nanogrids, SOLAR_FILE, base_loads)
    wind_cluster = build_wind_for_window(time_index, n_nanogrids, SOLAR_FILE, base_loads)

    rows = []
    for config in CONFIGS_TO_COMPARE:
        gen = pv_cluster + wind_cluster if config["use_wind"] else pv_cluster    # combined PV and wind supply, or PV alone

        _, inter_optimal, _ = run_interconnected(gen, loads, config=config)   # pooled battery re-selected
        iso_result, _       = run_isolated(gen, loads, config=config)         # each nanogrid sized on its own

        rows.append({
            "name":            config["name"],
            "lcoe_iso":        iso_result["lcoe_eur_kWh"],
            "lcoe_inter":      inter_optimal["lcoe_eur_kWh"],
            "battery_iso":     iso_result["battery_kWh_total"],
            "battery_inter":   inter_optimal["battery_kWh_total"],
            "batteries_iso_per_grid": iso_result["optimal_batteries"],
            "lcoe_terms_iso":   lcoe_breakdown(iso_result, config),
            "lcoe_terms_inter": lcoe_breakdown(inter_optimal, config),
            "curtailed_iso":   iso_result["annual_curtailed_kWh"],
            "curtailed_inter": inter_optimal["annual_curtailed_kWh"],
            "co2_iso":         iso_result["diesel_emissions_kg"],
            "co2_inter":       inter_optimal["diesel_emissions_kg"],
            "cluster_load_kWh": iso_result["cluster_load_kWh"],
            "iso_feasible":    isolated_is_feasible(iso_result, config),  # read by scenario_cabling.py only
        })

    window_label = f"{MODEL_WINDOW_START} to {MODEL_WINDOW_END}"
    return rows, n_nanogrids, window_label


# ============================================================================
#  Figure: battery capacity
# ============================================================================
def plot_battery_stacked(rows, n_nanogrids, window_label):
    """Battery capacity per configuration: isolated, stacked per nanogrid, next to interconnected."""
    names     = [r["name"]                   for r in rows]
    iso_grids = [r["batteries_iso_per_grid"] for r in rows]   # per-nanogrid isolated batteries [kWh]
    inter_val = [r["battery_inter"]          for r in rows]
    iso_tot   = [float(np.sum(g))            for g in iso_grids]   # fleet battery = sum of the nanogrid sizes

    x = np.arange(len(names))     # group centres: 0, 1, 2 (one per configuration)
    width = 0.35                  # half a bar sits either side of the centre

    y_top = max(iso_tot + inter_val) * 1.18   # headroom above the tallest bar for its label

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    seg_labels = []                                      # segment labels, hidden in the slide version
    for i in range(len(names)):

        # Isolated: one segment per nanogrid
        bottom = 0.0
        for g in range(n_nanogrids):
            size   = iso_grids[i][g]
            colour = NANOGRID_COLOURS[g] if g < len(NANOGRID_COLOURS) else plt.cm.tab10.colors[g]
            ax.bar(x[i] - width / 2, size, width, bottom=bottom,
                   color=colour, edgecolor="black", linewidth=0.6)
            if size > y_top * LABEL_MIN_FRAC_BATTERY:            # only number a segment tall enough to read
                seg_labels.append(ax.text(x[i] - width / 2, bottom + size / 2, f"{size:.1f}",
                                         ha="center", va="center", fontsize=8))
            bottom += size                                       # lift the base for the next nanogrid

        # interconnected bar: one pooled battery
        ax.bar(x[i] + width / 2, inter_val[i], width,
               color=COLOUR_INTERCONNECTED, edgecolor="black", linewidth=0.6)

        # Totals above the bars
        ax.annotate(f"{iso_tot[i]:.1f}",   xy=(x[i] - width / 2, iso_tot[i]),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(f"{inter_val[i]:.1f}", xy=(x[i] + width / 2, inter_val[i]),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

        if ANNOTATE_REDUCTION and iso_tot[i] > 0:
            reduction = (1 - inter_val[i] / iso_tot[i]) * 100   # positive = interconnecting is smaller
            word = "lower" if reduction >= 0 else "higher"
            top = max(iso_tot[i], inter_val[i])
            ax.annotate(f"{abs(reduction):.1f}% {word}", xy=(x[i], top),
                        xytext=(0, 18), textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold")

    # Legend: nanogrids and the interconnected bar
    handles = [Patch(facecolor=(NANOGRID_COLOURS[g] if g < len(NANOGRID_COLOURS) else plt.cm.tab10.colors[g]),
                     edgecolor="black",
                     label=(NANOGRID_NAMES[g] if g < len(NANOGRID_NAMES) else f"Nanogrid {g+1}"))
               for g in range(n_nanogrids)]
    handles.append(Patch(facecolor=COLOUR_INTERCONNECTED, edgecolor="black", label="Interconnected"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("BESS capacity (kWh)")
    ax.set_ylim(0, y_top)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)                                      # gridlines behind the bars

    plt.tight_layout()
    fname = f"outputs/battery_stacked_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.png"
    save_figure(plt.gcf(), fname, dpi=150, hide_in_slides=seg_labels)
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Figure: LCOE composition
# ============================================================================
def plot_lcoe_composition(rows, n_nanogrids, window_label):
    """Stacked LCOE per configuration, isolated (left) and interconnected (right), split into cost terms."""
    names       = [r["name"]             for r in rows]
    iso_terms   = [r["lcoe_terms_iso"]   for r in rows]
    inter_terms = [r["lcoe_terms_inter"] for r in rows]

    # Cost terms that are non-zero in at least one bar, in stacking order
    order = list(iso_terms[0].keys())
    used  = [name for name in order
             if any(abs(t[name]) > 1e-12 for t in iso_terms + inter_terms)]

    iso_tot   = [sum(t.values()) for t in iso_terms]
    inter_tot = [sum(t.values()) for t in inter_terms]
    y_top = max(iso_tot + inter_tot) * 1.15

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    seg_labels = []                                      # segment labels, hidden in the slide version
    for i in range(len(names)):

        for terms, xpos, total in ((iso_terms[i],   x[i] - width / 2, iso_tot[i]),
                                   (inter_terms[i], x[i] + width / 2, inter_tot[i])):
            bottom = 0.0
            for name in used:
                value = terms[name]
                ax.bar(xpos, value, width, bottom=bottom, color=COST_COLOURS.get(name, "lightgrey"),
                       edgecolor="black", linewidth=0.5)
                if value > LABEL_MIN_FRAC_LCOE * y_top:        # number a segment only if tall enough to read
                    seg_labels.append(ax.text(xpos, bottom + value / 2, f"{value:.2f}",
                                             ha="center", va="center", fontsize=7))
                bottom += value                                # raise the floor for the next term
            ax.annotate(f"{total:.3f}", xy=(xpos, total), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9)

        if ANNOTATE_REDUCTION and iso_tot[i] > 0:
            reduction = (1 - inter_tot[i] / iso_tot[i]) * 100   # positive = interconnecting is cheaper
            word = "lower" if reduction >= 0 else "higher"
            top = max(iso_tot[i], inter_tot[i])
            ax.annotate(f"{abs(reduction):.1f}% {word}", xy=(x[i], top),
                        xytext=(0, 16), textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold")

        # x labels: topology, with the configuration name below
        ax.annotate("isolated", xy=(x[i] - width / 2, 0), xytext=(0, -12),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7, annotation_clip=False)
        ax.annotate("inter.", xy=(x[i] + width / 2, 0), xytext=(0, -12),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7, annotation_clip=False)
        ax.annotate(names[i], xy=(x[i], 0), xytext=(0, -26),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=10, fontweight="bold", annotation_clip=False)

    # Legend in the order of the stack, top first
    handles = [Patch(facecolor=COST_COLOURS.get(name, "lightgrey"), edgecolor="black", label=COST_LABELS.get(name, name)) for name in used]
    ax.legend(handles=handles[::-1], loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(names))
    ax.set_ylabel("LCOE contribution (EUR/kWh)")
    ax.set_ylim(0, y_top)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fname = f"outputs/lcoe_composition_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.png"
    save_figure(plt.gcf(), fname, dpi=150, bbox_inches="tight", hide_in_slides=seg_labels)
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Table of reductions from interconnection [%]. The CO2 reduction is empty
#  for configurations without diesel.
# ============================================================================
def pct_reduction(iso_value, inter_value):
    if iso_value is None or iso_value <= 0:
        return float("nan")
    return (1 - inter_value / iso_value) * 100


def build_quantitative_table(rows):
    table_rows = []
    for r in rows:
        table_rows.append({
            "configuration":         r["name"],
            "LCOE reduction [%]":    pct_reduction(r["lcoe_iso"], r["lcoe_inter"]),
            "battery reduction [%]": pct_reduction(r["battery_iso"], r["battery_inter"]),
            "CO2 reduction [%]":     pct_reduction(r["co2_iso"], r["co2_inter"]) if r["co2_iso"] > 0 else float("nan"),
        })
    return pd.DataFrame(table_rows)


# ============================================================================
#  Run
# ============================================================================
if __name__ == "__main__":

    os.makedirs("outputs", exist_ok=True)
    rows, n_nanogrids, window_label = collect_results()

    table = build_quantitative_table(rows)
    print("\n" + "=" * 70)
    print(f"  QUANTITATIVE COMPARISON  ({window_label})")
    print("=" * 70)
    print(table.to_string(index=False, na_rep="-", float_format=lambda v: f"{v:,.1f}"))
    print("=" * 70)
    if WRITE_TABLE_CSV:
        csv_name = f"outputs/quantitative_table_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.csv"
        table.to_csv(csv_name, index=False)
        print(f"Saved: {csv_name}")

    print("\n  Curtailment (kWh), isolated and interconnected:")
    for r in rows:
        print(f"    {r['name']:<14}{r['curtailed_iso']:>12,.0f}  {r['curtailed_inter']:>12,.0f}")

    plot_battery_stacked(rows, n_nanogrids, window_label)
    plot_lcoe_composition(rows, n_nanogrids, window_label)
