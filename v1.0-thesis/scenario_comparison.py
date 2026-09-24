# ============================================================================
#  scenario_comparison.py
#  SQ1 comparison outputs across the three technology configurations
#  (PV-BESS, PV-Wind-BESS, PV-Diesel-BESS):
#     1. fleet battery capacity, isolated versus interconnected -> figure
#     2. cluster LCOE composition, isolated versus interconnected -> figure
#     3. a quantitative table (LCOE reduction, battery reduction,
#        CO2 reduction)                                          -> console + CSV
#
#  This is a self-contained, press-run script. It does NOT use the CONFIGURATION
#  selector in configuration.py; instead it loops over the three configurations
#  itself and draws them side by side, so one figure compares all three.
#
#  The basic and heavy load cases are the same script run twice: set
#  USE_2025_2026_WINDOW in configuration.py (False = 2025 calendar year for the
#  basic load, True = Jun 2025 to May 2026 for the heavy air-conditioning load),
#  restart the kernel, and run this file again. The window dates are read from
#  configuration.py, so titles and saved filenames label themselves automatically
#  and the two runs do not overwrite each other.
#
#  The optional overlays are held at the comparison defaults (cable modelled as
#  free via CABLING_COSTS_ON = False, no carbon tax) so the three
#  configurations are judged on the storage-pooling effect alone.
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch                 # a plain coloured rectangle, used to build the legend by hand

from configuration import *
from pv_profile import build_pv_for_window
from wind_profile import build_wind_for_window
from operation import run_isolated, run_interconnected   # the two operating modes: isolated fleet and interconnected cluster
from optimiser import grid_extension_breakeven, lcoe_breakdown
from load_profile import prepare_model_loads


# ============================================================================
#  TOGGLES
# ============================================================================
CONFIGS_TO_COMPARE = [CONFIG_PV_ONLY, CONFIG_PV_WIND, CONFIG_PV_DIESEL]   # order shown left to right
ANNOTATE_REDUCTION = True     # print the reduction above each configuration pair
WRITE_TABLE_CSV    = True     # also save the quantitative table as a CSV

# The battery comparison (Figure 18) is offered in two forms, so you can pick the
# one that reads best in the report. Both compare the three configurations,
# isolated versus interconnected, and differ only in how the isolated bar is
# drawn. Switch either off if you only want the other.
SHOW_BATTERY_STACKED = True   # isolated bar split into one coloured segment per lodge
SHOW_BATTERY_SOLID   = True   # isolated bar as a single solid slate bar

# Smallest segment that still gets a numeric label, as a fraction of the axis
# height. Lower the value to label thinner segments (for example the Greenhouse
# battery in the diesel configuration, or the small diesel fuel LCOE term).
# Raise it to hide cluttered labels on very thin segments.
LABEL_MIN_FRAC_BATTERY = 0.02   # stacked battery figure (was 0.03)
LABEL_MIN_FRAC_LCOE    = 0.03   # LCOE composition figure (was 0.05)

# Grid-extension break-even (interconnected cluster only, one figure per time window).
# This section is INDEPENDENT of USE_2025_2026_WINDOW: it defines both windows itself
# and loops over them, so a single press-run produces both figures. It therefore runs
# its own model runs (3 configurations x 2 windows = 6), on top of the 3 the figures
# above already do. Switch it off if you only want the comparison figures.
SHOW_GRID_EXTENSION = False
GRID_WINDOWS = [
    {"label": "2025 calendar year",
     "start": "2025-01-01", "end": "2025-12-31",
     "solar": "data/solar/Solar_Data_TinosEcoLodge_2025_Solcast.csv"},
    {"label": "Jun 2025 to May 2026",
     "start": "2025-06-01", "end": "2026-05-31",
     "solar": "data/solar/Solar_Wind_Data_TinosEcoLodge_2025_2026_Solcast.csv"},
]


# ============================================================================
#  Backward-compatibility helper (NOT part of any figure or table here).
#  scenario_cabling.py and scenario_grid_extension.py both import collect_results
#  and still read the iso_feasible field, so the helper below stays alive only to
#  keep that field populated. Nothing drawn or tabulated in this file uses it any
#  more. It can be deleted here once scenario_cabling.py stops depending on it.
# ============================================================================
INFEASIBLE_BATTERY_KWH = 100.0    # a single lodge needing a larger standalone battery than this is flagged unfeasible for scenario_cabling
FEASIBILITY_TOL        = 0.999    # a no-diesel cluster below this renewable fraction has left load unmet


def isolated_is_feasible(iso_result, config):
    """Kept only for scenario_cabling.py. Returns whether the isolated fleet can
    reasonably serve this configuration's load on its own.
    """
    if not config["use_diesel"] and iso_result["renewable_fraction"] < FEASIBILITY_TOL:
        return False
    if iso_result["optimal_batteries"].max() > INFEASIBLE_BATTERY_KWH:   # .max() = the largest of the per-lodge battery sizes
        return False
    return True


# ============================================================================
#  Run the three configurations once and collect what both figures and the table
#  need. Loads, PV and wind are built a single time; wind is only added to the
#  supply for the configuration that uses it (exactly as run_model.run_one does).
# ============================================================================
def collect_results():

    base_loads, time_index = prepare_model_loads()
    n_households = base_loads.shape[1]
    loads = base_loads

    pv_cluster   = build_pv_for_window(time_index, n_households, SOLAR_FILE, base_loads)
    wind_cluster = build_wind_for_window(time_index, n_households, SOLAR_FILE, base_loads)   # built once, used only by the PV-Wind-BESS config

    rows = []
    for config in CONFIGS_TO_COMPARE:
        gen = pv_cluster + wind_cluster if config["use_wind"] else pv_cluster    # combined PV and wind supply, or PV alone

        _, inter_optimal, _ = run_interconnected(gen, loads, config=config)   # pooled battery re-selected
        iso_result, _       = run_isolated(gen, loads, config=config)         # each lodge sized on its own

        rows.append({
            "name":            config["name"],
            "lcoe_iso":        iso_result["lcoe_eur_kWh"],
            "lcoe_inter":      inter_optimal["lcoe_eur_kWh"],
            "battery_iso":     iso_result["battery_kWh_total"],
            "battery_inter":   inter_optimal["battery_kWh_total"],
            "batteries_iso_per_grid": iso_result["optimal_batteries"],   # per-lodge isolated battery [kWh], for the stacked isolated bar
            "lcoe_terms_iso":   lcoe_breakdown(iso_result, config),       # {cost term: EUR/kWh}, for the LCOE composition figure
            "lcoe_terms_inter": lcoe_breakdown(inter_optimal, config),
            "curtailed_iso":   iso_result["annual_curtailed_kWh"],
            "curtailed_inter": inter_optimal["annual_curtailed_kWh"],
            "co2_iso":         iso_result["diesel_emissions_kg"],
            "co2_inter":       inter_optimal["diesel_emissions_kg"],
            "cluster_load_kWh": iso_result["cluster_load_kWh"],           # read by scenario_cabling.py and scenario_grid_extension.py
            "iso_feasible":    isolated_is_feasible(iso_result, config),  # read by scenario_cabling.py only
        })

    window_label = f"{MODEL_WINDOW_START} to {MODEL_WINDOW_END}"
    return rows, n_households, window_label


# ============================================================================
#  Figure 1, form A: isolated bar split per lodge.
# ============================================================================
def plot_battery_stacked(rows, n_households, window_label):
    """Fleet battery per configuration, isolated versus interconnected, with the
    isolated bar stacked by lodge (one coloured segment per lodge) so you can see
    which lodges drive the standalone requirement. The interconnected bar is the
    single pooled battery, in purple. Every bar is drawn at its true height, so
    the figure states the actual storage each topology needs.
    """
    names     = [r["name"]                   for r in rows]
    iso_grids = [r["batteries_iso_per_grid"] for r in rows]   # per-lodge isolated batteries [kWh]
    inter_val = [r["battery_inter"]          for r in rows]
    iso_tot   = [float(np.sum(g))            for g in iso_grids]   # fleet battery = sum of the lodge sizes

    x = np.arange(len(names))     # group centres: 0, 1, 2 (one per configuration)
    width = 0.35                  # half a bar sits either side of the centre

    y_top = max(iso_tot + inter_val) * 1.18   # headroom above the tallest bar for its label

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for i in range(len(names)):

        # isolated bar: stack the lodges on top of one another
        bottom = 0.0
        for g in range(n_households):
            size   = iso_grids[i][g]
            colour = LODGE_COLOURS[g] if g < len(LODGE_COLOURS) else plt.cm.tab10.colors[g]
            ax.bar(x[i] - width / 2, size, width, bottom=bottom,
                   color=colour, edgecolor="black", linewidth=0.6)
            if size > y_top * LABEL_MIN_FRAC_BATTERY:            # only number a segment tall enough to read
                ax.text(x[i] - width / 2, bottom + size / 2, f"{size:.1f}",
                        ha="center", va="center", fontsize=8)
            bottom += size                                       # lift the base for the next lodge

        # interconnected bar: one pooled battery
        ax.bar(x[i] + width / 2, inter_val[i], width,
               color=COLOUR_INTERCONNECTED, edgecolor="black", linewidth=0.6)

        # totals on top of each bar
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

    # Legend built by hand: one entry per lodge, plus the pooled bar.
    handles = [Patch(facecolor=(LODGE_COLOURS[g] if g < len(LODGE_COLOURS) else plt.cm.tab10.colors[g]),
                     edgecolor="black",
                     label=(LODGE_NAMES[g] if g < len(LODGE_NAMES) else f"Nanogrid {g+1}"))
               for g in range(n_households)]
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
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Figure 1, form B: isolated bar as one solid slate bar.
# ============================================================================
def plot_battery_solid(rows, n_households, window_label):
    """Fleet battery per configuration, isolated versus interconnected, with the
    isolated fleet shown as a single solid slate bar (its total, not split by
    lodge) beside the purple interconnected pool. This is the cleaner read when
    the point is the headline reduction rather than which lodge drives it.
    """
    names     = [r["name"]                   for r in rows]
    iso_grids = [r["batteries_iso_per_grid"] for r in rows]
    inter_val = [r["battery_inter"]          for r in rows]
    iso_tot   = [float(np.sum(g))            for g in iso_grids]   # fleet battery = sum of the lodge sizes

    x = np.arange(len(names))
    width = 0.35

    y_top = max(iso_tot + inter_val) * 1.18

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for i in range(len(names)):

        # isolated bar: one solid slate bar at the fleet total
        ax.bar(x[i] - width / 2, iso_tot[i], width,
               color=COLOUR_ISOLATED, edgecolor="black", linewidth=0.6)
        # interconnected bar: one pooled battery
        ax.bar(x[i] + width / 2, inter_val[i], width,
               color=COLOUR_INTERCONNECTED, edgecolor="black", linewidth=0.6)

        ax.annotate(f"{iso_tot[i]:.1f}",   xy=(x[i] - width / 2, iso_tot[i]),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(f"{inter_val[i]:.1f}", xy=(x[i] + width / 2, inter_val[i]),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

        if ANNOTATE_REDUCTION and iso_tot[i] > 0:
            reduction = (1 - inter_val[i] / iso_tot[i]) * 100
            word = "lower" if reduction >= 0 else "higher"
            top = max(iso_tot[i], inter_val[i])
            ax.annotate(f"{abs(reduction):.1f}% {word}", xy=(x[i], top),
                        xytext=(0, 18), textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold")

    handles = [Patch(facecolor=COLOUR_ISOLATED,      edgecolor="black", label="Isolated"),
               Patch(facecolor=COLOUR_INTERCONNECTED, edgecolor="black", label="Interconnected")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("BESS capacity (kWh)")
    ax.set_ylim(0, y_top)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fname = f"outputs/battery_solid_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Figure 2: LCOE composition, isolated versus interconnected.
# ============================================================================
def plot_lcoe_composition(rows, n_households, window_label):
    """For each configuration the isolated LCOE and the interconnected LCOE are
    drawn as stacked bars split into their cost terms (PV, wind, diesel, cable,
    battery), like the single-configuration breakdown in reporting but for all
    three configurations side by side. In each pair the left bar is isolated and
    the right bar is interconnected, so the height difference is the
    interconnection saving and the colours show where it comes from (the
    shrinking battery block at the top).
    """
    names       = [r["name"]             for r in rows]
    iso_terms   = [r["lcoe_terms_iso"]   for r in rows]   # {cost term: EUR/kWh}, already in stacking order
    inter_terms = [r["lcoe_terms_inter"] for r in rows]

    # Keep only the terms that are non-zero somewhere, in the fixed stacking order
    # the breakdown already uses (PV at the bottom, battery capex on top).
    order = list(iso_terms[0].keys())
    used  = [name for name in order
             if any(abs(t[name]) > 1e-12 for t in iso_terms + inter_terms)]

    iso_tot   = [sum(t.values()) for t in iso_terms]
    inter_tot = [sum(t.values()) for t in inter_terms]
    y_top = max(iso_tot + inter_tot) * 1.15

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(names)):

        # isolated (left) and interconnected (right) stacks share the term colours
        for terms, xpos, total in ((iso_terms[i],   x[i] - width / 2, iso_tot[i]),
                                   (inter_terms[i], x[i] + width / 2, inter_tot[i])):
            bottom = 0.0
            for name in used:
                value = terms[name]
                ax.bar(xpos, value, width, bottom=bottom, color=COST_COLOURS.get(name, "lightgrey"),
                       edgecolor="black", linewidth=0.5)
                if value > LABEL_MIN_FRAC_LCOE * y_top:        # number a segment only if tall enough to read
                    ax.text(xpos, bottom + value / 2, f"{value:.2f}",
                            ha="center", va="center", fontsize=7)
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

        # two-row x labels: which bar is which, then the configuration name below
        ax.annotate("isolated", xy=(x[i] - width / 2, 0), xytext=(0, -12),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7, annotation_clip=False)
        ax.annotate("inter.", xy=(x[i] + width / 2, 0), xytext=(0, -12),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7, annotation_clip=False)
        ax.annotate(names[i], xy=(x[i], 0), xytext=(0, -26),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=10, fontweight="bold", annotation_clip=False)

    # Legend: cost terms, reversed so the top-of-bar term reads first.
    handles = [Patch(facecolor=COST_COLOURS.get(name, "lightgrey"), edgecolor="black", label=COST_LABELS.get(name, name)) for name in used]
    ax.legend(handles=handles[::-1], loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(names))     # configuration names are drawn manually on the lower row
    ax.set_ylabel("LCOE contribution (EUR/kWh)")
    ax.set_ylim(0, y_top)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fname = f"outputs/lcoe_composition_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")   # bbox_inches keeps the outside legend and lower labels
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Quantitative table: the numbers to quote in the Comparison Analysis text.
#  Reductions are the percentage the interconnected value sits below the isolated
#  one. The CO2 reduction is blank for the two configurations that burn no diesel.
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


def collect_grid_extension(window):
    """Run the three configurations, interconnected only, for one time window.

    The window is passed in rather than read from configuration.py, so both
    windows can be built inside a single kernel run without editing anything.
    """
    base_loads, time_index = prepare_model_loads(start=window["start"], end=window["end"])
    n_households = base_loads.shape[1]
    loads = base_loads

    pv_cluster   = build_pv_for_window(time_index, n_households, window["solar"], base_loads)
    wind_cluster = build_wind_for_window(time_index, n_households, window["solar"], base_loads)

    out = []
    for config in CONFIGS_TO_COMPARE:
        gen = pv_cluster + wind_cluster if config["use_wind"] else pv_cluster

        _, inter_optimal, _ = run_interconnected(gen, loads, config=config)

        # Returns three numbers, unpacked in the order optimiser.py defines them.
        nano_cost, grid_energy, breakeven_capex = grid_extension_breakeven(
            inter_optimal, Grid_price_eur_kWh, Grid_project_lifetime_years
        )

        out.append({
            "name":            config["name"],
            "lcoe":            inter_optimal["lcoe_eur_kWh"],
            "load_kWh":        inter_optimal["cluster_load_kWh"],
            "nano_cost":       nano_cost,
            "grid_energy":     grid_energy,
            "breakeven_capex": breakeven_capex,
        })
    return out, n_households


def plot_grid_extension(grid_rows, n_households, window):
    names  = [g["name"]            for g in grid_rows]
    values = [g["breakeven_capex"] for g in grid_rows]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # A negative break-even means the nanogrid wins outright, so those bars are
    # coloured differently rather than being quietly drawn below the axis.
    colours = ["firebrick" if v >= 0 else "seagreen" for v in values]
    ax.bar(x, values, 0.5, color=colours, edgecolor="black")

    for i, g in enumerate(grid_rows):
        v = g["breakeven_capex"]
        # va flips with the sign so the label always sits outside the bar, not inside it.
        ax.annotate(f"{v:,.0f} EUR",
                    xy=(x[i], v),
                    xytext=(0, 4 if v >= 0 else -14), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")
        ax.annotate(f"LCOE {g['lcoe']:.3f}",
                    xy=(x[i], 0), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, style="italic", color="white")

    ax.axhline(0, color="black", linewidth=0.8)                 # the zero line matters here, so draw it explicitly

    # Headroom above and below, so the value labels are never clipped.
    lo, hi = min(values + [0]), max(values + [0])
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - 0.20 * span, hi + 0.25 * span)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Break-even grid-extension cost (EUR)")
    ax.set_title(
        f"Grid-extension break-even, interconnected cluster (N = {n_households})\n"
        f"{window['label']} | grid {Grid_price_eur_kWh:.2f} EUR/kWh over "
        f"{Grid_project_lifetime_years} years | cable "
        f"{'on' if CABLING_COSTS_ON else 'OFF'}"
    )
    # One line of interpretation on the figure itself, because the direction of this
    # metric is easy to read backwards.
    ax.text(0.99, 0.97,
            "Below the bar: extending the grid is cheaper\n"
            "Above the bar: the nanogrid is cheaper",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, style="italic", color="grey")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fname = f"outputs/grid_extension_breakeven_{window['start']}_to_{window['end']}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Run
# ============================================================================
if __name__ == "__main__":

    os.makedirs("outputs", exist_ok=True)
    rows, n_households, window_label = collect_results()

    # --- quantitative table ---
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

    # --- curtailment, printed for reference only (kept out of the table since it barely moves) ---
    print("\n  curtailment for reference (kWh), isolated -> interconnected:")
    for r in rows:
        print(f"    {r['name']:<14}{r['curtailed_iso']:>12,.0f}  ->{r['curtailed_inter']:>12,.0f}")

    # --- Figure 18: fleet battery, isolated versus interconnected ---
    # Two forms of the same comparison; keep whichever reads best in the report.
    # Form A splits the isolated bar per lodge; form B shows it as one slate bar.
    if SHOW_BATTERY_STACKED:
        plot_battery_stacked(rows, n_households, window_label)
    if SHOW_BATTERY_SOLID:
        plot_battery_solid(rows, n_households, window_label)

    # --- Figure 19: LCOE composition, isolated versus interconnected ---
    plot_lcoe_composition(rows, n_households, window_label)

    # --- grid-extension break-even, one per time window (optional) ---
    if SHOW_GRID_EXTENSION:
        if not CABLING_COSTS_ON:
            print("\nNOTE: CABLING_COSTS_ON is False, so the interconnected LCOE below does not pay")
            print("      for its cable. The grid-extension break-even is therefore optimistic.")

        for window in GRID_WINDOWS:
            grid_rows, n_grid = collect_grid_extension(window)

            print("\n" + "=" * 78)
            print(f"  GRID-EXTENSION BREAK-EVEN, interconnected  ({window['label']})")
            print(f"  grid {Grid_price_eur_kWh:.2f} EUR/kWh over {Grid_project_lifetime_years} years")
            print("=" * 78)
            print(f"  {'config':<14}{'LCOE':>8}{'nanogrid EUR':>15}{'grid energy EUR':>18}{'break-even EUR':>17}")
            print("-" * 78)
            for g in grid_rows:
                print(f"  {g['name']:<14}{g['lcoe']:>8.3f}{g['nano_cost']:>15,.0f}"
                      f"{g['grid_energy']:>18,.0f}{g['breakeven_capex']:>17,.0f}")
            print("-" * 78)
            print("  A grid extension cheaper than the break-even figure beats the nanogrid;")
            print("  a dearer one does not. Negative = the nanogrid wins even with a free extension.")
            print("=" * 78)

            plot_grid_extension(grid_rows, n_grid, window)