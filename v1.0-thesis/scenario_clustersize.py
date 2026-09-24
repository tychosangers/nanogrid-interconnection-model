# ============================================================================
#  scenario_clustersize.py  (SQ3)
#  Cluster size and the interconnection benefit: does pooling more nanogrids help
#  on average? For every subset of the four lodges (six pairs, four triples, one
#  quad) the isolated and interconnected cases are compared, and the battery and
#  LCOE reductions from interconnection are read off. The reductions are then
#  grouped by cluster size N to see whether the average benefit rises with N.
#
#  A subset is just a column slice of the full load and PV arrays, so the existing
#  isolated and interconnected scenarios are reused as-is; nothing in them changes.
#
#  Two outputs are produced:
#    1. A two-panel grouped bar chart (battery reduction on top, LCOE reduction
#       below) for the headline configuration, with the eleven subsets laid out in
#       three groups (N = 2, 3, 4) and a dashed mean line per group.
#    2. A table of the average battery and LCOE reduction by N for all three
#       configurations, plus an across-configuration average.
#
#  SQ3 treats interconnection as free (cabling is SQ4), so run with
#  CABLING_COSTS_ON = False in configuration.py. The wind profile is built here
#  directly, independent of the CONFIGURATION set in configuration.py, so all three
#  configurations run from a single press.
#
#  Note on runtime: the table runs 11 subsets for each of three configurations,
#  isolated and interconnected, so about 66 scenario runs. Trim CONFIGS_FOR_TABLE
#  to [CONFIG_PV_ONLY] for a quick pass if you only need the headline chart.
# ============================================================================

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch                     # coloured rectangles for the cluster-size legend

from configuration import (CONFIG_PV_ONLY, CONFIG_PV_WIND, CONFIG_PV_DIESEL,
                           CABLING_COSTS_ON, SOLAR_FILE, LODGE_NAMES, LODGE_TAGS)
from pv_profile import build_pv_for_window
from wind_profile import build_wind_for_window
from load_profile import prepare_model_loads
from operation import run_isolated, run_interconnected   # the two operating modes: isolated fleet and interconnected cluster


# ============================================================================
#  TOGGLES
# ============================================================================
# Which subset sizes to analyse. All eleven subsets of the four nanogrids:
# 6 pairs + 4 triples + 1 quad.
SUBSET_SIZES = [2, 3, 4]

# Configurations to include in the averages table. The battery reduction is near
# zero for PV+diesel (the diesel already suppresses the battery), so its story is
# carried by the LCOE reduction instead.
CONFIGS_FOR_TABLE = [CONFIG_PV_ONLY, CONFIG_PV_WIND, CONFIG_PV_DIESEL]

# Configuration shown in the headline grouped bar chart. PV-only is the cleanest,
# since the battery is the only cost that differs between isolated and cluster.
BARCHART_CONFIG = CONFIG_PV_ONLY

SAVE_CSV = True                       # write the per-subset details and the summary to outputs/


# ============================================================================
#  Derived settings
# ============================================================================
GROUP_GAP = 1.5    # blank x-space left between the N groups on the bar chart

# One colour per configuration, taken from the "colour" field in configuration.py
# so the sensitivity, cluster-size and any future config figures share one source.
CONFIG_COLOURS = {
    CONFIG_PV_ONLY["name"]:   CONFIG_PV_ONLY["colour"],
    CONFIG_PV_WIND["name"]:   CONFIG_PV_WIND["colour"],
    CONFIG_PV_DIESEL["name"]: CONFIG_PV_DIESEL["colour"],
}


def load_generation():
    """Load the real lodge loads and build the aligned PV and wind once.

    Wind is built here regardless of the configuration in configuration.py, so the
    PV+wind row of the table can be produced in the same run as PV-only.
    """
    base_loads, time_index = prepare_model_loads()
    n_households = base_loads.shape[1]
    pv_cluster   = build_pv_for_window(time_index, n_households, SOLAR_FILE, base_loads)
    wind_cluster = build_wind_for_window(time_index, n_households, SOLAR_FILE, base_loads)
    return base_loads, pv_cluster, wind_cluster, n_households


def gen_for_config(config, pv_cluster, wind_cluster):
    """The renewable supply a configuration dispatches: PV, or PV plus wind."""
    return pv_cluster + wind_cluster if config["use_wind"] else pv_cluster


def subset_reductions(config, gen, loads):
    """Battery and LCOE reduction from interconnection for every subset of lodges.

    itertools.combinations(range(n), size) yields each size-element subset of the
    lodge indices exactly once, in a fixed order that runs from nanogrid 1 through
    4 (for pairs: 1+2, 1+3, 1+4, 2+3, 2+4, 3+4). Each subset is a column slice, so
    the isolated and interconnected scenarios run on it unchanged.
    """
    n = loads.shape[1]
    rows = []
    for size in SUBSET_SIZES:
        for subset in itertools.combinations(range(n), size):
            cols = list(subset)
            gen_sub  = gen[:, cols]                       # each nanogrid keeps its own PV (and wind) column
            load_sub = loads[:, cols]                     # and its own load column

            iso_result, _       = run_isolated(gen_sub, load_sub, config=config)
            _, inter_optimal, _ = run_interconnected(gen_sub, load_sub, config=config)

            batt_iso   = iso_result["battery_kWh_total"]      # sum of the individually-sized batteries
            batt_inter = float(inter_optimal["battery_kWh_total"])   # the single shared battery
            lcoe_iso   = iso_result["lcoe_eur_kWh"]
            lcoe_inter = float(inter_optimal["lcoe_eur_kWh"])

            # Percentages so the sizes stay comparable. nan-guard for the diesel
            # case, where a subset can need no isolated battery at all.
            batt_drop = (1 - batt_inter / batt_iso) * 100 if batt_iso > 0 else np.nan
            lcoe_drop = (1 - lcoe_inter / lcoe_iso) * 100

            rows.append({
                "config":        config["name"],
                "size":          size,
                "subset":        " + ".join(LODGE_NAMES[c] for c in cols),   # full lodge names for the table and CSV
                "short":         "+".join(LODGE_TAGS[c] for c in cols),      # short lodge tags (BL, SL, UH, GH) for the chart
                "batt_iso":      batt_iso,
                "batt_inter":    batt_inter,
                "batt_drop_pct": batt_drop,
                "lcoe_iso":      lcoe_iso,
                "lcoe_inter":    lcoe_inter,
                "lcoe_drop_pct": lcoe_drop,
            })

    # Keep the natural combination order (no sorting), so the bar chart reads
    # nanogrid 1 to 4 within each group.
    return pd.DataFrame(rows).reset_index(drop=True)


def print_config_table(df):
    """Per-subset reductions and the by-size averages for one configuration."""
    name = df["config"].iloc[0]
    print("\n" + "=" * 72)
    print(f"  SQ3 cluster size: {name}   (interconnection assumed free)")
    print("=" * 72)
    print(f"  {'subset':28} {'N':>2} {'batt drop %':>12} {'lcoe drop %':>12}")
    for _, r in df.iterrows():
        batt = "   n/a" if np.isnan(r["batt_drop_pct"]) else f"{r['batt_drop_pct']:>12.1f}"
        print(f"  {r['subset']:28} {int(r['size']):>2} {batt} {r['lcoe_drop_pct']:>12.1f}")
    print("-" * 72)
    # groupby("size") splits the rows into one group per N; .mean() averages each
    # column within a group and skips nan, so diesel's blank battery cells drop out.
    by_size = df.groupby("size")[["batt_drop_pct", "lcoe_drop_pct"]].mean()
    counts  = df.groupby("size").size()
    print(f"  {'mean by N':28} {'N':>2} {'batt drop %':>12} {'lcoe drop %':>12}")
    for n in by_size.index:
        print(f"  {'(' + str(counts[n]) + ' groups)':28} {n:>2} "
              f"{by_size.loc[n, 'batt_drop_pct']:>12.1f} {by_size.loc[n, 'lcoe_drop_pct']:>12.1f}")
    overall = df[["batt_drop_pct", "lcoe_drop_pct"]].mean()
    print(f"  {'all subsets':28} {'':>2} {overall['batt_drop_pct']:>12.1f} {overall['lcoe_drop_pct']:>12.1f}")
    print("=" * 72)


def build_summary(results_by_config):
    """Average reduction by N for each configuration, and averaged across them.

    Returns a tidy DataFrame with one row per (configuration, N) and the mean
    battery and LCOE reduction, then adds an "across configs" block that averages
    those config means at each N.
    """
    rows = []
    for name, df in results_by_config.items():
        by_size = df.groupby("size")[["batt_drop_pct", "lcoe_drop_pct"]].mean()
        counts  = df.groupby("size").size()
        for n in by_size.index:
            rows.append({"config": name, "N": int(n), "groups": int(counts[n]),
                         "batt_drop_pct": by_size.loc[n, "batt_drop_pct"],
                         "lcoe_drop_pct": by_size.loc[n, "lcoe_drop_pct"]})

    summary = pd.DataFrame(rows)

    # Across-configuration average at each N: the mean of the per-config means.
    across = (summary.groupby("N")[["batt_drop_pct", "lcoe_drop_pct"]]
                     .mean().reset_index())
    across["config"] = "across configs"
    across["groups"] = summary.groupby("N")["groups"].first().values   # same subset count at each N

    combined = pd.concat([summary, across], ignore_index=True)

    print("\n" + "=" * 72)
    print("  SQ3 summary: mean reduction by cluster size N")
    print("=" * 72)
    print(f"  {'configuration':16} {'N':>2} {'groups':>7} {'batt drop %':>12} {'lcoe drop %':>12}")
    for _, r in combined.iterrows():
        batt = "   n/a" if np.isnan(r["batt_drop_pct"]) else f"{r['batt_drop_pct']:>12.1f}"
        tag = "  <-- average of the three" if r["config"] == "across configs" and r["N"] == SUBSET_SIZES[0] else ""
        print(f"  {r['config']:16} {int(r['N']):>2} {int(r['groups']):>7} {batt} {r['lcoe_drop_pct']:>12.1f}{tag}")
    print("=" * 72)
    return combined


def plot_grouped_by_n(df):
    """Two-panel grouped bar chart for one configuration.

    Top panel is the BESS reduction, bottom panel the LCOE reduction. The eleven
    subsets are placed left to right in three groups (N = 2, 3, 4) with a blank gap
    between groups, and a dashed line marks each group's mean. Every bar carries the
    configuration's own colour; cluster size is read from the group layout and the
    N label above each group, not from the bar colour.
    """
    df = df.reset_index(drop=True)

    # Assign an x position to each bar, inserting a gap whenever the group changes.
    xs = []
    x = 0.0
    prev_size = None
    for _, r in df.iterrows():
        if prev_size is not None and r["size"] != prev_size:
            x += GROUP_GAP
        xs.append(x)
        x += 1.0
        prev_size = r["size"]
    df = df.assign(x=xs)                                  # a new column with the bar positions

    name       = df["config"].iloc[0]
    bar_colour = CONFIG_COLOURS.get(name, "grey")        # this configuration's palette colour

    fig, (ax_b, ax_l) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for ax, col, ylabel in [(ax_b, "batt_drop_pct", "BESS reduction (%)"),
                            (ax_l, "lcoe_drop_pct", "LCOE reduction (%)")]:
        ax.set_axisbelow(True)                           # gridlines behind the bars
        ax.bar(df["x"], df[col], color=bar_colour, edgecolor="white", width=0.8, zorder=3)
        # small value label above each bar
        for xi, yi in zip(df["x"], df[col]):
            if not np.isnan(yi):
                ax.text(xi, yi, f"{yi:.0f}", ha="center", va="bottom", fontsize=7)
        # dashed mean line spanning each N group
        for size in SUBSET_SIZES:
            g = df[df["size"] == size]
            m = g[col].mean()
            ax.hlines(m, g["x"].min() - 0.45, g["x"].max() + 0.45,
                      colors="black", linestyles="--", linewidth=1, zorder=4)
            ax.text(g["x"].max() + 0.5, m, f"mean {m:.0f}%", va="center", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Cluster-size label above each group on the top panel. get_xaxis_transform()
    # gives an axes where x is in data units (the bar positions) and y is a fraction
    # of the panel height (0 at the bottom, 1 at the top), so 1.02 sits just above
    # the top of the panel whatever the bar heights are.
    for size in SUBSET_SIZES:
        g = df[df["size"] == size]
        centre = (g["x"].min() + g["x"].max()) / 2
        ax_b.text(centre, 1.02, f"N = {size}", transform=ax_b.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=9)
        
    # room on the right so the last group's value and "mean" labels stay inside the axes
    ax_l.set_xlim(right=df["x"].max() + 1.5)   # sharex=True, so this widens both panels
    ax_l.set_xticks(df["x"])
    ax_l.set_xticklabels(df["short"], fontsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    safe = name.replace(" ", "").replace("+", "_")
    fname = f"outputs/clustersize_bars_{safe}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


def plot_summary_by_n(summary):
    """Combined mean-reduction-versus-N figure, one line per configuration.

    This is the figure that answers the cluster-size question directly: whether the
    average benefit grows with the number of nanogrids pooled. Diesel is left out of
    the BESS panel, because its generator already suppresses the battery so the
    per-subset changes are near zero and noisy, but it is kept in the LCOE panel,
    where its small pooled-fuel saving is meaningful.
    """
    # .isin(real_names) keeps only the true configuration rows and drops the
    # "across configs" block that build_summary appended.
    real_names = [c["name"] for c in CONFIGS_FOR_TABLE]
    df = summary[summary["config"].isin(real_names)].copy()

    diesel_name = CONFIG_PV_DIESEL["name"]
    batt_names  = [n for n in real_names if n != diesel_name]     # BESS panel: renewables only

    fig, (ax_b, ax_l) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Top panel: BESS reduction, diesel excluded. sort_values("N") orders the
    # points so the line connects N = 2, 3, 4 left to right.
    for name in batt_names:
        d = df[df["config"] == name].sort_values("N")
        ax_b.plot(d["N"], d["batt_drop_pct"], marker="o",
                  color=CONFIG_COLOURS.get(name, "grey"), label=name)
        for x, y in zip(d["N"], d["batt_drop_pct"]):
            ax_b.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                          xytext=(0, 7), ha="center", fontsize=7)
    ax_b.set_ylim(top=df["batt_drop_pct"].max() + 1.5)
    ax_b.set_ylabel("Mean BESS reduction (%)")
    ax_b.set_axisbelow(True)
    ax_b.legend(fontsize=8)
    ax_b.grid(True, linestyle="--", alpha=0.4)

    # Bottom panel: LCOE reduction, all configurations.
    for name in real_names:
        d = df[df["config"] == name].sort_values("N")
        ax_l.plot(d["N"], d["lcoe_drop_pct"], marker="o",
                  color=CONFIG_COLOURS.get(name, "grey"), label=name)
        for x, y in zip(d["N"], d["lcoe_drop_pct"]):
            ax_l.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                          xytext=(0, 7), ha="center", fontsize=7)
    ax_l.set_ylabel("Mean LCOE reduction (%)")
    ax_l.set_ylim(top=df["lcoe_drop_pct"].max() * 1.5)   # proportional headroom so the N = 4 labels clear the top
    ax_l.set_xlabel("Cluster size N (number of nanogrids pooled)")
    ax_l.set_xticks(SUBSET_SIZES)
    ax_l.set_axisbelow(True)
    ax_l.legend(fontsize=8)
    ax_l.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fname = "outputs/clustersize_summary_by_n.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Run
# ============================================================================
def main():
    os.makedirs("outputs", exist_ok=True)

    if CABLING_COSTS_ON:
        print("NOTE: CABLING_COSTS_ON is True. SQ3 assumes free interconnection;")
        print("      set CABLING_COSTS_ON = False in configuration.py for SQ3.\n")

    base_loads, pv_cluster, wind_cluster, n_households = load_generation()

    results_by_config = {}
    for config in CONFIGS_FOR_TABLE:
        gen = gen_for_config(config, pv_cluster, wind_cluster)
        df = subset_reductions(config, gen, base_loads)
        results_by_config[config["name"]] = df
        print_config_table(df)

    summary = build_summary(results_by_config)

    # Headline grouped bar chart for the chosen configuration.
    plot_grouped_by_n(results_by_config[BARCHART_CONFIG["name"]])

    # Combined cross-configuration answer: mean reduction versus N.
    plot_summary_by_n(summary)

    if SAVE_CSV:
        details = pd.concat(results_by_config.values(), ignore_index=True)   # every subset of every configuration
        details.to_csv("outputs/clustersize_subsets.csv", index=False)
        summary.to_csv("outputs/clustersize_summary.csv", index=False)
        print("Saved: outputs/clustersize_subsets.csv and outputs/clustersize_summary.csv")

    return results_by_config, summary


if __name__ == "__main__":
    results_by_config, summary = main()