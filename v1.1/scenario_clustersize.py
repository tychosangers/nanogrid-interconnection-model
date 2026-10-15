# ============================================================================
#  scenario_clustersize.py  (SQ2)
#  Interconnection benefit per cluster size. For every subset of the four
#  nanogrids (six pairs, four triples, one cluster of four), isolated and
#  interconnected operation are compared and the battery and LCOE reductions
#  are grouped by cluster size N.
#  Outputs: a bar chart per subset for one configuration, and the mean
#  reductions per N for all configurations (table and line figure).
#  Run with CABLING_COSTS_ON = False; wind is built here for all configurations.
# ============================================================================

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configuration import (CONFIG_PV_BESS, CONFIG_PV_WIND_BESS, CONFIG_PV_DIESEL_BESS,
                           CABLING_COSTS_ON, SOLAR_FILE, NANOGRID_NAMES, NANOGRID_TAGS, save_figure)
from pv_profile import build_pv_for_window
from wind_profile import build_wind_for_window
from load_profile import prepare_model_loads
from operation import run_isolated, run_interconnected


# ============================================================================
#  Settings
# ============================================================================
# Subset sizes: 6 pairs, 4 triples and 1 cluster of four
SUBSET_SIZES = [2, 3, 4]

# Configurations in the table of mean reductions
CONFIGS_FOR_TABLE = [CONFIG_PV_BESS, CONFIG_PV_WIND_BESS, CONFIG_PV_DIESEL_BESS]

# Configuration shown in the bar chart per subset
BARCHART_CONFIG = CONFIG_PV_BESS

SAVE_CSV = True


# ============================================================================
#  Derived settings
# ============================================================================
GROUP_GAP = 1.5

# Configuration colours from configuration.py
CONFIG_COLOURS = {
    CONFIG_PV_BESS["name"]:   CONFIG_PV_BESS["colour"],
    CONFIG_PV_WIND_BESS["name"]:   CONFIG_PV_WIND_BESS["colour"],
    CONFIG_PV_DIESEL_BESS["name"]: CONFIG_PV_DIESEL_BESS["colour"],
}


def load_generation():
    """Load the nanogrid loads and build PV and wind generation for all configurations."""
    base_loads, time_index = prepare_model_loads()
    n_nanogrids = base_loads.shape[1]
    pv_cluster   = build_pv_for_window(time_index, n_nanogrids, SOLAR_FILE, base_loads)
    wind_cluster = build_wind_for_window(time_index, n_nanogrids, SOLAR_FILE, base_loads)
    return base_loads, pv_cluster, wind_cluster, n_nanogrids


def gen_for_config(config, pv_cluster, wind_cluster):
    """The renewable supply a configuration dispatches: PV, or PV plus wind."""
    return pv_cluster + wind_cluster if config["use_wind"] else pv_cluster


def subset_reductions(config, gen, loads):
    """Battery and LCOE reduction from interconnection for every subset of nanogrids [%].

    Subsets are column slices of the load and generation arrays, in the order
    of itertools.combinations (for pairs: 1+2, 1+3, 1+4, 2+3, 2+4, 3+4).
    """
    n = loads.shape[1]
    rows = []
    for size in SUBSET_SIZES:
        for subset in itertools.combinations(range(n), size):
            cols = list(subset)
            gen_sub  = gen[:, cols]
            load_sub = loads[:, cols]                     # and its own load column

            iso_result, _       = run_isolated(gen_sub, load_sub, config=config)
            _, inter_optimal, _ = run_interconnected(gen_sub, load_sub, config=config)

            batt_iso   = iso_result["battery_kWh_total"]      # sum of the individually-sized batteries
            batt_inter = float(inter_optimal["battery_kWh_total"])   # the single shared battery
            lcoe_iso   = iso_result["lcoe_eur_kWh"]
            lcoe_inter = float(inter_optimal["lcoe_eur_kWh"])

            # NaN when the isolated subset needs no battery (diesel configuration)
            batt_drop = (1 - batt_inter / batt_iso) * 100 if batt_iso > 0 else np.nan
            lcoe_drop = (1 - lcoe_inter / lcoe_iso) * 100

            rows.append({
                "config":        config["name"],
                "size":          size,
                "subset":        " + ".join(NANOGRID_NAMES[c] for c in cols),
                "short":         "+".join(NANOGRID_TAGS[c] for c in cols),
                "batt_iso":      batt_iso,
                "batt_inter":    batt_inter,
                "batt_drop_pct": batt_drop,
                "lcoe_iso":      lcoe_iso,
                "lcoe_inter":    lcoe_inter,
                "lcoe_drop_pct": lcoe_drop,
            })

    return pd.DataFrame(rows).reset_index(drop=True)


def print_config_table(df):
    """Per-subset reductions and the by-size averages for one configuration."""
    name = df["config"].iloc[0]
    print("\n" + "=" * 72)
    print(f"  SQ2 cluster size: {name}   (interconnection assumed free)")
    print("=" * 72)
    print(f"  {'subset':28} {'N':>2} {'batt drop %':>12} {'lcoe drop %':>12}")
    for _, r in df.iterrows():
        batt = "   n/a" if np.isnan(r["batt_drop_pct"]) else f"{r['batt_drop_pct']:>12.1f}"
        print(f"  {r['subset']:28} {int(r['size']):>2} {batt} {r['lcoe_drop_pct']:>12.1f}")
    print("-" * 72)
    # Mean per cluster size (NaN values are skipped)
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
    """Mean battery and LCOE reduction per configuration and N, plus the mean across configurations."""
    rows = []
    for name, df in results_by_config.items():
        by_size = df.groupby("size")[["batt_drop_pct", "lcoe_drop_pct"]].mean()
        counts  = df.groupby("size").size()
        for n in by_size.index:
            rows.append({"config": name, "N": int(n), "groups": int(counts[n]),
                         "batt_drop_pct": by_size.loc[n, "batt_drop_pct"],
                         "lcoe_drop_pct": by_size.loc[n, "lcoe_drop_pct"]})

    summary = pd.DataFrame(rows)

    across = (summary.groupby("N")[["batt_drop_pct", "lcoe_drop_pct"]]
                     .mean().reset_index())
    across["config"] = "across configs"
    across["groups"] = summary.groupby("N")["groups"].first().values   # same subset count at each N

    combined = pd.concat([summary, across], ignore_index=True)

    print("\n" + "=" * 72)
    print("  SQ2 summary: mean reduction by cluster size N")
    print("=" * 72)
    print(f"  {'configuration':16} {'N':>2} {'groups':>7} {'batt drop %':>12} {'lcoe drop %':>12}")
    for _, r in combined.iterrows():
        batt = "   n/a" if np.isnan(r["batt_drop_pct"]) else f"{r['batt_drop_pct']:>12.1f}"
        tag = "  <-- average of the three" if r["config"] == "across configs" and r["N"] == SUBSET_SIZES[0] else ""
        print(f"  {r['config']:16} {int(r['N']):>2} {int(r['groups']):>7} {batt} {r['lcoe_drop_pct']:>12.1f}{tag}")
    print("=" * 72)
    return combined


def plot_grouped_by_n(df):
    """BESS reduction (top) and LCOE reduction (bottom) per subset, grouped by N, with the mean per group."""
    df = df.reset_index(drop=True)

    # x positions with a gap between groups
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
        # Value labels
        for xi, yi in zip(df["x"], df[col]):
            if not np.isnan(yi):
                ax.text(xi, yi, f"{yi:.0f}", ha="center", va="bottom", fontsize=7)
        # Mean line per group
        for size in SUBSET_SIZES:
            g = df[df["size"] == size]
            if len(g) > 1:                               # a mean of one bar adds nothing, so skip N = 4
                m = g[col].mean()
                ax.hlines(m, g["x"].min() - 0.45, g["x"].max() + 0.45,
                          colors="black", linestyles="--", linewidth=1, zorder=4)
                ax.text(g["x"].max() + 0.5, m, f"mean {m:.0f}%", va="center", fontsize=7)
        ax.set_ylabel(ylabel)
        lo = min(0, df[col].min()) * 1.10
        hi = max(0, df[col].max()) * 1.10
        ax.set_ylim(lo, hi)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # N label above each group (x in data units, y as fraction of the panel)
    for size in SUBSET_SIZES:
        g = df[df["size"] == size]
        centre = (g["x"].min() + g["x"].max()) / 2
        ax_b.text(centre, 1.02, f"N = {size}", transform=ax_b.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=9)
        
    ax_l.set_xlim(right=df["x"].max() + 1.5)   # sharex=True, so this widens both panels
    ax_l.set_xticks(df["x"])
    ax_l.set_xticklabels(df["short"], rotation=45, ha="right",
                         rotation_mode="anchor", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    safe = name.replace(" ", "").replace("+", "_")
    fname = f"outputs/clustersize_bars_{safe}.png"
    save_figure(plt.gcf(), fname, scale=1.6, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


def plot_summary_by_n(summary):
    """Mean BESS reduction (top, renewable configurations) and LCOE reduction (bottom) versus N."""
    # Configuration rows only
    real_names = [c["name"] for c in CONFIGS_FOR_TABLE]
    df = summary[summary["config"].isin(real_names)].copy()

    diesel_name = CONFIG_PV_DIESEL_BESS["name"]
    batt_names  = [n for n in real_names if n != diesel_name]     # BESS panel: renewables only

    fig, (ax_b, ax_l) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Top: BESS reduction, without the diesel configuration
    for name in batt_names:
        d = df[df["config"] == name].sort_values("N")
        below = name == CONFIG_PV_WIND_BESS["name"]
        ax_b.plot(d["N"], d["batt_drop_pct"], marker="o",
                  color=CONFIG_COLOURS.get(name, "grey"), label=name)
        for x, y in zip(d["N"], d["batt_drop_pct"]):
            ax_b.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                          xytext=(0, -10 if below else 7),
                          ha="center", va="top" if below else "bottom", fontsize=7)
    renew = df[df["config"].isin(batt_names)]["batt_drop_pct"]
    ax_b.set_ylim(renew.min() - 3, renew.max() * 1.10)
    ax_b.set_ylabel("Mean BESS reduction (%)")
    ax_b.set_axisbelow(True)
    ax_b.legend(fontsize=8)
    ax_b.grid(True, linestyle="--", alpha=0.4)

    # Bottom: LCOE reduction, all configurations
    for name in real_names:
        d = df[df["config"] == name].sort_values("N")
        below = name == CONFIG_PV_WIND_BESS["name"]
        ax_l.plot(d["N"], d["lcoe_drop_pct"], marker="o",
                  color=CONFIG_COLOURS.get(name, "grey"), label=name)
        for x, y in zip(d["N"], d["lcoe_drop_pct"]):
            ax_l.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                          xytext=(0, -10 if below else 7),
                          ha="center", va="top" if below else "bottom", fontsize=7)
    ax_l.set_ylabel("Mean LCOE reduction (%)")
    ax_l.set_ylim(top=df["lcoe_drop_pct"].max() * 1.5)
    ax_l.set_xlabel("Cluster size N (number of nanogrids pooled)")
    ax_l.set_xticks(SUBSET_SIZES)
    ax_l.set_axisbelow(True)
    ax_l.legend(fontsize=8)
    ax_l.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fname = "outputs/clustersize_summary_by_n.png"
    save_figure(plt.gcf(), fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Run
# ============================================================================
def main():
    os.makedirs("outputs", exist_ok=True)

    if CABLING_COSTS_ON:
        print("NOTE: CABLING_COSTS_ON is True. SQ2 assumes free interconnection;")
        print("      set CABLING_COSTS_ON = False in configuration.py for SQ2.\n")

    base_loads, pv_cluster, wind_cluster, n_nanogrids = load_generation()

    results_by_config = {}
    for config in CONFIGS_FOR_TABLE:
        gen = gen_for_config(config, pv_cluster, wind_cluster)
        df = subset_reductions(config, gen, base_loads)
        results_by_config[config["name"]] = df
        print_config_table(df)

    summary = build_summary(results_by_config)

    plot_grouped_by_n(results_by_config[BARCHART_CONFIG["name"]])

    plot_summary_by_n(summary)

    if SAVE_CSV:
        details = pd.concat(results_by_config.values(), ignore_index=True)   # every subset of every configuration
        details.to_csv("outputs/clustersize_subsets.csv", index=False)
        summary.to_csv("outputs/clustersize_summary.csv", index=False)
        print("Saved: outputs/clustersize_subsets.csv and outputs/clustersize_summary.csv")

    return results_by_config, summary


if __name__ == "__main__":
    results_by_config, summary = main()
