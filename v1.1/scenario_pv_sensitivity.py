# ============================================================================
#  scenario_pv_sensitivity.py  (SQ1, heavy load scenario)
#  LCOE against cluster PV capacity, isolated and interconnected. At each PV
#  capacity both topologies size their battery again. The extra PV is placed
#  on the Small Lodge, which carries the air-conditioning load ("Small Lodge
#  only"); "Whole cluster" spreads it over all nanogrids by load share. The PV
#  cost depends only on the cluster total, so both variants cost the same at
#  a given capacity.
#  Outputs: LCOE against PV capacity, and a comparison of both topologies at
#  the PV capacity with the lowest isolated LCOE.
#  Run with USE_2025_2026_WINDOW = True in configuration.py.
# ============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from configuration import *
from pv_profile import build_pv_for_window, capacity_per_grid
from wind_profile import build_wind_for_window
from operation import run_isolated, run_interconnected
from load_profile import prepare_model_loads, MODEL_BUILDINGS
from optimiser import lcoe_breakdown


# ============================================================================
#  Settings
# ============================================================================
RUN_PV_WIND = False

RUN_WHOLE_CLUSTER    = False
RUN_SMALL_LODGE_ONLY = True

SMALL_LODGE_NAME = "Small house"

# PV sweep from the installed capacity (PV_CAPACITY_KW * N_NANOGRIDS) to PV_TOTAL_MAX
PV_TOTAL_MAX = 10.2               # kWp, top of the PV sweep
PV_STEP      = 0.5

# Battery step of the inner sweeps [kWh]; None uses BATTERY_STEP_KWH. A coarser step runs faster.
BATTERY_STEP_SWEEP = None


# ============================================================================
#  Derived settings
# ============================================================================
SMALL_IDX = MODEL_BUILDINGS.index(SMALL_LODGE_NAME)     # column of the Small Lodge

CONFIGS_TO_RUN = [CONFIG_PV_BESS] + ([CONFIG_PV_WIND_BESS] if RUN_PV_WIND else [])

VARIANTS = ([("Whole cluster", "whole")]    if RUN_WHOLE_CLUSTER    else []) +\
           ([("Small Lodge only", "small")] if RUN_SMALL_LODGE_ONLY else [])
VARIANT_KEY = dict(VARIANTS)

ISO_COLOUR   = COLOUR_ISOLATED
INTER_COLOUR = COLOUR_INTERCONNECTED

# Smallest bar segment that gets a value label, as a fraction of the axis height
LABEL_MIN_FRAC_BATTERY = 0.02
LABEL_MIN_FRAC_LCOE    = 0.03


def make_battery_sizes():
    """The battery grid handed to the two scenario runners, or None for the config default."""
    if BATTERY_STEP_SWEEP is None:
        return None
    return np.arange(BATTERY_MIN_KWH, BATTERY_MAX_KWH + BATTERY_STEP_SWEEP, BATTERY_STEP_SWEEP)


def scaled_generation(pv_cluster, base_total, total_target, s2_base, variant):
    """PV generation per nanogrid for one cluster PV capacity and allocation variant.

    Generation is linear in capacity, so the installed-capacity generation is
    scaled. "whole" scales every nanogrid; "small" adds all extra PV to the Small Lodge.
    """
    if variant == "whole":
        return pv_cluster * (total_target / base_total)

    pv = pv_cluster.copy()                              # a fresh array so the base is not overwritten
    s2_target = s2_base + (total_target - base_total)   # the Small Lodge absorbs the entire increase
    pv[:, SMALL_IDX] = pv_cluster[:, SMALL_IDX] * (s2_target / s2_base)
    return pv


def run_pv_point(config, gen, loads, total_target, battery_sizes):
    """Isolated and interconnected results at one cluster PV capacity.

    The PV cost follows the cluster total through the PV_CAPACITY_KW override.
    """
    n = loads.shape[1]
    overrides = {"PV_CAPACITY_KW": total_target / n}   # cost follows the new cluster PV total

    _, inter, _ = run_interconnected(gen, loads, config=config,
                                              overrides=overrides, battery_sizes=battery_sizes)
    iso, _      = run_isolated(gen, loads, config=config,
                                        overrides=overrides, battery_sizes=battery_sizes)

    return {
        "total_pv":   total_target,
        "iso_batt":   iso["battery_kWh_total"],
        "iso_lcoe":   iso["lcoe_eur_kWh"],
        "inter_batt": inter["battery_kWh_total"],
        "inter_lcoe": inter["lcoe_eur_kWh"],
    }


def collect(config, pv_cluster, wind_cluster, loads, base_total, s2_base, totals, battery_sizes):
    """Run every PV size for every allocation variant of one configuration."""
    out = {}
    for vlabel, vkey in VARIANTS:
        rows = []
        for total in totals:
            pv  = scaled_generation(pv_cluster, base_total, total, s2_base, vkey)
            gen = pv + wind_cluster if (config["use_wind"] and wind_cluster is not None) else pv
            print(f"\n[{config['name']} | {vlabel} | PV total {total:.1f} kWp]")
            rows.append(run_pv_point(config, gen, loads, total, battery_sizes))
        out[vlabel] = rows
    return out


# ============================================================================
#  Helpers
# ============================================================================
def xy(rows, key):
    """PV capacities and values of one metric."""
    xs = [r["total_pv"] for r in rows]
    ys = [r[key]        for r in rows]
    return xs, ys


def pick_isolated_lowest_lcoe(rows):
    """PV capacity with the lowest isolated LCOE; at_edge is True if it is the top of the sweep."""
    best = min(rows, key=lambda r: r["iso_lcoe"])
    at_edge = abs(best["total_pv"] - rows[-1]["total_pv"]) < 1e-9
    return best["total_pv"], best, at_edge


# ============================================================================
#  Figure: LCOE against PV capacity, one panel per variant, lowest LCOE marked
# ============================================================================
def plot_lcoe_vs_pv(config, results_by_variant, window_label, n_nanogrids):

    variants = list(results_by_variant.keys())
    fig, axes = plt.subplots(1, len(variants), figsize=(6.5 * len(variants), 5.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    inter_items = []
    
    for ax, vlabel in zip(axes, variants):
        rows = results_by_variant[vlabel]

        # Isolated
        xs_i, ys_i = xy(rows, "iso_lcoe")
        ax.plot(xs_i, ys_i, marker="o", color=ISO_COLOUR, label="Isolated")
        i = int(np.argmin(ys_i))
        ax.scatter([xs_i[i]], [ys_i[i]], marker="*", s=200, color=ISO_COLOUR,
                   edgecolor="black", zorder=6)
        ax.annotate(f"{xs_i[i]:.1f} kWp\n{ys_i[i]:.3f}",
                    xy=(xs_i[i], ys_i[i]), xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=ISO_COLOUR)

        # Interconnected
        xs_c, ys_c = xy(rows, "inter_lcoe")
        line_c, = ax.plot(xs_c, ys_c, marker="s", color=INTER_COLOUR, label="Interconnected")
        i = int(np.argmin(ys_c))
        star_c = ax.scatter([xs_c[i]], [ys_c[i]], marker="*", s=200, color=INTER_COLOUR,
                            edgecolor="black", zorder=6)
        note_c = ax.annotate(f"{xs_c[i]:.1f} kWp\n{ys_c[i]:.3f}",
                             xy=(xs_c[i], ys_c[i]), xytext=(0, 10), textcoords="offset points",
                             ha="center", va="bottom", fontsize=8, color=INTER_COLOUR)
        inter_items += [line_c, star_c, note_c]

        ax.set_xlabel("Total PV capacity (kWp)")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].set_ylabel("System LCOE (EUR/kWh)")

    handles = [
        Line2D([0], [0], color=ISO_COLOUR,   marker="o", label="Isolated nanogrids"),
        Line2D([0], [0], color=INTER_COLOUR, marker="s", label="Interconnected cluster"),
        Line2D([0], [0], color="black", marker="*", linestyle="none", markersize=12,
               label="Lowest LCOE"),
    ]
    legend_full = fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
                             bbox_to_anchor=(0.5, 1.02), fontsize=9)
    legend_slide = fig.legend(handles=[handles[0], handles[2]], loc="upper center", ncol=2,
                              frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    legend_slide.set_visible(False)                     # only shown in the slide version

    plt.tight_layout()
    safe = config["name"].replace(" ", "").replace("+", "_")     # tidy the config name for the filename
    fname = f"outputs/pv_sensitivity_lcoe_{safe}_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.png"
    save_figure(plt.gcf(), fname, dpi=150, bbox_inches="tight",
                hide_in_slides=inter_items + [legend_full], show_in_slides=[legend_slide])
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Figure: battery capacity and LCOE composition of both topologies at the PV
#  capacity with the lowest isolated LCOE
# ============================================================================
def plot_lowest_lcoe_comparison(config, results_by_variant, pv_cluster, wind_cluster,
                                loads, base_total, s2_base, battery_sizes,
                                window_label, n_nanogrids):

    safe = config["name"].replace(" ", "").replace("+", "_")

    for vlabel, rows in results_by_variant.items():
        vkey = VARIANT_KEY[vlabel]
        pv_star, star_row, at_edge = pick_isolated_lowest_lcoe(rows)
        if at_edge:
            print(f"  [{vlabel}] NOTE: the lowest-LCOE PV sits at the top of the sweep "
                  f"({pv_star:.1f} kWp). Raise PV_TOTAL_MAX to confirm it is a true minimum.")

        # Run both topologies again at this PV capacity for the full results
        pv  = scaled_generation(pv_cluster, base_total, pv_star, s2_base, vkey)
        gen = pv + wind_cluster if (config["use_wind"] and wind_cluster is not None) else pv
        overrides = {"PV_CAPACITY_KW": pv_star / n_nanogrids}

        iso_result, _   = run_isolated(gen, loads, config=config,
                                                overrides=overrides, battery_sizes=battery_sizes)
        _, inter_opt, _ = run_interconnected(gen, loads, config=config,
                                                      overrides=overrides, battery_sizes=battery_sizes)

        iso_batt   = float(iso_result["battery_kWh_total"])
        inter_batt = float(inter_opt["battery_kWh_total"])
        iso_lcoe   = float(iso_result["lcoe_eur_kWh"])
        inter_lcoe = float(inter_opt["lcoe_eur_kWh"])
        batt_drop  = (1 - inter_batt / iso_batt) * 100 if iso_batt > 0 else float("nan")
        lcoe_drop  = (1 - inter_lcoe / iso_lcoe) * 100

        per_nanogrid   = np.asarray(iso_result["optimal_batteries"], dtype=float)   # isolated battery split by nanogrid
        iso_break   = lcoe_breakdown(iso_result, config, overrides)              # {cost term: EUR/kWh}, sums to iso_lcoe
        inter_break = lcoe_breakdown(inter_opt,  config, overrides)              # same, sums to inter_lcoe

        fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(11, 5.4))

        # Left: battery, isolated stacked per nanogrid
        batt_top = max(iso_batt, inter_batt)               # panel height, sets the label-visibility floor
        bottom = 0.0
        for i in range(len(MODEL_BUILDINGS)):
            colour = NANOGRID_COLOURS[i] if i < len(NANOGRID_COLOURS) else plt.cm.tab10.colors[i]
            label  = NANOGRID_NAMES[i]   if i < len(NANOGRID_NAMES)   else MODEL_BUILDINGS[i]
            ax_b.bar(0, per_nanogrid[i], bottom=bottom, color=colour, edgecolor="black", linewidth=0.6, label=label)
            if per_nanogrid[i] > batt_top * LABEL_MIN_FRAC_BATTERY:   # number a segment tall enough to read
                ax_b.text(0, bottom + per_nanogrid[i] / 2, f"{per_nanogrid[i]:.1f}",
                          ha="center", va="center", fontsize=8)
            bottom += per_nanogrid[i]
        ax_b.bar(1, inter_batt, color=INTER_COLOUR, edgecolor="black", linewidth=0.6, label="Interconnected (shared)")

        ax_b.text(0, iso_batt,   f"{iso_batt:.1f}",   ha="center", va="bottom", fontsize=9)
        ax_b.text(1, inter_batt, f"{inter_batt:.1f}", ha="center", va="bottom", fontsize=9)
        ax_b.set_xticks([0, 1])
        ax_b.set_xticklabels(["Isolated", "Interconnected"])
        ax_b.set_ylabel("BESS capacity (kWh)")
        ax_b.set_title(f"BESS capacity, PV {pv_star:.1f} kWp   (-{batt_drop:.0f}%)")
        ax_b.legend(fontsize=7, loc="upper right")
        ax_b.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax_b.set_axisbelow(True)

        # Right: LCOE composition, terms that are zero in both bars left out
        lcoe_top = max(iso_lcoe, inter_lcoe)               # panel height, sets the label-visibility floor
        used_terms = []
        for x, break_dict in [(0, iso_break), (1, inter_break)]:
            bottom = 0.0
            for term, value in break_dict.items():
                if value <= 1e-9:
                    continue
                ax_c.bar(x, value, bottom=bottom, color=COST_COLOURS.get(term, "lightgrey"),
                         edgecolor="black", linewidth=0.5)
                if value > lcoe_top * LABEL_MIN_FRAC_LCOE:      # number a segment tall enough to read
                    ax_c.text(x, bottom + value / 2, f"{value:.2f}",
                              ha="center", va="center", fontsize=7)
                bottom += value
                if term not in used_terms:
                    used_terms.append(term)

        ax_c.text(0, iso_lcoe,   f"{iso_lcoe:.3f}",   ha="center", va="bottom", fontsize=9)
        ax_c.text(1, inter_lcoe, f"{inter_lcoe:.3f}", ha="center", va="bottom", fontsize=9)
        ax_c.set_xticks([0, 1])
        ax_c.set_xticklabels(["Isolated", "Interconnected"])
        ax_c.set_ylabel("LCOE (EUR/kWh)")
        ax_c.set_title(f"LCOE composition, PV {pv_star:.1f} kWp   (-{lcoe_drop:.0f}%)")
        legend_handles = [Patch(facecolor=COST_COLOURS.get(t, "lightgrey"), edgecolor="black", label=COST_LABELS.get(t, t)) for t in used_terms]
        ax_c.legend(handles=legend_handles, fontsize=7, loc="upper right")
        ax_c.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax_c.set_axisbelow(True)

        plt.tight_layout()
        fname = f"outputs/pv_lowest_lcoe_comparison_{safe}_{vkey}_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()

        print(f"\n  Lowest-LCOE-PV comparison [{vlabel}] at {pv_star:.1f} kWp:")
        print(f"    battery {iso_batt:.0f} -> {inter_batt:.0f} kWh  (-{batt_drop:.0f}%)")
        print(f"    LCOE    {iso_lcoe:.3f} -> {inter_lcoe:.3f} EUR/kWh  (-{lcoe_drop:.0f}%)")
        print(f"  Saved: {fname}")


def print_table(config, results_by_variant):
    """Print battery and LCOE per PV capacity and variant."""
    for vlabel, rows in results_by_variant.items():
        print("\n" + "=" * 66)
        print(f"  {config['name']}  |  {vlabel}")
        print("=" * 66)
        print(f"  {'PV kWp':>7}{'iso batt':>11}{'iso LCOE':>11}{'int batt':>11}{'int LCOE':>11}")
        print("-" * 66)
        for r in rows:
            print(f"  {r['total_pv']:>7.1f}"
                  f"{r['iso_batt']:>11.1f}{r['iso_lcoe']:>11.3f}"
                  f"{r['inter_batt']:>11.1f}{r['inter_lcoe']:>11.3f}")
        print("=" * 66)


# ============================================================================
#  Run
# ============================================================================
def main():
    os.makedirs("outputs", exist_ok=True)

    if not USE_2025_2026_WINDOW:
        print("NOTE: USE_2025_2026_WINDOW is False, so the small lodge has no A/C load in this")
        print("      window and the PV sensitivity is far less meaningful. The SQ1 heavy case")
        print("      expects the heavy-load window (set USE_2025_2026_WINDOW = True, restart kernel).")

    base_loads, time_index = prepare_model_loads()
    n_nanogrids = base_loads.shape[1]

    pv_cluster   = build_pv_for_window(time_index, n_nanogrids, SOLAR_FILE, base_loads)
    wind_cluster = build_wind_for_window(time_index, n_nanogrids, SOLAR_FILE, base_loads) if RUN_PV_WIND else None

    base_per_grid = PV_CAPACITY_KW                       # average kWp per nanogrid at the real system
    base_total    = base_per_grid * n_nanogrids
    s2_base       = capacity_per_grid(base_loads, PV_CAPACITY_KW, ALLOCATION)[SMALL_IDX]   # Small Lodge base PV [kWp]

    if PV_TOTAL_MAX <= base_total:
        raise ValueError(f"PV_TOTAL_MAX ({PV_TOTAL_MAX} kWp) must exceed the base cluster PV "
                         f"({base_total:.2f} kWp). Raise PV_TOTAL_MAX.")

    # PV capacities; the extra half step keeps PV_TOTAL_MAX in the range
    totals = np.arange(base_total, PV_TOTAL_MAX + 0.5 * PV_STEP, PV_STEP)
    battery_sizes = make_battery_sizes()
    window_label = f"{MODEL_WINDOW_START} to {MODEL_WINDOW_END}"

    step_note = f"{BATTERY_STEP_SWEEP} kWh (coarse)" if BATTERY_STEP_SWEEP is not None else f"{BATTERY_STEP_KWH} kWh (config)"
    print(f"\nPV sweep: {len(totals)} sizes from {base_total:.2f} to {PV_TOTAL_MAX:.1f} kWp in "
          f"{PV_STEP} kWp steps | battery step {step_note} | small lodge = {SMALL_LODGE_NAME} (column {SMALL_IDX})")

    for config in CONFIGS_TO_RUN:
        results = collect(config, pv_cluster, wind_cluster, base_loads,
                          base_total, s2_base, totals, battery_sizes)
        print_table(config, results)
        plot_lcoe_vs_pv(config, results, window_label, n_nanogrids)
        plot_lowest_lcoe_comparison(config, results, pv_cluster, wind_cluster,
                                    base_loads, base_total, s2_base, battery_sizes,
                                    window_label, n_nanogrids)

    return results


if __name__ == "__main__":
    results = main()
