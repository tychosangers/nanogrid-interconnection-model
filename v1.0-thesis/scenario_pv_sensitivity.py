# ============================================================================
#  scenario_pv_sensitivity.py
#  SQ1 heavy-load robustness: how the heavy-load (air-conditioning) case responds
#  to PV capacity.
#
#  In the heavy-load window the small lodge carries a large winter A/C load while
#  the PV is fixed at the real installed size. That fixed PV is why the small lodge
#  needs an enormous battery when it stands alone. This script relaxes that by
#  sweeping the cluster PV capacity upward and, at each PV size, letting BOTH
#  topologies re-optimise their battery. Raising PV cuts the battery need but adds
#  PV capex, so the isolated LCOE-versus-PV curve is U-shaped: its lowest point is
#  the lowest-LCOE PV size for the isolated case. The interconnected curve is drawn
#  alongside to show what pooling changes.
#
#  At low PV the battery the model sizes is very large and the LCOE correspondingly
#  high. These points are plotted exactly as the model returns them, with no
#  filtering or judgement applied.
#
#  The headline variant is "Small lodge only": the extra PV is added only to the
#  lodge that carries the A/C load, which is the realistic fix (an owner adds
#  panels on the building that gained the load, not across the whole site) and the
#  fairer test (the isolated fleet is given the cheapest way to meet its load
#  before being compared, so any remaining benefit of interconnection is real).
#  "Whole cluster" is kept as an optional contrast panel that spreads the extra PV
#  across all four lodges by load share. The cost side depends only on the cluster
#  PV total, so the two variants cost the same at a given point on the x axis; any
#  LCOE difference between them comes purely from WHERE the PV sits.
#
#  This is a self-contained, press-run script. It is meant for the heavy-load
#  window, so set USE_2025_2026_WINDOW = True in configuration.py and restart the
#  kernel before running. It reuses run_isolated and run_interconnected unchanged,
#  so the numbers here match the SQ1 comparison figure.
# ============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch                 # a plain coloured rectangle, used to build the composition legend by hand
from matplotlib.lines import Line2D                  # a plain marker, used to add the lowest-LCOE star to the sweep legend

from configuration import *
from pv_profile import build_pv_for_window, capacity_per_grid
from wind_profile import build_wind_for_window
from operation import run_isolated, run_interconnected   # the two operating modes: isolated fleet and interconnected cluster
from load_profile import prepare_model_loads, MODEL_BUILDINGS
from optimiser import lcoe_breakdown                 # splits a finished scenario LCOE into its cost terms, for the composition bars


# ============================================================================
#  TOGGLES
# ============================================================================
RUN_PV_WIND = False               # add a second run for the PV-Wind-BESS configuration (kept off: wind does not isolate the PV lever, see header)

RUN_WHOLE_CLUSTER    = False      # variant 1 (optional contrast): spread the extra PV across all lodges by load share
RUN_SMALL_LODGE_ONLY = True       # variant 2 (headline): add the extra PV only to the small lodge

SMALL_LODGE_NAME = "Small house"  # the lodge that carries the heavy-load (A/C) demand

# PV sweep. The base total is read from configuration.py (PV_capacity_kW times the
# number of nanogrids), so the sweep always starts at the real installed size and
# rises to PV_TOTAL_MAX in fixed steps. A narrow range is used here because the
# small-lodge lowest-LCOE point sits well inside it; widen PV_TOTAL_MAX if you move
# to the whole-cluster variant, whose lowest-LCOE point lies further out.
PV_TOTAL_MAX = 10.2               # kWp, top of the PV sweep
PV_STEP      = 0.5                # kWp, spacing between PV sizes from the base total up to PV_TOTAL_MAX

# Battery step for the inner sweeps. The dispatch runs step by step, so a full PV
# grid is heavy; a coarser battery step here makes a quick pass much faster while
# keeping the same battery range as configuration.py (so low-PV points, which need
# a large battery, are not truncated). Set to None to use the configuration.py step
# for the final, high-resolution figure.
BATTERY_STEP_SWEEP = None          # kWh, or None to fall back to Battery_step_kWh


# ============================================================================
#  Derived settings
# ============================================================================
SMALL_IDX = MODEL_BUILDINGS.index(SMALL_LODGE_NAME)     # column of the small lodge in the load array

CONFIGS_TO_RUN = [CONFIG_PV_ONLY] + ([CONFIG_PV_WIND] if RUN_PV_WIND else [])

VARIANTS = ([("Whole cluster", "whole")]    if RUN_WHOLE_CLUSTER    else []) + \
           ([("Small lodge only", "small")] if RUN_SMALL_LODGE_ONLY else [])
VARIANT_KEY = dict(VARIANTS)                            # label -> short key ("whole"/"small"), used when rebuilding generation

# Topology colours, taken from configuration.py so this file matches the SQ1
# comparison figures: slate for the isolated fleet, reserved purple for the cluster.
ISO_COLOUR   = COLOUR_ISOLATED
INTER_COLOUR = COLOUR_INTERCONNECTED

# The stacked isolated battery bar splits by lodge, so it uses the shared lodge
# colours and names from configuration.py. Cost-term colours for the composition
# bars come from the shared COST_COLOURS in configuration.py as well (already in
# scope via the wildcard import above), so a lodge and a cost term keep the same
# colour across every figure.

# Smallest segment that still gets a numeric label, as a fraction of the panel
# height, matching scenario_comparison.py. Lower to label thinner segments.
LABEL_MIN_FRAC_BATTERY = 0.02   # stacked isolated battery panel (Figure 21, left)
LABEL_MIN_FRAC_LCOE    = 0.03   # LCOE composition panel (Figure 21, right)


def make_battery_sizes():
    """The battery grid handed to the two scenario runners, or None for the config default."""
    if BATTERY_STEP_SWEEP is None:
        return None
    # Same min and max as configuration.py, only the step is coarser.
    return np.arange(Battery_min_kWh, Battery_max_kWh + BATTERY_STEP_SWEEP, BATTERY_STEP_SWEEP)


def scaled_generation(pv_cluster, base_total, total_target, s2_base, variant):
    """Per-lodge PV generation for one PV size and one allocation variant.

    pv_cluster is the generation at the real installed PV. Generation is linear in
    installed capacity, so multiplying the base array reproduces a larger array
    exactly (for the load-share allocation used here). "whole" scales every lodge
    together; "small" gives the whole increase to the small lodge alone.
    """
    if variant == "whole":
        return pv_cluster * (total_target / base_total)

    # Small lodge only: keep the other lodges as they are and grow the small lodge.
    pv = pv_cluster.copy()                              # a fresh array so the base is not overwritten
    s2_target = s2_base + (total_target - base_total)   # the small lodge absorbs the entire increase
    pv[:, SMALL_IDX] = pv_cluster[:, SMALL_IDX] * (s2_target / s2_base)
    return pv


def run_pv_point(config, gen, loads, total_target, battery_sizes):
    """One isolated-vs-interconnected comparison at a single PV size.

    The cost side reads PV capacity from the override below (cluster total divided
    by the number of nanogrids), so it moves in step with the generation that was
    scaled above. Both variants therefore cost the same at a given total PV.
    """
    n = loads.shape[1]
    overrides = {"PV_capacity_kW": total_target / n}   # cost follows the new cluster PV total

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
#  Small readers used by the console table and the figures
# ============================================================================
def xy(rows, key):
    """x (PV size) and y (the chosen metric) lists for one line, all points kept."""
    xs = [r["total_pv"] for r in rows]
    ys = [r[key]        for r in rows]
    return xs, ys


def pick_isolated_lowest_lcoe(rows):
    """The PV size that minimises the isolated LCOE across the sweep.

    min(..., key=...) scans the rows and returns the one with the smallest isolated
    LCOE; at_edge flags when that PV size sits at the top of the sweep, which means
    the true lowest-LCOE point may lie beyond PV_TOTAL_MAX and the range should be
    widened.
    """
    best = min(rows, key=lambda r: r["iso_lcoe"])
    at_edge = abs(best["total_pv"] - rows[-1]["total_pv"]) < 1e-9
    return best["total_pv"], best, at_edge


# ============================================================================
#  Figure 20: LCOE against PV capacity, one panel per variant. Each line's lowest
#  point (its lowest-LCOE PV size) is marked with a star. Every PV size is plotted
#  as the model returns it, including the low-PV points where the battery, and
#  therefore the LCOE, is very large.
# ============================================================================
def plot_lcoe_vs_pv(config, results_by_variant, window_label, n_households):

    variants = list(results_by_variant.keys())
    fig, axes = plt.subplots(1, len(variants), figsize=(6.5 * len(variants), 5.2),
                             sharey=True, squeeze=False)
    axes = axes[0]                                      # squeeze=False gives a 2-D array; take the single row

    for ax, vlabel in zip(axes, variants):
        rows = results_by_variant[vlabel]

        # Isolated line.
        xs_i, ys_i = xy(rows, "iso_lcoe")
        ax.plot(xs_i, ys_i, marker="o", color=ISO_COLOUR, label="Isolated")
        i = int(np.argmin(ys_i))                        # lowest LCOE = lowest-LCOE PV size for this line
        ax.scatter([xs_i[i]], [ys_i[i]], marker="*", s=200, color=ISO_COLOUR,
                   edgecolor="black", zorder=6)
        ax.annotate(f"{xs_i[i]:.1f} kWp\n{ys_i[i]:.3f}",
                    xy=(xs_i[i], ys_i[i]), xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=ISO_COLOUR)

        # Interconnected line.
        xs_c, ys_c = xy(rows, "inter_lcoe")
        ax.plot(xs_c, ys_c, marker="s", color=INTER_COLOUR, label="Interconnected")
        i = int(np.argmin(ys_c))
        ax.scatter([xs_c[i]], [ys_c[i]], marker="*", s=200, color=INTER_COLOUR,
                   edgecolor="black", zorder=6)
        ax.annotate(f"{xs_c[i]:.1f} kWp\n{ys_c[i]:.3f}",
                    xy=(xs_c[i], ys_c[i]), xytext=(0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INTER_COLOUR)

        ax.set_xlabel("Total PV capacity (kWp)")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].set_ylabel("System LCOE (EUR/kWh)")

    # One legend for the whole figure, built by hand so the star gets an entry
    # alongside the two lines.
    handles = [
        Line2D([0], [0], color=ISO_COLOUR,   marker="o", label="Isolated nanogrids"),
        Line2D([0], [0], color=INTER_COLOUR, marker="s", label="Interconnected cluster"),
        Line2D([0], [0], color="black", marker="*", linestyle="none", markersize=12,
               label="Lowest LCOE"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)

    plt.tight_layout()
    safe = config["name"].replace(" ", "").replace("+", "_")     # tidy the config name for the filename
    fname = f"outputs/pv_sensitivity_lcoe_{safe}_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Figure 21: comparison at the lowest-LCOE PV. At the PV size that minimises the
#  isolated LCOE both topologies are re-run at full detail and shown side by side,
#  in the SQ1 style: the isolated battery stacked by lodge next to the shared
#  interconnected battery, and the LCOE split into its cost terms. Reading both
#  topologies at the SAME PV makes the gap an honest same-configuration gap.
# ============================================================================
def plot_lowest_lcoe_comparison(config, results_by_variant, pv_cluster, wind_cluster,
                                loads, base_total, s2_base, battery_sizes,
                                window_label, n_households):

    safe = config["name"].replace(" ", "").replace("+", "_")

    for vlabel, rows in results_by_variant.items():
        vkey = VARIANT_KEY[vlabel]
        pv_star, star_row, at_edge = pick_isolated_lowest_lcoe(rows)
        if at_edge:
            print(f"  [{vlabel}] NOTE: the lowest-LCOE PV sits at the top of the sweep "
                  f"({pv_star:.1f} kWp). Raise PV_TOTAL_MAX to confirm it is a true minimum.")

        # Rebuild the generation at the lowest-LCOE PV and re-run both topologies with
        # the full result dicts (the sweep kept only totals). The cost override makes
        # the PV capex follow that PV size, exactly as in the sweep.
        pv  = scaled_generation(pv_cluster, base_total, pv_star, s2_base, vkey)
        gen = pv + wind_cluster if (config["use_wind"] and wind_cluster is not None) else pv
        overrides = {"PV_capacity_kW": pv_star / n_households}

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

        per_lodge   = np.asarray(iso_result["optimal_batteries"], dtype=float)   # isolated battery split by lodge
        iso_break   = lcoe_breakdown(iso_result, config, overrides)              # {cost term: EUR/kWh}, sums to iso_lcoe
        inter_break = lcoe_breakdown(inter_opt,  config, overrides)              # same, sums to inter_lcoe

        # ---- figure: battery on the left, LCOE composition on the right ----
        fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(11, 5.4))

        # Left: isolated battery stacked by lodge, interconnected as one shared bar.
        batt_top = max(iso_batt, inter_batt)               # panel height, sets the label-visibility floor
        bottom = 0.0
        for i in range(len(MODEL_BUILDINGS)):
            colour = LODGE_COLOURS[i] if i < len(LODGE_COLOURS) else plt.cm.tab10.colors[i]
            label  = LODGE_NAMES[i]   if i < len(LODGE_NAMES)   else MODEL_BUILDINGS[i]
            ax_b.bar(0, per_lodge[i], bottom=bottom, color=colour, edgecolor="black", linewidth=0.6, label=label)
            if per_lodge[i] > batt_top * LABEL_MIN_FRAC_BATTERY:   # number a segment tall enough to read
                ax_b.text(0, bottom + per_lodge[i] / 2, f"{per_lodge[i]:.1f}",
                          ha="center", va="center", fontsize=8)
            bottom += per_lodge[i]
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

        # Right: LCOE composition. Loop the fixed stacking order (the dict keeps its
        # order), skip terms that are zero in both bars, and remember which appeared
        # so the legend only lists real cost terms.
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
    """A compact per-variant table of battery and LCOE against PV size."""
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
    n_households = base_loads.shape[1]

    pv_cluster   = build_pv_for_window(time_index, n_households, SOLAR_FILE, base_loads)
    wind_cluster = build_wind_for_window(time_index, n_households, SOLAR_FILE, base_loads) if RUN_PV_WIND else None

    base_per_grid = PV_capacity_kW                       # average kWp per lodge at the real system
    base_total    = base_per_grid * n_households         # the real installed cluster PV, start of the sweep
    s2_base       = capacity_per_grid(base_loads, PV_capacity_kW, ALLOCATION)[SMALL_IDX]   # small lodge's base PV [kWp]

    if PV_TOTAL_MAX <= base_total:
        raise ValueError(f"PV_TOTAL_MAX ({PV_TOTAL_MAX} kWp) must exceed the base cluster PV "
                         f"({base_total:.2f} kWp). Raise PV_TOTAL_MAX.")

    # PV sizes from the real system up to the ceiling in fixed steps. The small
    # +0.5*step keeps PV_TOTAL_MAX itself in the grid despite floating-point drift.
    totals = np.arange(base_total, PV_TOTAL_MAX + 0.5 * PV_STEP, PV_STEP)
    battery_sizes = make_battery_sizes()
    window_label = f"{MODEL_WINDOW_START} to {MODEL_WINDOW_END}"

    step_note = f"{BATTERY_STEP_SWEEP} kWh (coarse)" if BATTERY_STEP_SWEEP is not None else f"{Battery_step_kWh} kWh (config)"
    print(f"\nPV sweep: {len(totals)} sizes from {base_total:.2f} to {PV_TOTAL_MAX:.1f} kWp in "
          f"{PV_STEP} kWp steps | battery step {step_note} | small lodge = {SMALL_LODGE_NAME} (column {SMALL_IDX})")

    for config in CONFIGS_TO_RUN:
        results = collect(config, pv_cluster, wind_cluster, base_loads,
                          base_total, s2_base, totals, battery_sizes)
        print_table(config, results)
        plot_lcoe_vs_pv(config, results, window_label, n_households)
        plot_lowest_lcoe_comparison(config, results, pv_cluster, wind_cluster,
                                    base_loads, base_total, s2_base, battery_sizes,
                                    window_label, n_households)

    return results     # the last configuration's results, handy in the variable explorer


if __name__ == "__main__":
    results = main()