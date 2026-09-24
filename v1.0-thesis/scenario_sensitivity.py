# ============================================================================
#  scenario_sensitivity.py
#  Sensitivity figures for SQ1, on the basic (2025) window.
#
#  HEADLINE, three figures that build one argument (across all configurations):
#     1. A deviation tornado of the LCOE reduction (percentage points). Zero in the
#        middle is each configuration's base, the left arm is where the benefit
#        falls and the right arm where it rises. This sets up the apparent picture.
#     2. The SAME tornado for the absolute LCOE gap (EUR/kWh), same parameter order.
#        Here PV, wind and diesel capex collapse to zero because a cost identical in
#        both topologies cancels in the difference, leaving only battery capex,
#        discount rate and diesel fuel. This is the reveal, and the reason the
#        benefit is reported as an absolute gap.
#     3. The benefit against battery capex, the one surviving driver, drawn in both
#        the absolute gap and the percentage, as a line plot (not a spider).
#
#  APPENDIX (per configuration, one at a time via CONFIG_TO_RUN, plus the
#  per-configuration gap panel), all from the SAME set of runs:
#     1. LCOE reduction from interconnection    -> spider
#     2. battery reduction from interconnection -> spider
#     3. absolute interconnected LCOE, spider   -> what the electricity costs
#     4. absolute interconnected LCOE, tornado  -> which assumption sets the level
#     5. absolute LCOE gap, one panel per configuration
#
#  For each cost parameter, all of the above are recomputed at a few values
#  (low bound ... base ... high bound) while every other parameter is held at its
#  base. Figures 1 to 3 plot each line against "value relative to base", so the
#  SLOPE of a line is the intrinsic sensitivity and the LENGTH of a line is the real
#  appendix uncertainty range. The base sits at 1.0, where all lines cross.
#
#  Figures 2, 3 and 4 cost no extra model runs: every sample point already runs
#  both scenarios for the LCOE figure, so the battery sizes and the absolute LCOEs
#  are simply recorded from the same results, and the tornado reads the endpoints of
#  the lines that were computed anyway. Note the battery reduction can be flat (sizing
#  pinned by the need to serve all load, so cost cannot move it) or even negative (pooled
#  battery larger than the isolated fleet), and both are real outcomes, not bugs.
#
#  Figure 4 is built on the INTERCONNECTED system only, since that is the topology the
#  thesis recommends; the isolated base LCOE is drawn on it as a reference line. The
#  battery is re-optimised at every parameter value, so each bar spans the LCOE of
#  differently-sized optimal systems, not of one system re-costed.
#
#  This is a self-contained, press-run script and it runs ONE configuration per
#  run. Pick it with the toggle below, run it, then switch and run again. It is
#  meant for the SQ1 window: set USE_2025_2026_WINDOW = False in configuration.py.
#
#  The base is anchored at an 8% discount rate here (BASE_OVERRIDES), so the
#  spider is centred on 8% whatever value configuration.py currently holds.
# ============================================================================

import os
import math
import matplotlib.pyplot as plt

from configuration import *
from pv_profile import build_pv_for_window
from wind_profile import build_wind_for_window
from operation import run_isolated, run_interconnected   # the two operating modes: isolated fleet and interconnected cluster
from load_profile import prepare_model_loads


# ============================================================================
#  TOGGLE: choose ONE configuration per run
# ============================================================================
CONFIG_TO_RUN = CONFIG_PV_ONLY          # CONFIG_PV_ONLY, CONFIG_PV_WIND or CONFIG_PV_DIESEL

# Draw the second spider (battery reduction) as well as the LCOE one. It reuses
# the same runs, so switching it off saves no computation, only a figure.
PLOT_BATTERY_REDUCTION = True

# Absolute-LCOE figures. Both are built from the SAME runs as the spiders above,
# so neither costs an extra simulation.
#   PLOT_ABSOLUTE_LCOE : spider of the interconnected LCOE itself [EUR/kWh],
#                        which shows curvature that a tornado cannot show.
#   PLOT_TORNADO       : tornado of the interconnected LCOE, parameters ranked by
#                        how far they move it between their low and high bounds.
PLOT_ABSOLUTE_LCOE = True
PLOT_TORNADO       = True

# Extra bar on the tornado with EVERY parameter pushed to its favourable bound at
# once, and then to its unfavourable bound at once. This is a perfect-correlation
# extreme envelope, NOT a probability range, and it is labelled as such on the
# figure. It is the only thing on this page that costs extra runs (two of them).
PLOT_COMBINED_ENVELOPE = True

# Optionally mark, on each tornado bar, what the ISOLATED fleet would cost at the SAME
# parameter value as the end of that arm. This is the only honest way to put isolation on
# a tornado: a single vertical line at the base isolated LCOE would be compared against
# arms computed under different assumptions, which is meaningless, because the isolated
# fleet also gets dearer when a parameter is pushed to its high bound (and by more, since
# it carries the larger battery). Switching this on shows that interconnection stays
# cheaper across the whole range, at the price of a much wider x axis, which flattens the
# bars and makes the ranking harder to read. Off by default.
TORNADO_SHOW_ISOLATED = False

# Combined figure across ALL THREE configurations: a 1x3 panel of the ABSOLUTE
# LCOE gap (isolated minus interconnected, EUR/kWh) against each parameter. This is
# the headline sensitivity figure for the end of SQ1, because the absolute gap puts
# the three configs on one comparable scale, unlike the percentage reduction whose
# denominator differs by config. It runs its own loop over the three configurations,
# so it is independent of CONFIG_TO_RUN and produces one figure whichever config is
# selected above. Set it True on a single run to build the panel.
PLOT_GAP_PANEL = True

# Headline sensitivity figures: the benefit as a deviation tornado, three configs
# on one axis, in both the percentage and the gap metric (see the header). Runs its
# own loop over the three configurations, so it is independent of CONFIG_TO_RUN.
PLOT_BENEFIT_RANKING = True

# Deep-dive figure: the benefit against battery capex, the one surviving driver,
# drawn in both metrics. Runs from the same across-config sweep as the tornados.
PLOT_GAP_VS_CAPEX = True

# Number of sample points along each parameter line. Any parameter not listed
# here uses DEFAULT_POINTS. Give more points to a line that is curved (for
# example Battery CapEx in the diesel config, which flattens as storage is
# abandoned in favour of diesel) so the bend is resolved; straight lines gain
# nothing from extra points. Use odd numbers so the base stays centred.
DEFAULT_POINTS  = 5
POINTS_BY_PARAM = {
    "BESS CapEx": 9,     # set to 9 or 11 when running the diesel config
}

# Optional diagnostic (diesel config only): after the spiders, print the chosen
# isolated and interconnected battery sizes across a range of Battery CapEx
# values, next to the sweep floor, to check whether the flat tail of the Battery
# CapEx line is genuine economics or the battery hitting Battery_min_kWh.
RUN_BATTERY_FLOOR_CHECK = False
FLOOR_CHECK_CAPEX       = [300, 400, 500, 600, 700, 800]     # EUR/kWh, spanning the appendix low..high


# ============================================================================
#  Parameter ranges, taken from the techno-economic appendix table.
#  label -> (override key, base, low bound, high bound). The override key is the
#  exact configuration.py name that calculate_lcoe now recognises.
#  Wind OpEx is stored in EUR/kW/year, so the 3/5/10 percent of CapEx bounds are
#  written out as fractions of the wind CapEx base.
# ============================================================================
PARAMS = {
    "PV CapEx":      ("PV_capex_eur_kW",        720.0,        484.0,         1212.0),
    "BESS CapEx":    ("Battery_capex_eur_kWh",  400.0,        300.0,         800.0),
    "Discount rate": ("Discount_rate",          0.08,         0.05,          0.10),
    "Wind CapEx":    ("Wind_capex_eur_kW",      3765.0,       2680.0,        9880.0),
    "Wind OpEx":     ("Wind_opex_eur_kW_yr",    0.05 * 3765,  0.03 * 3765,   0.10 * 3765),
    "Diesel CapEx":  ("Diesel_capex_eur_kW",    500.0,        360.0,         800.0),
    "Diesel fuel":   ("Diesel_price_eur_L",     1.81,         1.5,           2.3),
    "Cable CapEx":   ("Cable_capex_eur_m",      30.0,         20.0,          60.0),
}

# Which parameters appear in each configuration's spider. The three shared cost
# drivers are included everywhere so the three charts are directly comparable.
# Keyed off the configuration names in configuration.py, so this stays correct if
# a name is ever changed there (the old hard-coded "PV only" strings raised a
# KeyError once the configs were renamed to PV-BESS and so on).
PARAMS_BY_CONFIG = {
    CONFIG_PV_ONLY["name"]:   ["PV CapEx", "BESS CapEx", "Discount rate"],
    CONFIG_PV_WIND["name"]:   ["Wind CapEx", "Wind OpEx", "PV CapEx", "BESS CapEx", "Discount rate"],
    CONFIG_PV_DIESEL["name"]: ["Diesel fuel", "Diesel CapEx", "PV CapEx", "BESS CapEx", "Discount rate"],
}

# Colour per configuration for the sensitivity figures. The user picked, for these
# figures only, the cost-term colour of each configuration's defining generator:
# PV-BESS in the PV-capex goldenrod, PV-Wind-BESS in the wind-capex green, and
# PV-Diesel-BESS in the diesel-capex grey. These cost terms never appear as stacks
# inside the tornado or line figures, so there is no colour clash, and none is the
# reserved interconnected purple.
CONFIG_COLOUR = {
    CONFIG_PV_ONLY["name"]:   CONFIG_PV_ONLY["colour"],
    CONFIG_PV_WIND["name"]:   CONFIG_PV_WIND["colour"],
    CONFIG_PV_DIESEL["name"]: CONFIG_PV_DIESEL["colour"],
}

# The cable is a cost only the interconnected fleet carries, and only when
# CABLING_COSTS_ON is True in configuration.py. With cabling off the cable length
# is set to zero, so the cable capex has no effect on the LCOE at all and a line
# for it would be a flat, meaningless zero. It is therefore appended to the
# parameter list only when cabling costs are switched on.
# Note the base value sits ON the low bound (20 EUR/m, sensitivity range 20 to 85),
# so this uncertainty is one-sided: the cable can only make interconnection dearer.
INCLUDE_CABLE = CABLING_COSTS_ON

# Every parameter held at its base value. Passing this on every run anchors the
# whole spider on the base case (discount rate 8%), independent of configuration.py.
BASE_OVERRIDES = {spec[0]: spec[1] for spec in PARAMS.values()}


def sample_points(base, low, high, n):
    # n sample values spanning low..high, always including low, base and high.
    # Half the points sit between low and base, half between base and high, and
    # the base itself is kept exactly (not recomputed) so every line crosses at
    # a relative value of exactly 1.0 and the shared base point can be reused.
    #
    # The two "if" guards handle a base that sits exactly ON one of its bounds
    # (Cable CapEx has base = low = 20 EUR/m). Without them that side of the line
    # would be filled with copies of the base value.
    k = (n + 1) // 2                                                   # points on each side, base shared

    if low == base:
        left = [base]                                                  # nothing below the base to sample
    else:
        left = [low + (base - low) * i / (k - 1) for i in range(k - 1)] + [base]   # low .. base, ending on the exact base

    if high == base:
        right = []                                                     # nothing above the base to sample
    else:
        right = [base + (high - base) * i / (k - 1) for i in range(1, k)]          # first step past base .. high

    return left + right                                               # low-side points, base, then high-side points


def run_point(config, gen, loads, overrides):
    # One isolated-vs-interconnected comparison. Returns everything the figures
    # need from that single pair of runs, so no result is ever recomputed:
    #   the two reductions [%] for the spiders, and the two absolute LCOEs
    #   [EUR/kWh] for the absolute spider and the tornado.
    # The values are handed back in a dictionary so the caller reads them by name
    # (result["lcoe_inter"]) instead of remembering the order of four returns.
    _, inter, _ = run_interconnected(gen, loads, config=config, overrides=overrides)
    iso, _      = run_isolated(gen, loads, config=config, overrides=overrides)

    lcoe_inter = inter["lcoe_eur_kWh"]
    lcoe_iso   = iso["lcoe_eur_kWh"]
    lcoe_red   = (1 - lcoe_inter / lcoe_iso) * 100

    iso_batt = iso["battery_kWh_total"]
    if iso_batt > 0:
        # can be negative if the pooled battery ends up larger than the isolated fleet
        batt_red = (1 - inter["battery_kWh_total"] / iso_batt) * 100
    else:
        batt_red = float("nan")      # no isolated battery at all, so a reduction is undefined

    return {"lcoe_red": lcoe_red, "batt_red": batt_red,
            "lcoe_inter": lcoe_inter, "lcoe_iso": lcoe_iso,
            "lcoe_gap": lcoe_iso - lcoe_inter}      # absolute LCOE reduction [EUR/kWh]: the headline benefit metric


def plot_spider(lines, param_labels, ylabel, metric_name, base_value, base_note,
                config_name, n_households, window_label, filename_stem):
    # One spider figure. "lines" maps a parameter label to its (x, y) lists, so the
    # same function draws the LCOE figure and the battery figure with no duplication.
    fig, ax = plt.subplots(figsize=(9, 6))
    for label in param_labels:
        xs, ys = lines[label]
        ax.plot(xs, ys, marker="o", label=label)

    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")     # the base case sits at a relative value of 1.0

    # Only draw a zero line if some value is actually negative (a pooled battery larger
    # than the isolated fleet). Drawing it unconditionally would force the axis to
    # include zero and un-zoom the figure, which is not wanted for the LCOE spider.
    # The list comprehension collects every y value, skipping NaN (a value is NaN when
    # it is not equal to itself, so this keeps only real numbers).
    all_y = [y for label in param_labels for y in lines[label][1] if y == y]
    if all_y and min(all_y) < 0:
        ax.axhline(0.0, color="grey", linewidth=0.8)

    ax.set_xlabel("Parameter value relative to base")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric_name} sensitivity: {config_name} (N = {n_households})\n{window_label}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.annotate(base_note,
                xy=(0.98, 0.03), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white", edgecolor="grey", alpha=0.8))

    plt.tight_layout()
    safe = config_name.replace(" ", "").replace("+", "_")             # tidy the config name for the filename
    fname = f"outputs/{filename_stem}_{safe}_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


def plot_tornado(rows, base_lcoe, envelope, show_isolated,
                 config_name, n_households, window_label):
    # Tornado of the ABSOLUTE interconnected LCOE.
    #
    # rows     : list of (label, inter at low bound, inter at high bound, swing,
    #                     iso at low bound, iso at high bound), one per parameter
    # envelope : None, or (inter favourable, inter unfavourable, iso favourable, iso unfavourable)
    #
    # Each parameter gets two horizontal bars that both start at the base LCOE: one
    # running out to the LCOE reached at that parameter's low bound, the other to the
    # LCOE at its high bound. Matplotlib is happy with a negative bar width, which is
    # what draws a bar to the LEFT of the base line, so no special handling is needed
    # for parameters that get cheaper at their low bound.
    #
    # Every point on this chart is a RE-OPTIMISED system: the battery is resized at
    # each parameter value, so the two arms of a bar are not symmetric about the base.
    #
    # There is deliberately NO vertical line at the base isolated LCOE. It would invite
    # the reader to compare an arm (parameter at its bound) against the isolated fleet
    # (parameter at its base), which is not a like-for-like comparison. When
    # show_isolated is on, the isolated cost is instead marked arm by arm, at the same
    # parameter value as the arm it belongs to.

    rows = sorted(rows, key=lambda r: r[3])                # ascending swing; barh draws the first row at the bottom
    y_positions = list(range(len(rows)))
    labels = [r[0] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 1.0 + 0.75 * (len(rows) + (1.5 if envelope else 0))))

    def label_ends(values, y):                             # value label just outside each arm end
        for value in values:
            side = 4 if value >= base_lcoe else -4         # push the text away from the base line
            ax.annotate(f"{value:.3f}", xy=(value, y), xytext=(side, 0),
                        textcoords="offset points", va="center",
                        ha="left" if value >= base_lcoe else "right", fontsize=8)

    for y, (label, inter_low, inter_high, swing, iso_low, iso_high) in zip(y_positions, rows):
        ax.barh(y, inter_low - base_lcoe, left=base_lcoe, height=0.6,
                color="steelblue", edgecolor="black", linewidth=0.5)
        ax.barh(y, inter_high - base_lcoe, left=base_lcoe, height=0.6,
                color="firebrick", edgecolor="black", linewidth=0.5)
        label_ends((inter_low, inter_high), y)

        if show_isolated:                                  # isolated fleet at the SAME parameter value as each arm
            ax.plot([iso_low, iso_high], [y, y], linestyle="none",
                    marker="D", markersize=6, markerfacecolor="none",
                    markeredgecolor="dimgrey", markeredgewidth=1.2, zorder=5)

    if envelope is not None:
        inter_fav, inter_unfav, iso_fav, iso_unfav = envelope
        y_env = len(rows) + 0.8                            # sits above the parameter bars, visually separated
        ax.barh(y_env, inter_fav - base_lcoe, left=base_lcoe, height=0.6,
                color="lightgrey", edgecolor="black", linewidth=0.5, hatch="//")
        ax.barh(y_env, inter_unfav - base_lcoe, left=base_lcoe, height=0.6,
                color="lightgrey", edgecolor="black", linewidth=0.5, hatch="//")
        label_ends((inter_fav, inter_unfav), y_env)

        if show_isolated:
            ax.plot([iso_fav, iso_unfav], [y_env, y_env], linestyle="none",
                    marker="D", markersize=6, markerfacecolor="none",
                    markeredgecolor="dimgrey", markeredgewidth=1.2, zorder=5)

        y_positions = y_positions + [y_env]
        labels = labels + ["All parameters\ncombined"]

    ax.axvline(base_lcoe, color="black", linewidth=1.2)                                  # the base case

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Interconnected cluster LCOE (EUR/kWh)")
    ax.set_title(f"LCOE sensitivity: {config_name} (N = {n_households})\n{window_label}")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)                                                               # gridlines behind the bars

    # Legend built by hand, because the bars themselves are drawn in a loop and would
    # otherwise add one legend entry per parameter. Patch() makes a plain coloured box.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="steelblue", edgecolor="black", label="Parameter at low bound"),
               Patch(facecolor="firebrick", edgecolor="black", label="Parameter at high bound"),
               plt.Line2D([0], [0], color="black", linewidth=1.2,
                          label=f"Base interconnected ({base_lcoe:.3f})")]
    if show_isolated:
        handles.append(plt.Line2D([0], [0], linestyle="none", marker="D", markersize=6,
                                  markerfacecolor="none", markeredgecolor="dimgrey",
                                  label="Isolated fleet at the same parameter value"))
    if envelope is not None:
        handles.append(Patch(facecolor="lightgrey", edgecolor="black", hatch="//",
                             label="All parameters at once (extremes)"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    # Give the value labels room on both sides of the widest bar
    ax.margins(x=0.12)

    plt.tight_layout()
    safe = config_name.replace(" ", "").replace("+", "_")
    fname = f"outputs/sensitivity_tornado_{safe}_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


def gap_lines_for_config(config, pv_cluster, wind_cluster, loads):
    # Run the parameter sweep for ONE configuration and return its absolute
    # LCOE-gap lines, ready to fill one panel of the combined figure. Reuses
    # run_point, so it costs the same runs the per-config figures would.
    # Returns (name, labels, gap_lines, red_lines, base_gap, base_red), where
    # gap_lines maps each parameter to (x relative to base, y absolute gap
    # [EUR/kWh]) and red_lines to (x relative to base, y LCOE reduction [%]).
    name = config["name"]
    gen  = pv_cluster + wind_cluster if config["use_wind"] else pv_cluster   # add wind only where the config uses it
    labels = list(PARAMS_BY_CONFIG[name])                                    # list(...) copies, so appending is safe
    if INCLUDE_CABLE:
        labels = labels + ["Cable CapEx"]

    base = run_point(config, gen, loads, dict(BASE_OVERRIDES))               # shared base point (relative value 1.0)
    base_gap = base["lcoe_gap"]
    base_red = base["lcoe_red"]

    gap_lines = {}                                                          # label -> (x relative to base, y absolute gap [EUR/kWh])
    red_lines = {}                                                          # label -> (x relative to base, y LCOE reduction [%])
    for label in labels:
        key, base_value, low, high = PARAMS[label]
        n_points = POINTS_BY_PARAM.get(label, DEFAULT_POINTS)
        xs, ys_gap, ys_red = [], [], []
        for value in sample_points(base_value, low, high, n_points):
            if value == base_value:
                point = base                                                # reuse the base, no rerun
            else:
                ov = dict(BASE_OVERRIDES)
                ov[key] = value
                point = run_point(config, gen, loads, ov)
            xs.append(value / base_value)                                   # position on the shared "relative to base" axis
            ys_gap.append(point["lcoe_gap"])
            ys_red.append(point["lcoe_red"])
        gap_lines[label] = (xs, ys_gap)
        red_lines[label] = (xs, ys_red)

    return name, labels, gap_lines, red_lines, base_gap, base_red


def plot_gap_panels(results, n_households, window_label, flat_tol=1e-6):
    # One 1x3 figure: absolute LCOE gap versus parameter, one panel per config.
    # results is a list of (name, labels, gap_lines, red_lines, base_gap, base_red);
    # this figure uses the gap lines only. Only parameters
    # that ACTUALLY move the gap are drawn; anything whose gap is flat within
    # flat_tol is dropped, because a topology-invariant cost (PV/wind/diesel capex,
    # wind opex) is identical on both sides and cancels exactly, so it can never
    # move the gap. Those are named under each panel title so the reader sees they
    # were tested and cancel by construction, which is the whole point of the metric.
    # The panels share a y-axis, so a similar base height across panels is itself
    # the finding: the absolute benefit is alike where the percentage view hid it.
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]                                    # wrap a lone Axes so the loop still works

    for ax, (name, labels, gap_lines, red_lines, base_gap, base_red) in zip(axes, results):
        movers, flat = [], []
        for label in labels:
            ys = gap_lines[label][1]
            if max(ys) - min(ys) > flat_tol:             # keep only lines that genuinely move the gap
                movers.append(label)
            else:
                flat.append(label)                       # flat by construction (cancels between topologies)

        for label in movers:
            xs, ys = gap_lines[label]
            ax.plot(xs, ys, marker="o", label=label)
        ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")       # base sits at relative value 1.0

        # Only draw the zero line if a gap actually goes negative (interconnection
        # dearer than isolation, possible once cabling is on). Otherwise it would
        # force the axis to include zero and squash the panels.
        panel_min = min(y for label in movers for y in gap_lines[label][1])
        if panel_min < 0:
            ax.axhline(0.0, color="grey", linewidth=0.8)

        title = f"{name}\nbase gap = {base_gap:.3f} EUR/kWh"
        if flat:                                         # name the cancelling parameters under the title
            title += f"\nflat (cancels): {', '.join(flat)}"
        ax.set_xlabel("Parameter value relative to base")
        ax.set_title(title, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Absolute LCOE gap, isolated minus interconnected (EUR/kWh)")

    plt.tight_layout()
    fname = f"outputs/sensitivity_gap_panel_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


def plot_benefit_vs_battery_capex(results, metric, n_households, window_label):
    # One panel, three configs overlaid: the interconnection benefit against battery
    # capex, the one strong driver that survives the flat-line cull in every
    # configuration, so this figure carries the whole benefit story in one plot. In
    # the gap metric all three gaps rise with battery price, the ranking never
    # crosses, and none dips to zero, which is the "robust" part of the story. The
    # percentage metric is offered as a cross-check; expect the diesel line to look
    # jumpy, because its base benefit is tiny so a small absolute change reads as a
    # large percentage. The x-axis is the actual battery price (EUR/kWh), got by
    # scaling the relative sweep by its base. This is a line plot of the benefit
    # against one parameter, not a spider (which carries one line per parameter).
    #
    # metric == "gap"     -> absolute LCOE gap, EUR/kWh
    # metric == "percent" -> LCOE reduction from interconnection, %
    line_index = 2 if metric == "gap" else 3            # gap_lines or red_lines
    if metric == "gap":
        ylabel = "Absolute LCOE gap, isolated minus interconnected (EUR/kWh)"
        stem   = "sensitivity_benefit_vs_batterycapex_gap"
    else:
        ylabel = "LCOE reduction from interconnection (%)"
        stem   = "sensitivity_benefit_vs_batterycapex_percent"

    base_capex = PARAMS["BESS CapEx"][1]             # base battery capex [EUR/kWh]

    fig, ax = plt.subplots(figsize=(8, 6))
    for entry in results:
        name  = entry[0]
        lines = entry[line_index]
        if "BESS CapEx" not in lines:
            continue                                    # skip a config that somehow lacks the sweep
        xs_rel, ys = lines["BESS CapEx"]
        xs_abs = [x * base_capex for x in xs_rel]       # relative-to-base -> absolute EUR/kWh
        ax.plot(xs_abs, ys, marker="o", color=CONFIG_COLOUR.get(name, "grey"), label=name)

    ax.axvline(base_capex, color="black", linewidth=0.8, linestyle="--")    # base battery capex

    all_y = [y for entry in results for y in entry[line_index].get("BESS CapEx", ([], []))[1]]
    if all_y and min(all_y) < 0:                        # zero line only if the benefit turns negative
        ax.axhline(0.0, color="grey", linewidth=0.8)

    ax.set_xlabel("BESS capex (EUR/kWh)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    fname = f"outputs/{stem}_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


def benefit_deviations(results, metric):
    # For one metric, return per-configuration arm ends and base benefit.
    #   devs[name][param] = (dev_low, dev_high)
    #     dev_low  = benefit at the parameter's low bound  minus its base
    #     dev_high = benefit at the parameter's high bound minus its base
    #   bases[name] = the configuration's base benefit
    # sample_points puts the low bound first and the high bound last in each line,
    # so ys[0] and ys[-1] are the two arm ends. order_seen keeps the parameters in
    # first-seen order, before any sorting.
    line_index = 2 if metric == "gap" else 3          # gap_lines or red_lines in each result tuple
    base_index = 4 if metric == "gap" else 5          # base_gap or base_red
    devs, bases, order_seen = {}, {}, []
    for entry in results:
        name = entry[0]
        base = entry[base_index]
        bases[name] = base
        d = {}
        for label, (xs, ys) in entry[line_index].items():
            d[label] = (ys[0] - base, ys[-1] - base)
            if label not in order_seen:
                order_seen.append(label)
        devs[name] = d
    return devs, bases, order_seen


def benefit_param_order(results, metric="percent"):
    # Shared parameter order for both tornados, so a reader can track a parameter
    # from one figure to the other and watch its bar vanish. Sorted by the largest
    # swing any configuration shows in the chosen metric, ascending, because barh
    # draws the first row at the bottom and we want the biggest swing on top.
    devs, _, order_seen = benefit_deviations(results, metric)

    def max_swing(label):
        vals = [abs(dhi - dlo) for n in devs if label in devs[n]
                for (dlo, dhi) in [devs[n][label]]]
        return max(vals, default=0.0)

    return sorted(order_seen, key=max_swing)


def plot_benefit_tornado(results, metric, param_order, n_households, window_label):
    # Deviation tornado of the interconnection benefit, three configurations on one
    # axis. Zero in the middle is each configuration's base case; a bar runs from
    # its low-bound deviation to its high-bound deviation, so the arm to the left of
    # zero is where the benefit falls and the arm to the right is where it rises.
    # Because colour carries the configuration, the side (not the colour) tells low
    # from high. A parameter that cancels between the two topologies (any cost that
    # is identical when isolated and interconnected) sits at zero with no visible
    # bar, which is exactly the point of the gap version: PV, wind and diesel capex
    # collapse there, leaving only battery capex, discount rate and diesel fuel.
    #
    # metric == "percent" -> deviation in LCOE reduction, percentage points
    # metric == "gap"     -> deviation in the absolute LCOE gap, EUR/kWh
    if metric == "gap":
        xlabel = "Change in LCOE gap from base, isolated minus interconnected (EUR/kWh)"
        value_fmt = lambda v: f"{v:+.3f}"
        base_fmt  = lambda v: f"{v:.3f}"
        stem = "sensitivity_benefit_tornado_gap"
    else:
        xlabel = "Change in LCOE reduction from base (percentage points)"
        value_fmt = lambda v: f"{v:+.1f}"
        base_fmt  = lambda v: f"{v:.1f}%"
        stem = "sensitivity_benefit_tornado_percent"

    devs, bases, _ = benefit_deviations(results, metric)
    names   = [entry[0] for entry in results]
    n_cfg   = len(names)
    group_h = 0.8                                             # total height taken by one parameter's group of bars
    bar_h   = group_h / n_cfg

    # A label floor, so the cancelling arms near zero are left unlabelled and only
    # the real movers get their numbers.
    fig_max = max((abs(v) for name in names for p in devs[name]
                   for v in devs[name][p]), default=1.0)
    label_floor = 0.02 * fig_max

    fig, ax = plt.subplots(figsize=(10, 1.4 + 0.75 * len(param_order)))
    for row, label in enumerate(param_order):
        for j, name in enumerate(names):
            if label not in devs[name]:
                continue                                     # this configuration does not use the parameter
            dev_low, dev_high = devs[name][label]
            left  = min(dev_low, dev_high)
            right = max(dev_low, dev_high)
            y = row + ((n_cfg - 1) / 2 - j) * bar_h          # centre the group on the row, first config on top
            ax.barh(y, right - left, left=left, height=bar_h * 0.9,
                    color=CONFIG_COLOUR.get(name, "grey"), edgecolor="black", linewidth=0.5)
            if abs(left) > label_floor:                      # label the left tip, pushed further left
                ax.annotate(value_fmt(left), xy=(left, y), xytext=(-3, 0),
                            textcoords="offset points", va="center", ha="right", fontsize=7)
            if abs(right) > label_floor:                     # label the right tip, pushed further right
                ax.annotate(value_fmt(right), xy=(right, y), xytext=(3, 0),
                            textcoords="offset points", va="center", ha="left", fontsize=7)

    ax.axvline(0.0, color="black", linewidth=1.2)            # the base case, zero deviation
    ax.set_yticks(range(len(param_order)))
    ax.set_yticklabels(param_order)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)                                   # gridlines behind the bars

    # Legend: one entry per configuration, with its base benefit for context.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CONFIG_COLOUR.get(name, "grey"), edgecolor="black",
                     label=f"{name} (base {base_fmt(bases[name])})") for name in names]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    # Centre the axis on zero (equal width each side) and zoom in to the data,
    # leaving room for the tip labels. pad is the fraction of the longest bar kept
    # as blank space on each side: lower it to zoom in more, raise it if a label clips.
    pad = 0.22
    ax.set_xlim(-fig_max * (1 + pad), fig_max * (1 + pad))

    print(f"\n  benefit tornado, deviation low..high by parameter ({'EUR/kWh gap' if metric == 'gap' else 'percentage points'}):")
    for label in param_order[::-1]:                          # most influential first
        cells = "   ".join(f"{n} [{value_fmt(devs[n][label][0])}, {value_fmt(devs[n][label][1])}]"
                           for n in names if label in devs[n])
        print(f"    {label:<14} {cells}")

    plt.tight_layout()
    fname = f"outputs/{stem}_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Run
# ============================================================================
if __name__ == "__main__":

    os.makedirs("outputs", exist_ok=True)

    base_loads, time_index = prepare_model_loads()
    n_households = base_loads.shape[1]
    loads = base_loads

    pv_cluster   = build_pv_for_window(time_index, n_households, SOLAR_FILE, loads)
    wind_cluster = build_wind_for_window(time_index, n_households, SOLAR_FILE, loads)
    gen = pv_cluster + wind_cluster if CONFIG_TO_RUN["use_wind"] else pv_cluster   # add wind only where the config uses it

    config_name  = CONFIG_TO_RUN["name"]
    param_labels = list(PARAMS_BY_CONFIG[config_name])                 # list(...) copies, so appending does not edit PARAMS_BY_CONFIG
    if INCLUDE_CABLE:
        param_labels = param_labels + ["Cable CapEx"]                  # only meaningful when CABLING_COSTS_ON
    window_label = f"{MODEL_WINDOW_START} to {MODEL_WINDOW_END}"

    print("\n" + "=" * 78)
    print(f"  SENSITIVITY SPIDERS: {config_name}  ({window_label})")
    if not CABLING_COSTS_ON:
        print("  cabling costs OFF: cable capex left out of the sensitivity (it cannot move the LCOE)")
    print("=" * 78)

    # base case computed once; every line passes through it at relative value 1.0
    base = run_point(CONFIG_TO_RUN, gen, loads, dict(BASE_OVERRIDES))
    base_lcoe_red  = base["lcoe_red"]
    base_batt_red  = base["batt_red"]
    base_lcoe_inter = base["lcoe_inter"]
    base_lcoe_iso   = base["lcoe_iso"]
    print(f"  base LCOE reduction    = {base_lcoe_red:.2f}%")
    print(f"  base battery reduction = {base_batt_red:.2f}%")
    print(f"  base LCOE              = {base_lcoe_iso:.4f} (isolated) -> {base_lcoe_inter:.4f} EUR/kWh (interconnected)\n")

    lines_lcoe = {}                                  # label -> (x relative to base, y LCOE reduction)
    lines_batt = {}                                  # label -> (x relative to base, y battery reduction)
    lines_abs  = {}                                  # label -> (x relative to base, y interconnected LCOE [EUR/kWh])
    summary    = []                                  # (label, LCOE swing, battery swing)
    tornado_rows = []                                # (label, LCOE at low bound, LCOE at high bound, swing) for the tornado

    for label in param_labels:
        key, base_value, low, high = PARAMS[label]
        n_points = POINTS_BY_PARAM.get(label, DEFAULT_POINTS)   # how many points for this line; defaults to 5

        xs, ys_lcoe, ys_batt, ys_abs, ys_abs_iso = [], [], [], [], []
        for value in sample_points(base_value, low, high, n_points):
            if value == base_value:                  # reuse the shared base point, no need to rerun
                point = base
            else:
                ov = dict(BASE_OVERRIDES)            # dict(...) makes a fresh copy so the base is not altered
                ov[key] = value
                point = run_point(CONFIG_TO_RUN, gen, loads, ov)
            xs.append(value / base_value)            # position on the shared "relative to base" axis
            ys_lcoe.append(point["lcoe_red"])
            ys_batt.append(point["batt_red"])
            ys_abs.append(point["lcoe_inter"])
            ys_abs_iso.append(point["lcoe_iso"])     # the isolated fleet costed under the SAME parameter value

        lines_lcoe[label] = (xs, ys_lcoe)
        lines_batt[label] = (xs, ys_batt)
        lines_abs[label]  = (xs, ys_abs)

        # swing = how far the metric moves from the low bound to the high bound
        swing_lcoe = abs(ys_lcoe[-1] - ys_lcoe[0])
        swing_batt = abs(ys_batt[-1] - ys_batt[0])
        summary.append((label, swing_lcoe, swing_batt))

        # first and last sample point ARE the low and high bounds, so the tornado is free.
        # The isolated values are carried along so the tornado can mark them at the same
        # parameter value as the arm they belong to, rather than at the base.
        tornado_rows.append((label, ys_abs[0], ys_abs[-1], abs(ys_abs[-1] - ys_abs[0]),
                             ys_abs_iso[0], ys_abs_iso[-1]))

        print(f"    {label:<14} LCOE {ys_lcoe[0]:6.2f}% -> {ys_lcoe[-1]:6.2f}%   "
              f"battery {ys_batt[0]:6.2f}% -> {ys_batt[-1]:6.2f}%   "
              f"abs LCOE {ys_abs[0]:.4f} -> {ys_abs[-1]:.4f} EUR/kWh")

    # tables ordered by influence, the tornado-style summary without a second chart.
    # math.isnan(x) is True when x is "not a number", which happens where a reduction
    # is undefined; those are sorted last rather than breaking the ordering.
    def sort_key(swing):
        return -1.0 if math.isnan(swing) else swing

    print("\n  parameters ordered by swing in LCOE reduction (percentage points):")
    for label, swing_lcoe, _ in sorted(summary, key=lambda r: sort_key(r[1]), reverse=True):
        print(f"    {label:<14} {swing_lcoe:5.2f} pp")

    print("\n  parameters ordered by swing in BESS capacity reduction (percentage points):")
    for label, _, swing_batt in sorted(summary, key=lambda r: sort_key(r[2]), reverse=True):
        print(f"    {label:<14} {swing_batt:5.2f} pp")

    print("\n  parameters ordered by swing in absolute interconnected LCOE (EUR/kWh):")
    for label, inter_low, inter_high, swing, iso_low, iso_high in sorted(tornado_rows, key=lambda r: r[3], reverse=True):
        print(f"    {label:<14} {swing:.4f}   ({inter_low:.4f} .. {inter_high:.4f})   "
              f"isolated at the same bounds ({iso_low:.4f} .. {iso_high:.4f})")
    print("=" * 78)

    # --- figure 1: LCOE reduction ---
    plot_spider(lines_lcoe, param_labels,
                ylabel="LCOE reduction (%)", metric_name="LCOE reduction",
                base_value=base_lcoe_red, base_note=f"base reduction = {base_lcoe_red:.1f}%",
                config_name=config_name, n_households=n_households, window_label=window_label,
                filename_stem="sensitivity_spider")

    # --- figure 2: battery reduction (same runs, no extra computation) ---
    if PLOT_BATTERY_REDUCTION:
        plot_spider(lines_batt, param_labels,
                    ylabel="BESS capacity reduction (%)", metric_name="BESS reduction",
                    base_value=base_batt_red, base_note=f"base reduction = {base_batt_red:.1f}%",
                    config_name=config_name, n_households=n_households, window_label=window_label,
                    filename_stem="sensitivity_spider_battery")

    # --- figure 3: absolute interconnected LCOE (same runs again) ---
    if PLOT_ABSOLUTE_LCOE:
        plot_spider(lines_abs, param_labels,
                    ylabel="Interconnected cluster LCOE (EUR/kWh)", metric_name="Absolute LCOE",
                    base_value=base_lcoe_inter, base_note=f"base LCOE = {base_lcoe_inter:.3f} EUR/kWh",
                    config_name=config_name, n_households=n_households, window_label=window_label,
                    filename_stem="sensitivity_spider_lcoe_abs")

    # --- figure 4: tornado of the absolute interconnected LCOE ---
    if PLOT_TORNADO:
        envelope = None
        if PLOT_COMBINED_ENVELOPE:
            # Two extra runs. Every parameter in THIS configuration is pushed to one bound
            # at a time: all low bounds together, then all high bounds together. Which of the
            # two comes out cheaper is not assumed, it is read off the results below.
            ov_low, ov_high = dict(BASE_OVERRIDES), dict(BASE_OVERRIDES)
            for label in param_labels:
                key, base_value, low, high = PARAMS[label]
                ov_low[key]  = low
                ov_high[key] = high

            point_low  = run_point(CONFIG_TO_RUN, gen, loads, ov_low)
            point_high = run_point(CONFIG_TO_RUN, gen, loads, ov_high)

            # Which corner is the favourable one is decided by the interconnected LCOE, and the
            # isolated value is then taken from the SAME corner, so the two always describe the
            # same set of assumptions.
            if point_low["lcoe_inter"] <= point_high["lcoe_inter"]:
                fav, unfav = point_low, point_high            # all low bounds is the cheap corner
            else:
                fav, unfav = point_high, point_low

            envelope = (fav["lcoe_inter"], unfav["lcoe_inter"],
                        fav["lcoe_iso"],   unfav["lcoe_iso"])

            print(f"\n  combined envelope (all parameters at once, perfect correlation, NOT a probability range):")
            print(f"    interconnected LCOE {fav['lcoe_inter']:.4f} .. {unfav['lcoe_inter']:.4f} EUR/kWh  "
                  f"(base {base_lcoe_inter:.4f})")
            print(f"    isolated LCOE       {fav['lcoe_iso']:.4f} .. {unfav['lcoe_iso']:.4f} EUR/kWh  "
                  f"(base {base_lcoe_iso:.4f})")
            print("=" * 78)

        plot_tornado(tornado_rows, base_lcoe_inter, envelope, TORNADO_SHOW_ISOLATED,
                     config_name, n_households, window_label)

    # --- optional battery-floor diagnostic (diesel config only) ---
    if RUN_BATTERY_FLOOR_CHECK and not CONFIG_TO_RUN["use_wind"] and CONFIG_TO_RUN["use_diesel"]:
        floor_fleet = Battery_min_kWh * n_households                 # smallest possible isolated fleet total
        print("\n" + "=" * 70)
        print(f"  BATTERY FLOOR CHECK: {config_name}")
        print(f"  Battery_min_kWh = {Battery_min_kWh} kWh per nanogrid (isolated fleet floor = {floor_fleet} kWh)")
        print("=" * 70)
        print(f"  {'capex':>6}{'iso total':>11}{'iso min lodge':>15}{'inter batt':>12}{'reduction':>11}{'  at floor?'}")

        for capex in FLOOR_CHECK_CAPEX:
            overrides = dict(BASE_OVERRIDES)                         # base case with one parameter changed
            overrides["Battery_capex_eur_kWh"] = capex

            _, inter, _ = run_interconnected(gen, loads, config=CONFIG_TO_RUN, overrides=overrides)
            iso, _      = run_isolated(gen, loads, config=CONFIG_TO_RUN, overrides=overrides)

            iso_min_lodge = iso["optimal_batteries"].min()           # .min() = smallest of the per-lodge battery sizes
            reduction     = (1 - inter["lcoe_eur_kWh"] / iso["lcoe_eur_kWh"]) * 100
            at_floor      = (inter["battery_kWh_total"] <= Battery_min_kWh + 1e-6) and (iso_min_lodge <= Battery_min_kWh + 1e-6)

            print(f"  {capex:>6}{iso['battery_kWh_total']:>11.2f}{iso_min_lodge:>15.2f}"
                  f"{inter['battery_kWh_total']:>12.2f}{reduction:>10.2f}%{'   yes' if at_floor else '   no'}")
        print("=" * 70)

    # --- across-configuration analysis: sweep the three configs once, reuse everywhere ---
    if PLOT_BENEFIT_RANKING or PLOT_GAP_PANEL or PLOT_GAP_VS_CAPEX:
        print("\n" + "=" * 78)
        print("  ACROSS-CONFIGURATION SENSITIVITY: sweeping the three configurations")
        print("=" * 78)
        gap_results = []
        for cfg in (CONFIG_PV_ONLY, CONFIG_PV_WIND, CONFIG_PV_DIESEL):
            gap_results.append(gap_lines_for_config(cfg, pv_cluster, wind_cluster, loads))
            name_, base_gap_, base_red_ = gap_results[-1][0], gap_results[-1][4], gap_results[-1][5]
            print(f"    {name_:<14} base gap = {base_gap_:.4f} EUR/kWh   base reduction = {base_red_:.2f}%")

        # headline: the benefit as a deviation tornado in both metrics. The
        # percentage figure sets up the apparent picture, the gap figure reveals
        # that only three parameters actually move the benefit. Both share one
        # parameter order (set by the percentage swing) so a parameter can be
        # tracked from the first figure to the second and watched collapse to zero.
        if PLOT_BENEFIT_RANKING:
            order = benefit_param_order(gap_results, "percent")
            plot_benefit_tornado(gap_results, "percent", order, n_households, window_label)
            plot_benefit_tornado(gap_results, "gap",     order, n_households, window_label)

        # deep dive on the one surviving driver, in both metrics
        if PLOT_GAP_VS_CAPEX:
            plot_benefit_vs_battery_capex(gap_results, "gap",     n_households, window_label)
            plot_benefit_vs_battery_capex(gap_results, "percent", n_households, window_label)

        # appendix line view: the per-configuration gap panel
        if PLOT_GAP_PANEL:
            plot_gap_panels(gap_results, n_households, window_label)