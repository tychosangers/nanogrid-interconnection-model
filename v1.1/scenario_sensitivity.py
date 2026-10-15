# ============================================================================
#  scenario_sensitivity.py  (SQ1)
#  One-at-a-time sensitivity of the interconnection benefit for the three
#  configurations. Each parameter is varied from its low to its high bound with
#  all others at their base value; the battery is sized again at every point.
#  Outputs: deviation tornados of the relative LCOE reduction [percentage
#  points] and of the absolute LCOE gap [EUR/kWh], with one shared parameter
#  order, and the absolute gap against BESS capex.
#  Run with USE_2025_2026_WINDOW = False. The base case uses the base values in
#  PARAMS (discount rate 8%), independent of configuration.py.
# ============================================================================

import os
import matplotlib.pyplot as plt

from configuration import *
from pv_profile import build_pv_for_window
from wind_profile import build_wind_for_window
from operation import run_isolated, run_interconnected
from load_profile import prepare_model_loads


# ============================================================================
#  Settings
# ============================================================================
# Sample points per parameter range (odd, so the base value is included)
DEFAULT_POINTS  = 5
POINTS_BY_PARAM = {
    "BESS CapEx": 9,
}


# ============================================================================
#  Parameter ranges: label: (override key, base, low bound, high bound).
#  Wind OpEx bounds are 3%, 5% and 10% of the wind capex per year.
# ============================================================================
PARAMS = {
    "PV CapEx":      ("PV_CAPEX_EUR_KW",        720.0,        484.0,         1212.0),
    "BESS CapEx":    ("BATTERY_CAPEX_EUR_KWH",  400.0,        300.0,         800.0),
    "Discount rate": ("DISCOUNT_RATE",          0.08,         0.05,          0.10),
    "Wind CapEx":    ("WIND_CAPEX_EUR_KW",      3765.0,       2680.0,        9880.0),
    "Wind OpEx":     ("WIND_OPEX_EUR_KW_YR",    0.05 * 3765,  0.03 * 3765,   0.10 * 3765),
    "Diesel CapEx":  ("DIESEL_CAPEX_EUR_KW",    500.0,        360.0,         800.0),
    "Diesel fuel":   ("DIESEL_PRICE_EUR_L",     1.81,         1.5,           2.3),
    "Cable CapEx":   ("CABLE_CAPEX_EUR_M",      30.0,         20.0,          60.0),
}

# Parameters varied per configuration
PARAMS_BY_CONFIG = {
    CONFIG_PV_BESS["name"]:   ["PV CapEx", "BESS CapEx", "Discount rate"],
    CONFIG_PV_WIND_BESS["name"]:   ["Wind CapEx", "Wind OpEx", "PV CapEx", "BESS CapEx", "Discount rate"],
    CONFIG_PV_DIESEL_BESS["name"]: ["Diesel fuel", "Diesel CapEx", "PV CapEx", "BESS CapEx", "Discount rate"],
}

# Configuration colours from configuration.py
CONFIG_COLOUR = {
    CONFIG_PV_BESS["name"]:   CONFIG_PV_BESS["colour"],
    CONFIG_PV_WIND_BESS["name"]:   CONFIG_PV_WIND_BESS["colour"],
    CONFIG_PV_DIESEL_BESS["name"]: CONFIG_PV_DIESEL_BESS["colour"],
}

# Cable capex is only varied when CABLING_COSTS_ON is True.
INCLUDE_CABLE = CABLING_COSTS_ON

# All parameters at their base value
BASE_OVERRIDES = {spec[0]: spec[1] for spec in PARAMS.values()}


def sample_points(base, low, high, n):
    """n values from low to high, including low, base and high exactly.

    Half of the points lie below and half above the base. A base equal to one
    of its bounds gets no points on that side.
    """
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
    """Isolated and interconnected run with the given overrides.

    Returns the LCOE and battery reduction [%], both LCOEs and the LCOE gap [EUR/kWh].
    """
    _, inter, _ = run_interconnected(gen, loads, config=config, overrides=overrides)
    iso, _      = run_isolated(gen, loads, config=config, overrides=overrides)

    lcoe_inter = inter["lcoe_eur_kWh"]
    lcoe_iso   = iso["lcoe_eur_kWh"]
    lcoe_red   = (1 - lcoe_inter / lcoe_iso) * 100

    iso_batt = iso["battery_kWh_total"]
    if iso_batt > 0:
        # Negative when the shared battery is larger than the isolated total
        batt_red = (1 - inter["battery_kWh_total"] / iso_batt) * 100
    else:
        batt_red = float("nan")

    return {"lcoe_red": lcoe_red, "batt_red": batt_red,
            "lcoe_inter": lcoe_inter, "lcoe_iso": lcoe_iso,
            "lcoe_gap": lcoe_iso - lcoe_inter}


def gap_lines_for_config(config, pv_cluster, wind_cluster, loads):
    """Parameter sweep of one configuration.

    Returns (name, labels, gap_lines, red_lines, base_gap, base_red). gap_lines
    and red_lines map each parameter to (value relative to base, LCOE gap
    [EUR/kWh]) and (value relative to base, LCOE reduction [%]).
    """
    name = config["name"]
    gen  = pv_cluster + wind_cluster if config["use_wind"] else pv_cluster   # add wind only where the config uses it
    labels = list(PARAMS_BY_CONFIG[name])                                    # list(...) copies, so appending is safe
    if INCLUDE_CABLE:
        labels = labels + ["Cable CapEx"]

    base = run_point(config, gen, loads, dict(BASE_OVERRIDES))               # shared base point (relative value 1.0)
    base_gap = base["lcoe_gap"]
    base_red = base["lcoe_red"]

    gap_lines = {}
    red_lines = {}
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


def plot_benefit_vs_battery_capex(results, metric, n_nanogrids, window_label):
    """Interconnection benefit against BESS capex for the three configurations.

    metric "gap": absolute LCOE gap [EUR/kWh]; "percent": LCOE reduction [%].
    """
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
        xs_abs = [x * base_capex for x in xs_rel]       # relative-to-base: absolute EUR/kWh
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
    """Deviation of the benefit from its base at the low and high bound of each parameter.

    Returns devs[name][param] = (dev_low, dev_high), the base benefit per
    configuration, and the parameters in first-seen order.
    """
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
    """Parameter order for the tornados: largest swing in any configuration on top."""
    devs, _, order_seen = benefit_deviations(results, metric)

    def max_swing(label):
        vals = [abs(dhi - dlo) for n in devs if label in devs[n]
                for (dlo, dhi) in [devs[n][label]]]
        return max(vals, default=0.0)

    return sorted(order_seen, key=max_swing)


def plot_benefit_tornado(results, metric, param_order, n_nanogrids, window_label):
    """Deviation tornado of the interconnection benefit for the three configurations.

    Each bar runs from the deviation at the low bound to that at the high bound
    of a parameter. metric "percent": LCOE reduction [percentage points];
    "gap": absolute LCOE gap [EUR/kWh].
    """
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
    group_h = 0.8
    bar_h   = group_h / n_cfg

    # Bars shorter than this are not labelled
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
            y = row + ((n_cfg - 1) / 2 - j) * bar_h
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

    # Legend with the base benefit per configuration
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CONFIG_COLOUR.get(name, "grey"), edgecolor="black",
                     label=f"{name} (base {base_fmt(bases[name])})") for name in names]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    # Symmetric x-axis around zero; pad is the margin as a fraction of the longest bar
    pad = 0.22
    ax.set_xlim(-fig_max * (1 + pad), fig_max * (1 + pad))

    print(f"\n  benefit tornado, deviation low..high by parameter ({'EUR/kWh gap' if metric == 'gap' else 'percentage points'}):")
    for label in param_order[::-1]:                          # most influential first
        cells = "   ".join(f"{n} [{value_fmt(devs[n][label][0])}, {value_fmt(devs[n][label][1])}]"
                           for n in names if label in devs[n])
        print(f"    {label:<14} {cells}")

    plt.tight_layout()
    fname = f"outputs/{stem}_{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}.png"
    save_figure(plt.gcf(), fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Run
# ============================================================================
if __name__ == "__main__":

    os.makedirs("outputs", exist_ok=True)

    base_loads, time_index = prepare_model_loads()
    n_nanogrids = base_loads.shape[1]
    loads = base_loads

    pv_cluster   = build_pv_for_window(time_index, n_nanogrids, SOLAR_FILE, loads)
    wind_cluster = build_wind_for_window(time_index, n_nanogrids, SOLAR_FILE, loads)
    window_label = f"{MODEL_WINDOW_START} to {MODEL_WINDOW_END}"

    print("\n" + "=" * 78)
    print(f"  SENSITIVITY OF THE INTERCONNECTION BENEFIT  ({window_label})")
    if not CABLING_COSTS_ON:
        print("  cabling costs off: cable capex is not varied")
    print("=" * 78)
    gap_results = []
    for cfg in (CONFIG_PV_BESS, CONFIG_PV_WIND_BESS, CONFIG_PV_DIESEL_BESS):
        gap_results.append(gap_lines_for_config(cfg, pv_cluster, wind_cluster, loads))
        name_, base_gap_, base_red_ = gap_results[-1][0], gap_results[-1][4], gap_results[-1][5]
        print(f"    {name_:<14} base gap = {base_gap_:.4f} EUR/kWh   base reduction = {base_red_:.2f}%")

    # Tornados in both metrics, with one shared parameter order
    order = benefit_param_order(gap_results, "percent")
    plot_benefit_tornado(gap_results, "percent", order, n_nanogrids, window_label)
    plot_benefit_tornado(gap_results, "gap",     order, n_nanogrids, window_label)

    plot_benefit_vs_battery_capex(gap_results, "gap", n_nanogrids, window_label)
