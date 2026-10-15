# ============================================================================
#  scenario_cabling.py  (SQ3)
#  Effect of cable cost and distance on the interconnection benefit, for the
#  three configurations. The cable is added in closed form to the results with
#  a free cable (CABLING_COSTS_ON = False): it adds a fixed annual cost and does
#  not change the battery sizing, so no dispatch is run again.
#  Results: the break-even cable length at a given price, and the break-even
#  cable price at the routed length of this site.
#
#  Parts (switched on below):
#     A. Full cluster at the installed PV (both load scenarios).
#     B. Full cluster with extra PV per nanogrid (heavy load scenario; by
#        default +3.0 kWp on the Small Lodge, 7.2 kWp in total).
#     C. Every subset of nanogrids, from the results of scenario_clustersize.py
#        (basic load scenario).
#  Output filenames include the modelling window.
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from configuration import *
from optimiser import capital_recovery_factor
from cabling import cable_length_routed
from scenario_comparison import collect_results, isolated_is_feasible
from load_profile import prepare_model_loads, MODEL_BUILDINGS
from pv_profile import build_pv_for_window, capacity_per_grid
from wind_profile import build_wind_for_window
from operation import run_isolated, run_interconnected                 # the two operating modes, used by part B


# ============================================================================
#  Settings
# ============================================================================
CABLE_COST_BASE_EUR_M = 30.0     # base installed cable cost per metre [EUR/m]
CABLE_COST_LOW_EUR_M  = 20.0
CABLE_COST_HIGH_EUR_M = 85.0

# --- Parts to run ---
RUN_INSTALLED_PV = True
RUN_CUSTOM_PV    = False
RUN_SUBSETS      = True

# --- Part B: extra PV per nanogrid [kWp] on top of the installed capacity ---
# Default: the lowest isolated LCOE of scenario_pv_sensitivity.py (7.2 kWp in total).
# Keys are data column names (LOAD_COLUMNS).
PV_EXTRA_KWP = {
    "Big house":     0.0,
    "Small house":   3.0,
    "Utility house": 0.0,
    "Greenhouse":    0.0,
}
CUSTOM_PV_CONFIGS = [CONFIG_PV_BESS]

# --- Part C: results file of scenario_clustersize.py ---
SUBSET_CSV           = "outputs/clustersize_subsets.csv"
SUBSET_FIGURE_CONFIG = "PV-BESS"

# --- Outputs ---
SHOW_BREAKEVEN_DISTANCE = True
SHOW_LCOE_BAND          = True
SHOW_SUBSET_FIGURE      = True
WRITE_TABLE_CSV         = True   # also save the console tables as CSV files

# The heavy load scenario needs a battery sweep up to 250 kWh.
HEAVY_MIN_BATTERY_MAX_KWH = 250

# Routed cable length of the full cluster [m]
ROUTED_LENGTH_M = cable_length_routed(NANOGRID_COORDINATES)

# Capital recovery factor of the cable [1/year]
CRF_CABLE = capital_recovery_factor(DISCOUNT_RATE, CABLE_LIFETIME)

WINDOW_TAG = f"{MODEL_WINDOW_START}_to_{MODEL_WINDOW_END}"   # goes into every output filename


# ============================================================================
#  Settings check
# ============================================================================
def check_settings():
    problems = []
    if CABLING_COSTS_ON:
        problems.append("CABLING_COSTS_ON is True; this script adds the cable analytically on the free-cable baseline, so set it to False.")
    if USE_2025_2026_WINDOW and (RUN_INSTALLED_PV or RUN_CUSTOM_PV) and BATTERY_MAX_KWH < HEAVY_MIN_BATTERY_MAX_KWH:
        problems.append(f"The heavy-load window is on but Battery_max_kWh is {BATTERY_MAX_KWH}; "
                        f"raise it to {HEAVY_MIN_BATTERY_MAX_KWH} so the isolated Small Lodge can be sized.")
    if not (RUN_INSTALLED_PV or RUN_CUSTOM_PV or RUN_SUBSETS):
        problems.append("All three parts are switched off; set at least one RUN_ toggle to True.")
    if problems:
        print("\nscenario_cabling cannot run with the current settings:")
        for p in problems:
            print("  - " + p)
        print("Fix these in configuration.py (or the toggles above) and RESTART THE KERNEL, then run this file again.")
        raise SystemExit

    load_name = "HEAVY A/C load" if USE_2025_2026_WINDOW else "BASIC household load"
    print(f"\nLoad scenario: {load_name}  ({MODEL_WINDOW_START} to {MODEL_WINDOW_END})")


# ============================================================================
#  Cable cost in closed form: the cable adds crf_cable * price * length / annual load
#  to the interconnected LCOE, independent of the battery size.
# ============================================================================
def cable_lcoe_per_eur_m(cluster_load_kWh, length_m=ROUTED_LENGTH_M):
    """LCOE added by the cable per EUR/m of cable price [EUR/kWh per EUR/m]."""
    return CRF_CABLE * length_m / cluster_load_kWh


def compute_cable_sensitivity(rows):
    """Interconnected LCOE at the low, base and high cable price, and the break-even price, per configuration."""
    out = []
    for r in rows:
        slope = cable_lcoe_per_eur_m(r["cluster_load_kWh"])              # EUR/kWh per EUR/m
        lcoe_nocable = r["lcoe_inter"]                                   # free-cable interconnected LCOE

        def lcoe_at(price):                                              # LCOE if the cable cost this much per metre
            return lcoe_nocable + price * slope

        # Break-even price: lcoe_nocable + price * slope = lcoe_iso (only with a
        # feasible isolated case and a saving with a free cable)
        gap_free_cable = r["lcoe_iso"] - lcoe_nocable                    # EUR/kWh that pooling saves with a free cable
        if r["iso_feasible"] and gap_free_cable > 0 and slope > 0:
            breakeven_price = gap_free_cable / slope                     # EUR/m
        else:
            breakeven_price = None

        out.append({
            "name":            r["name"],
            "lcoe_iso":        r["lcoe_iso"],
            "iso_feasible":    r["iso_feasible"],
            "lcoe_nocable":    lcoe_nocable,
            "gap_free_cable":  gap_free_cable,
            "lcoe_low":        lcoe_at(CABLE_COST_LOW_EUR_M),
            "lcoe_base":       lcoe_at(CABLE_COST_BASE_EUR_M),
            "lcoe_high":       lcoe_at(CABLE_COST_HIGH_EUR_M),
            "cable_part_base": CABLE_COST_BASE_EUR_M * slope,            # what the cable adds at the base price
            "breakeven_price": breakeven_price,
        })
    return out


def compute_breakeven(rows):
    """Break-even cable length per configuration at the low, base and high cable price.

    "reason" states why a configuration has no break-even length.
    """
    out = []
    for r in rows:
        gap_eur_kWh   = r["lcoe_iso"] - r["lcoe_inter"]
        annual_saving = gap_eur_kWh * r["cluster_load_kWh"]             # EUR/year saved before paying for any cable

        if not r["iso_feasible"]:
            out.append({"name": r["name"], "feasible": False, "reason": "isolated case unfeasible",
                        "annual_saving": annual_saving, "L_base": 0.0, "L_low_cost": 0.0, "L_high_cost": 0.0})
            continue
        if annual_saving <= 0:
            out.append({"name": r["name"], "feasible": False, "reason": "no saving from interconnection",
                        "annual_saving": annual_saving, "L_base": 0.0, "L_low_cost": 0.0, "L_high_cost": 0.0})
            continue

        out.append({
            "name":          r["name"],
            "feasible":      True,
            "reason":        "",
            "annual_saving": annual_saving,
            "L_base":        annual_saving / (CRF_CABLE * CABLE_COST_BASE_EUR_M),
            "L_low_cost":    annual_saving / (CRF_CABLE * CABLE_COST_LOW_EUR_M),    # cheapest cable, longest distance
            "L_high_cost":   annual_saving / (CRF_CABLE * CABLE_COST_HIGH_EUR_M),   # dearest cable, shortest distance
        })
    return out


# ============================================================================
#  Part B: full cluster with extra PV per nanogrid. PV generation is scaled
#  per nanogrid and the PV cost follows the new cluster total.
# ============================================================================
def collect_custom_pv_results():
    loads, time_index = prepare_model_loads()
    n_nanogrids = loads.shape[1]

    # Installed PV per nanogrid
    base_sizes = np.asarray(capacity_per_grid(loads, PV_CAPACITY_KW, ALLOCATION), dtype=float)   # kWp per nanogrid

    # Unknown names in PV_EXTRA_KWP
    for name in PV_EXTRA_KWP:
        if name not in MODEL_BUILDINGS:
            print(f"  WARNING: '{name}' in PV_EXTRA_KWP is not a building name; it is ignored. Valid names: {MODEL_BUILDINGS}")

    # Extra PV per nanogrid in column order
    extra_sizes = np.array([PV_EXTRA_KWP.get(name, 0.0) for name in MODEL_BUILDINGS], dtype=float)
    new_sizes   = base_sizes + extra_sizes

    pv_base = build_pv_for_window(time_index, n_nanogrids, SOLAR_FILE, loads)   # generation at the installed PV
    pv_new  = pv_base * (new_sizes / base_sizes)                                  # scale each nanogrid's column
    wind    = build_wind_for_window(time_index, n_nanogrids, SOLAR_FILE, loads)  # wind stays as installed

    # Mean PV per nanogrid for the cost calculation
    overrides = {"PV_CAPACITY_KW": new_sizes.sum() / n_nanogrids}

    print("\n  Custom PV per nanogrid [kWp]")
    print(f"  {'nanogrid':<16}{'installed':>11}{'extra':>8}{'modelled':>10}")
    for name, b, e, s in zip(MODEL_BUILDINGS, base_sizes, extra_sizes, new_sizes):   # zip walks the four lists side by side
        print(f"  {name:<16}{b:>11.3f}{e:>8.2f}{s:>10.3f}")
    print(f"  {'total':<16}{base_sizes.sum():>11.3f}{extra_sizes.sum():>8.2f}{new_sizes.sum():>10.3f}")

    rows = []
    for config in CUSTOM_PV_CONFIGS:
        gen = pv_new + wind if config["use_wind"] else pv_new
        print(f"\n[{config['name']} | custom PV {new_sizes.sum():.1f} kWp]")
        _, inter_optimal, _ = run_interconnected(gen, loads, config=config, overrides=overrides)
        iso_result, _       = run_isolated(gen, loads, config=config, overrides=overrides)
        rows.append({
            "name":             config["name"],
            "lcoe_iso":         iso_result["lcoe_eur_kWh"],
            "lcoe_inter":       float(inter_optimal["lcoe_eur_kWh"]),
            "battery_iso":      iso_result["battery_kWh_total"],
            "battery_inter":    float(inter_optimal["battery_kWh_total"]),
            "cluster_load_kWh": iso_result["cluster_load_kWh"],
            "iso_feasible":     isolated_is_feasible(iso_result, config),
        })

    window_label = f"{MODEL_WINDOW_START} to {MODEL_WINDOW_END}, PV {new_sizes.sum():.1f} kWp"
    return rows, n_nanogrids, window_label, new_sizes.sum()


# ============================================================================
#  Part C: every subset of nanogrids, each with its own cable (MST over its
#  nanogrids) and annual load, from the results of scenario_clustersize.py.
# ============================================================================
def compute_subset_cabling():
    if not os.path.exists(SUBSET_CSV):          # os.path.exists: True if the file is there
        print(f"\n  Part C skipped: {SUBSET_CSV} not found. Run scenario_clustersize.py first (basic load window).")
        return None

    df = pd.read_csv(SUBSET_CSV)
    loads, _ = prepare_model_loads()
    hours_per_step = TIME_STEP_MIN / 60          # 15-min steps: 0.25 h, to turn kW into kWh

    out = []
    for _, r in df.iterrows():
        tags   = r["short"].split("+")           # "BL+SL": ["BL", "SL"]
        cols   = [NANOGRID_TAGS.index(t) for t in tags]              # tag: column number of that nanogrid
        coords = [NANOGRID_COORDINATES[c] for c in cols]             # only the buildings in this subset
        length_m = cable_length_routed(coords)                    # routed MST over those buildings [m]
        load_kWh = loads[:, cols].sum() * hours_per_step          # annual load of the subset [kWh]

        slope    = cable_lcoe_per_eur_m(load_kWh, length_m)       # EUR/kWh per EUR/m, for this subset
        gap      = r["lcoe_iso"] - r["lcoe_inter"]                # saving with a free cable [EUR/kWh]
        cable    = CABLE_COST_BASE_EUR_M * slope
        pays     = gap > cable

        if gap > 0:
            breakeven_price = gap / slope                                     # EUR/m at this subset's own length
            breakeven_len   = gap * load_kWh / (CRF_CABLE * CABLE_COST_BASE_EUR_M)   # m at the base price
        else:
            breakeven_price = None
            breakeven_len   = 0.0

        out.append({
            "config":           r["config"],
            "size":             int(r["size"]),
            "short":            r["short"],
            "length_m":         length_m,
            "load_kWh":         load_kWh,
            "lcoe_iso":         r["lcoe_iso"],
            "gap_free_cable":   gap,
            "cable_part_base":  cable,
            "net_benefit_base": gap - cable,
            "pays_at_base":     pays,
            "breakeven_price":  breakeven_price,
            "breakeven_len_m":  breakeven_len,
        })
    return pd.DataFrame(out)


def print_subset_table(sub):
    print("\n" + "=" * 96)
    print(f"  CABLING PER SUBSET  (basic load, cable {CABLE_COST_BASE_EUR_M:.0f} EUR/m, all values EUR/kWh unless stated)")
    print("=" * 96)
    for config_name in sub["config"].unique():                        # .unique(): each configuration once
        part = sub[sub["config"] == config_name]
        print(f"\n  {config_name}")
        print(f"  {'subset':<14}{'cable m':>9}{'load kWh':>10}{'saving':>9}{'cable':>9}{'net':>9}"
              f"{'pays?':>7}{'b-e EUR/m':>11}{'b-e m':>9}")
        print("  " + "-" * 94)
        for _, r in part.iterrows():
            be_p = f"{r['breakeven_price']:.0f}" if pd.notna(r["breakeven_price"]) else "none"   # pd.notna: True if not empty
            print(f"  {r['short']:<14}{r['length_m']:>9.1f}{r['load_kWh']:>10.0f}{r['gap_free_cable']:>9.3f}"
                  f"{r['cable_part_base']:>9.3f}{r['net_benefit_base']:>9.3f}{('yes' if r['pays_at_base'] else 'no'):>7}"
                  f"{be_p:>11}{r['breakeven_len_m']:>9.0f}")
    print("=" * 96)
    print("  saving = free-cable LCOE gap; cable = what the cable adds at the base price; net = saving minus cable")
    print("  b-e EUR/m = break-even price at this subset's own cable length; b-e m = break-even length at the base price")


# ============================================================================
#  Figure: break-even cable length per configuration. Band from the high to the
#  low cable price, dot at the base price, vertical line at the routed length.
# ============================================================================
def plot_breakeven_distance(breakeven, n_nanogrids, window_label, file_tag=""):

    fig, ax = plt.subplots(figsize=(9, 4.6))

    xmax = ROUTED_LENGTH_M   # grows to fit the longest cheap-cable end below
    for i, b in enumerate(breakeven):
        if not b["feasible"]:
            # No break-even length: state the reason
            ax.annotate(f"no break-even ({b['reason']})", xy=(0, i),
                        xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=8, style="italic", color="grey")
            continue

        dear  = b["L_high_cost"]   # dearest cable (85 EUR/m), shortest break-even
        cheap = b["L_low_cost"]    # cheapest cable (20 EUR/m), longest break-even
        base  = b["L_base"]        # base price (30 EUR/m)
        xmax  = max(xmax, cheap)

        # Band
        ax.plot([dear, cheap], [i, i], color=COLOUR_INTERCONNECTED, lw=7,
                solid_capstyle="round", alpha=0.45, zorder=1)
        # Base price
        ax.plot(base, i, "o", color=COLOUR_INTERCONNECTED, markersize=12,
                markeredgecolor="black", zorder=3)
        # Value labels
        ax.annotate(f"{base:.0f} m", (base, i), xytext=(0, 13), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.annotate(f"{dear:.0f}", (dear, i), xytext=(-9, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8, color="grey")
        ax.annotate(f"{cheap:.0f}", (cheap, i), xytext=(9, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color="grey")

    # Routed cluster length
    ax.axvline(ROUTED_LENGTH_M, color="black", linestyle="--", linewidth=1.2, zorder=2)
    trans = blended_transform_factory(ax.transData, ax.transAxes)   # x in data units, y in axis fraction
    ax.text(ROUTED_LENGTH_M, 0.04, f"actual cluster\nlength {ROUTED_LENGTH_M:.1f} m",
            transform=trans, ha="center", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey"))

    ax.set_yticks(range(len(breakeven)))
    ax.set_yticklabels([b["name"] for b in breakeven])
    ax.set_ylim(len(breakeven) - 0.5, -0.7)
    ax.set_xlim(0, xmax * 1.12)
    ax.set_xlabel("Break-even cluster cable length (m)")
    ax.set_axisbelow(True)                                           # gridlines behind the band and dots
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fname = f"outputs/cabling_breakeven_distance_{WINDOW_TAG}{file_tag}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Figure: isolated LCOE next to the interconnected LCOE at the base cable price,
#  with a whisker for the low and high price and the break-even price noted.
# ============================================================================
def plot_lcoe_band(cable_sens, n_nanogrids, window_label, file_tag=""):
    names = [s["name"] for s in cable_sens]
    x = np.arange(len(names))
    width = 0.38

    iso_vals   = [s["lcoe_iso"]  if s["iso_feasible"] else np.nan for s in cable_sens]
    inter_base = [s["lcoe_base"] for s in cable_sens]
    # Whisker from the low to the high cable price
    lower = [s["lcoe_base"] - s["lcoe_low"]  for s in cable_sens]
    upper = [s["lcoe_high"] - s["lcoe_base"] for s in cable_sens]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, iso_vals, width, label="Isolated",
           color=COLOUR_ISOLATED, edgecolor="black")
    ax.bar(x + width / 2, inter_base, width, label=f"Interconnected, cable at {CABLE_COST_BASE_EUR_M:.0f} EUR/m",
           color=COLOUR_INTERCONNECTED, edgecolor="black",
           yerr=[lower, upper], capsize=6, error_kw=dict(ecolor="black", lw=1.2))

    # Value labels and break-even price
    for i, s in enumerate(cable_sens):
        if s["iso_feasible"]:
            ax.annotate(f"{s['lcoe_iso']:.2f}", xy=(x[i] - width / 2, s["lcoe_iso"]),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        else:
            ax.annotate("isolated\nunfeasible", xy=(x[i] - width / 2, 0),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                        style="italic", color="grey")
        ax.annotate(f"{s['lcoe_base']:.2f}", xy=(x[i] + width / 2, s["lcoe_high"]),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
        if s["breakeven_price"] is not None:
            note = f"pays below\n{s['breakeven_price']:.0f} EUR/m"
        else:
            note = "no break-even\nprice"
        ax.annotate(note, xy=(x[i] + width / 2, s["lcoe_high"]),
                    xytext=(0, 20), textcoords="offset points", ha="center",
                    fontsize=8, color="dimgrey")

    tallest = max([s["lcoe_high"] for s in cable_sens] +
                  [s["lcoe_iso"] for s in cable_sens if s["iso_feasible"]])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("LCOE (EUR/kWh)")
    ax.set_ylim(top=tallest * 1.30)
    ax.set_axisbelow(True)                                           # gridlines behind the bars
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fname = f"outputs/cabling_lcoe_band_{WINDOW_TAG}{file_tag}.png"
    save_figure(plt.gcf(), fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Figure: per subset, the saving with a free cable next to the cable cost at
#  the base price [EUR/kWh], grouped by N.
# ============================================================================
def plot_subset_cabling(sub, config_name):
    part = sub[sub["config"] == config_name].reset_index(drop=True)
    if part.empty:                                  # .empty: True if no rows matched
        print(f"  No subset rows for {config_name}; figure skipped.")
        return

    y = np.arange(len(part))
    h = 0.38                                        # bar thickness

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(part) + 1.8))
    ax.barh(y - h / 2, part["gap_free_cable"], h, color=COLOUR_INTERCONNECTED,
            edgecolor="black", label="Saving from interconnection (free cable)")
    ax.barh(y + h / 2, part["cable_part_base"], h, color="lightgrey",
            edgecolor="black", label=f"Cable cost at {CABLE_COST_BASE_EUR_M:.0f} EUR/m")

    # Net benefit and cable length
    xmax = max(part["gap_free_cable"].max(), part["cable_part_base"].max())
    for i, r in part.iterrows():
        colour = "darkgreen" if r["pays_at_base"] else "firebrick"
        ax.text(xmax * 1.03, i, f"net {r['net_benefit_base']:+.3f}  ({r['length_m']:.0f} m)",
                va="center", fontsize=8, color=colour)

    # Separators between the groups
    sizes = part["size"].to_numpy()
    for i in range(1, len(sizes)):
        if sizes[i] != sizes[i - 1]:
            ax.axhline(i - 0.5, color="grey", linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}  (N={n})" for s, n in zip(part["short"], part["size"])])
    ax.set_ylim(len(part) - 0.5, -0.5)              # first subset at the top
    ax.set_xlim(0, xmax * 1.65)
    ax.set_xlabel("EUR/kWh")
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=8, frameon=False)

    plt.tight_layout()
    tag = config_name.replace(" ", "")
    fname = f"outputs/cabling_subsets_{tag}_{WINDOW_TAG}.png"
    save_figure(plt.gcf(), fname, scale=1.6, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


# ============================================================================
#  Tables, CSVs and figures for the full cluster (parts A and B)
# ============================================================================
def report_full_cluster(rows, n_nanogrids, window_label, file_tag=""):
    breakeven  = compute_breakeven(rows)
    cable_sens = compute_cable_sensitivity(rows)

    # Break-even length
    print("\n" + "=" * 74)
    print(f"  CABLING BREAK-EVEN DISTANCE  ({window_label})")
    print(f"  routed cluster length {ROUTED_LENGTH_M:.1f} m | cable {CABLE_COST_BASE_EUR_M:.0f} EUR/m "
          f"(band {CABLE_COST_LOW_EUR_M:.0f} to {CABLE_COST_HIGH_EUR_M:.0f}) | lifetime {CABLE_LIFETIME} yr")
    print("=" * 74)
    print(f"  {'config':<16}{'saving EUR/yr':>15}{'break-even m':>14}{'range m (dear..cheap)':>26}")
    print("-" * 74)
    for b in breakeven:
        if b["feasible"]:
            print(f"  {b['name']:<16}{b['annual_saving']:>15.0f}{b['L_base']:>14.0f}"
                  f"{b['L_high_cost']:>15.0f} ..{b['L_low_cost']:>8.0f}")
        else:
            print(f"  {b['name']:<16}{'no break-even: ' + b['reason']:>55}")
    print("=" * 74)

    # Break-even price at the routed length
    print("\n" + "=" * 86)
    print(f"  CABLE PRICE SENSITIVITY AT {ROUTED_LENGTH_M:.1f} m  ({window_label})")
    print("=" * 86)
    print(f"  {'config':<16}{'isolated':>10}{'free cable':>12}{'at 20/m':>10}{'at 30/m':>10}{'at 85/m':>10}{'break-even EUR/m':>18}")
    print("-" * 86)
    for s in cable_sens:
        be  = f"{s['breakeven_price']:.0f}" if s["breakeven_price"] is not None else "none"
        iso = f"{s['lcoe_iso']:.3f}" if s["iso_feasible"] else "unfeas."
        print(f"  {s['name']:<16}{iso:>10}{s['lcoe_nocable']:>12.3f}{s['lcoe_low']:>10.3f}"
              f"{s['lcoe_base']:>10.3f}{s['lcoe_high']:>10.3f}{be:>18}")
    print("=" * 86)
    print(f"  cable adds {cable_sens[0]['cable_part_base']:.3f} EUR/kWh at {CABLE_COST_BASE_EUR_M:.0f} EUR/m "
          f"(cluster load {rows[0]['cluster_load_kWh']:.0f} kWh/yr)")

    if WRITE_TABLE_CSV:
        n1 = f"outputs/cabling_breakeven_distance_{WINDOW_TAG}{file_tag}.csv"
        n2 = f"outputs/cabling_price_sensitivity_{WINDOW_TAG}{file_tag}.csv"
        pd.DataFrame(breakeven).to_csv(n1, index=False)
        pd.DataFrame(cable_sens).to_csv(n2, index=False)
        print(f"Saved: {n1}")
        print(f"Saved: {n2}")

    if SHOW_BREAKEVEN_DISTANCE:
        plot_breakeven_distance(breakeven, n_nanogrids, window_label, file_tag)
    if SHOW_LCOE_BAND:
        plot_lcoe_band(cable_sens, n_nanogrids, window_label, file_tag)


# ============================================================================
#  Run
# ============================================================================
if __name__ == "__main__":

    check_settings()
    os.makedirs("outputs", exist_ok=True)

    # Part A
    if RUN_INSTALLED_PV:
        print("\n### PART A: full cluster, installed PV ###")
        rows, n_nanogrids, window_label = collect_results()
        report_full_cluster(rows, n_nanogrids, window_label, file_tag="")

    # Part B
    if RUN_CUSTOM_PV:
        print("\n### PART B: full cluster, custom PV ###")
        rows_b, n_nanogrids, window_label_b, pv_total = collect_custom_pv_results()
        report_full_cluster(rows_b, n_nanogrids, window_label_b, file_tag=f"_pv{pv_total:.1f}kWp")

    # Part C
    if RUN_SUBSETS:
        print("\n### PART C: cabling per subset ###")
        if USE_2025_2026_WINDOW:
            print("  Skipped: part C uses the cluster-size results, which are for the basic load (2025).")
            print("  Set USE_2025_2026_WINDOW = False and restart the kernel to run it.")
        else:
            print(f"  Reading {SUBSET_CSV}. This file has no window in its name, so make sure")
            print("  scenario_clustersize.py was last run on the basic load (2025) window.")
            sub = compute_subset_cabling()
            if sub is not None:
                print_subset_table(sub)
                if WRITE_TABLE_CSV:
                    n3 = f"outputs/cabling_subsets_{WINDOW_TAG}.csv"
                    sub.to_csv(n3, index=False)
                    print(f"Saved: {n3}")
                if SHOW_SUBSET_FIGURE:
                    plot_subset_cabling(sub, SUBSET_FIGURE_CONFIG)
