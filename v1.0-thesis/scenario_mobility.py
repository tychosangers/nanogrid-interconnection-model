"""
scenario_mobility.py
SQ4.5: curtailment headroom for a mobile battery on the real Tinos Ecolodge.

The site is heavily PV oversized, so the stationary battery fills up mid morning
and the PV produced afterwards has nowhere to go. This script fixes the pooled
battery at its real installed size, runs the PV-only dispatch once, and reads off
the curtailed energy as the room a mobile battery could soak up.

Because the interconnected scenario pools the cluster into a single residual
nanogrid with one shared battery, running the pooled site here gives exactly the
same curtailment as the interconnected result. The mobile battery is treated
post hoc: each day it starts empty and can absorb up to the smaller of that day's
curtailment or its own size. It never competes with the load or the stationary
battery, so the captured energy is a clean upper bound.

Curtailment (surplus PV) and unserved load (reliability) are separate things.
The curtailment figures answer SQ4.5. The unmet-load figure and table are a
reliability cross-check: they show whether the fixed battery leaves any load
unserved, which is the real signal of the airco stressing the winter window.

Press run. All choices are the toggles just below.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from configuration import Time_step_min, SOC_max, SOC_min
from load_profile import prepare_model_loads
from pv_profile import build_pv_for_window
from operation import pool_cluster_residual
from simulation import run_simulation


# ============================================================================
#  Toggles
# ============================================================================

STATIONARY_BATTERY_KWH = 14.0          # real installed pooled battery [kWh], nominal (20% floor applies)

# Study window. False = calendar year 2025 (the SQ4.5 headline). True = June 2025
# to May 2026, which includes the small-lodge airco from February 2026. Each
# window also picks its own matching Solcast file, so PV and load line up.
USE_2025_2026_WINDOW = False

if USE_2025_2026_WINDOW:
    WINDOW_START = "2025-06-01"
    WINDOW_END   = "2026-05-31"
    SOLAR_FILE   = "data/solar/Solar_Wind_Data_TinosEcoLodge_2025_2026_Solcast.csv"
else:
    WINDOW_START = "2025-01-01"
    WINDOW_END   = "2025-12-31"
    SOLAR_FILE   = "data/solar/Solar_Data_TinosEcoLodge_2025_Solcast.csv"

# FOCUS_MONTHS drives the average-day profile, the battery fill time and the fill
# fraction. Default is the whole window; set [6, 7, 8] with FOCUS_LABEL "summer"
# to zoom back into the summer-only view.
FOCUS_MONTHS = list(range(1, 13))     # months used for the profile, fill timing and fill line
FOCUS_LABEL  = "Summer"                 # short word shown in titles and the summary

SUMMER_MONTHS = [6, 7, 8]              # fixed reference months for the summer daily average
WINTER_MONTHS = [12, 1, 2]            # fixed reference months for the winter daily average

MOBILE_SIZES = np.arange(0.0, 20.5, 0.5)   # mobile battery sizes tested on the capture curve [kWh]
EXAMPLE_SIZES = [2.0, 5.0, 10.0]           # sizes printed in the summary (e-bike, small pack, EV-ish)

FIGURES_DIR = Path("figures")
OUTPUTS_DIR = Path("outputs")


# ============================================================================
#  1. Run the pooled PV-only dispatch once at the fixed battery
# ============================================================================

hours_per_step = Time_step_min / 60.0                        # 0.25 h per 15-min step

loads, time_index = prepare_model_loads(start=WINDOW_START, end=WINDOW_END)   # (n_steps, 4) real lodges
n_households = loads.shape[1]
pv = build_pv_for_window(time_index, n_households, SOLAR_FILE, loads)                # (n_steps, 4), aligned to the load window

pooled_pv, pooled_load = pool_cluster_residual(pv, loads)    # collapse the cluster to one residual nanogrid
sim = run_simulation(pooled_pv, pooled_load, STATIONARY_BATTERY_KWH, use_diesel=False)   # PV only, no diesel

# Whole-site PV and load (summed across lodges) are used for the profile figure.
total_pv   = pv.sum(axis=1)
total_load = loads.sum(axis=1)


# ============================================================================
#  2. Tidy dataframe on the time index
# ============================================================================

# One row per 15-min step. curtailed and unmet come from the dispatch in kW;
# multiplying by the step length turns them into energy for that step [kWh].
df = pd.DataFrame({
    "pv_kw":         total_pv,
    "load_kw":       total_load,
    "curtailed_kw":  sim["curtailed"],
    "unmet_kw":      sim["unmet_kw"],
    "soc_kwh":       sim["soc_kWh"],
}, index=time_index)

df["curtailed_kwh"] = df["curtailed_kw"] * hours_per_step
df["unmet_kwh"]     = df["unmet_kw"] * hours_per_step

# Helper columns. .date drops the time so timesteps on the same day share a key;
# .month and .hour read the island local clock (the index is already in Athens time).
df["date"]  = df.index.date
df["month"] = df.index.month
df["hour"]  = df.index.hour

# Month axis, shared by the summary table and the figures.
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
x = np.arange(1, 13)


# ============================================================================
#  3. Curtailment and reliability: annual, seasonal, monthly
# ============================================================================

annual_curt_kwh  = df["curtailed_kwh"].sum()
annual_pv_kwh    = total_pv.sum() * hours_per_step          # whole-site PV generated over the window [kWh]
annual_load_kwh  = total_load.sum() * hours_per_step        # whole-site load over the window [kWh]
annual_unmet_kwh = df["unmet_kwh"].sum()                    # load PV+battery could not serve [kWh]
curt_share_of_pv = annual_curt_kwh / annual_pv_kwh * 100 if annual_pv_kwh else 0.0     # % of generation wasted
renewable_fraction = (annual_load_kwh - annual_unmet_kwh) / annual_load_kwh * 100 if annual_load_kwh else 0.0

# groupby("date") gathers the 96 quarter-hour rows of each day; .agg sums the
# curtailed energy and keeps that day's month (same for every row that day), so
# we get one tidy row per day with its month attached.
daily = df.groupby("date").agg(
    curtailed_kwh=("curtailed_kwh", "sum"),
    month=("month", "first"),
)

summer_daily = daily[daily["month"].isin(SUMMER_MONTHS)]["curtailed_kwh"]
winter_daily = daily[daily["month"].isin(WINTER_MONTHS)]["curtailed_kwh"]

summer_mean_daily = summer_daily.mean()
winter_mean_daily = winter_daily.mean()

# Mean daily curtailment per month, and total unserved load per month.
monthly_mean_daily = daily.groupby("month")["curtailed_kwh"].mean()
monthly_unmet_kwh  = df.groupby("month")["unmet_kwh"].sum()


# ============================================================================
#  4. When does the battery fill up? (over the focus months)
# ============================================================================

soc_full_kwh = SOC_max * STATIONARY_BATTERY_KWH             # top of the usable window [kWh]

# A boolean column: True on any step where the battery is at (or within a hair of) full.
df["is_full"] = df["soc_kwh"] >= soc_full_kwh - 0.01

# Decimal hour of day, so 11:15 reads as 11.25 and can be averaged.
df["hour_decimal"] = df.index.hour + df.index.minute / 60.0

# For each day that fills, the earliest hour at which it first hits full.
full_times = df[df["is_full"]].groupby("date")["hour_decimal"].min().reset_index()
full_times["month"] = pd.to_datetime(full_times["date"]).dt.month
focus_full_times = full_times[full_times["month"].isin(FOCUS_MONTHS)]["hour_decimal"]

median_full_hour = focus_full_times.median()

# Share of focus-month days on which the battery reaches full at all.
n_focus_days = daily[daily["month"].isin(FOCUS_MONTHS)].shape[0]
share_focus_days_full = len(focus_full_times) / n_focus_days * 100 if n_focus_days else 0.0


# ============================================================================
#  5. Mobile-battery capture curve
# ============================================================================

all_daily = daily["curtailed_kwh"].values                  # per-day curtailment, whole window [kWh]
focus_daily = daily[daily["month"].isin(FOCUS_MONTHS)]["curtailed_kwh"]
focus_daily_vals = focus_daily.values                      # per-day curtailment over the focus months [kWh]

captured_kwh   = []                                        # energy a mobile pack of each size would recover per year
captured_share = []                                        # that energy as a share of all curtailment
focus_full_frac = []                                       # share of focus-month days the pack fills completely

for c in MOBILE_SIZES:
    # np.minimum compares element by element: for each day take the smaller of
    # that day's curtailment or the pack size, then sum over the year.
    captured = np.minimum(all_daily, c).sum()
    captured_kwh.append(captured)
    captured_share.append(captured / annual_curt_kwh * 100 if annual_curt_kwh else 0.0)

    if c == 0:
        focus_full_frac.append(0.0)
    else:
        # A boolean array of days where curtailment >= pack size; .mean() of a
        # boolean array is the fraction that are True.
        focus_full_frac.append((focus_daily_vals >= c).mean() * 100)

captured_kwh   = np.array(captured_kwh)
captured_share = np.array(captured_share)
focus_full_frac = np.array(focus_full_frac)


# ============================================================================
#  6. Printed summary
# ============================================================================

print("=" * 62)
print("SQ4.5  Mobile-battery curtailment headroom (PV only, pooled site)")
print("=" * 62)
print(f"  Window              : {WINDOW_START} to {WINDOW_END}")
print(f"  Stationary battery  : {STATIONARY_BATTERY_KWH:.1f} kWh (pooled, 20% floor)")
print(f"  Focus months        : {FOCUS_LABEL}")
print("  " + "-" * 56)
print(f"  Annual curtailment  : {annual_curt_kwh:8.0f} kWh/year")
print(f"  Curtailed share     : {curt_share_of_pv:8.0f} % of PV generated")
print(f"  Mean summer day     : {summer_mean_daily:8.1f} kWh/day")
print(f"  Mean winter day     : {winter_mean_daily:8.1f} kWh/day")
print(f"  Battery full by     : {median_full_hour:8.2f} h (median {FOCUS_LABEL} day)")
print(f"  Days filling ({FOCUS_LABEL:>6}) : {share_focus_days_full:8.0f} %")
print("  " + "-" * 56)
print(f"  Unserved load       : {annual_unmet_kwh:8.0f} kWh/year")
print(f"  Renewable fraction  : {renewable_fraction:8.1f} % (share of load served by PV+battery)")
print("  Unserved load per month (PV-only reliability check):")
for m in x:
    u = monthly_unmet_kwh.get(m, 0.0)
    flag = "  <-- unmet load" if u > 0.5 else ""
    print(f"    {month_labels[m-1]} : {u:7.1f} kWh{flag}")
print("  " + "-" * 56)
print("  Mobile pack capture (from otherwise-curtailed PV):")
for c in EXAMPLE_SIZES:
    cap = np.minimum(all_daily, c).sum()
    share = cap / annual_curt_kwh * 100 if annual_curt_kwh else 0.0
    fill = (focus_daily_vals >= c).mean() * 100
    print(f"    {c:5.1f} kWh pack : {cap:7.0f} kWh/year "
          f"({share:4.0f}% of curtailment, fills on {fill:3.0f}% of {FOCUS_LABEL} days)")
print("=" * 62)


# ============================================================================
#  7. Figures
# ============================================================================

FIGURES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

CURT_COLOUR = "#c0392b"
PV_COLOUR   = "#f5a800"
LOAD_COLOUR = "#2c3e50"
SOC_COLOUR  = "#2980b9"
UNMET_COLOUR = "#8e44ad"

# --- Figure 1: curtailment over the window and by month ---
fig1, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7))

daily_series = daily["curtailed_kwh"]
daily_series.index = pd.to_datetime(daily_series.index)
ax_top.fill_between(daily_series.index, daily_series.values, color=CURT_COLOUR, alpha=0.5)
ax_top.plot(daily_series.index, daily_series.values, color=CURT_COLOUR, linewidth=0.8)
ax_top.set_ylabel("Curtailed PV (kWh/day)")
ax_top.set_title(f"Daily curtailment, PV only, {STATIONARY_BATTERY_KWH:.0f} kWh pooled battery")
ax_top.grid(axis="y", alpha=0.3)
ax_top.margins(x=0.01)

curt_vals = [monthly_mean_daily.get(m, 0.0) for m in x]
ax_bot.bar(x, curt_vals, color=CURT_COLOUR, edgecolor="white", linewidth=0.5)
for xi, v in zip(x, curt_vals):
    if v > 0:
        ax_bot.text(xi, v + max(curt_vals) * 0.01, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
ax_bot.set_xticks(x)
ax_bot.set_xticklabels(month_labels)
ax_bot.set_ylabel("Mean curtailment (kWh/day)")
ax_bot.set_title("Average daily curtailment per month")
ax_bot.grid(axis="y", alpha=0.3)

fig1.tight_layout()
fig1.savefig(FIGURES_DIR / "fig_sq45_curtailment_year.png", dpi=150)
plt.show()

# --- Figure 2: the average day over the focus months ---
focus = df[df["month"].isin(FOCUS_MONTHS)]
profile = focus.groupby("hour").agg(
    pv_kw=("pv_kw", "mean"),
    load_kw=("load_kw", "mean"),
    curtailed_kw=("curtailed_kw", "mean"),
    soc_kwh=("soc_kwh", "mean"),
)

fig2, ax = plt.subplots(figsize=(11, 5))
hours = profile.index

ax.fill_between(hours, profile["pv_kw"], color=PV_COLOUR, alpha=0.25, label="PV production")
ax.plot(hours, profile["pv_kw"], color=PV_COLOUR, linewidth=1.5)
ax.plot(hours, profile["load_kw"], color=LOAD_COLOUR, linewidth=1.5, label="Load")
ax.fill_between(hours, profile["curtailed_kw"], color=CURT_COLOUR, alpha=0.6, label="Curtailed (headroom)")
ax.set_xlabel("Hour of day (local)")
ax.set_ylabel("Power (kW)")
ax.set_xlim(0, 23)
ax.set_xticks(range(0, 24, 2))

# Secondary axis for the battery state of charge, shown as a percentage of the
# pack so the usable window (20% floor to 100% full) is readable at a glance.
ax_soc = ax.twinx()                                        # twinx makes a second y-axis that shares the same x-axis
soc_pct = profile["soc_kwh"] / STATIONARY_BATTERY_KWH * 100
ax_soc.plot(hours, soc_pct, color=SOC_COLOUR, linewidth=1.5, linestyle="--", label="Battery SoC")
ax_soc.set_ylabel("Battery state of charge (%)")
ax_soc.set_ylim(0, 105)
ax_soc.axhline(SOC_max * 100, color=SOC_COLOUR, linewidth=0.8, alpha=0.4)   # full
ax_soc.axhline(SOC_min * 100, color=SOC_COLOUR, linewidth=0.8, alpha=0.4)   # 20% floor
ax_soc.text(0.3, SOC_max * 100 - 4, "full", fontsize=7, color=SOC_COLOUR, alpha=0.7)
ax_soc.text(0.3, SOC_min * 100 + 1, "floor", fontsize=7, color=SOC_COLOUR, alpha=0.7)

if not np.isnan(median_full_hour):
    ax.axvline(median_full_hour, color="grey", linewidth=1.0, linestyle=":")
    ax.text(median_full_hour + 0.2, ax.get_ylim()[1] * 0.9,
            f"battery full ~{median_full_hour:.1f}h", fontsize=8, color="grey")

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_soc.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, framealpha=0.95)
ax.set_title(f"Average {FOCUS_LABEL} day: PV, load, battery and curtailment")
ax.grid(axis="y", alpha=0.3)

fig2.tight_layout()
fig2.savefig(FIGURES_DIR / "fig_sq45_average_day.png", dpi=150)
plt.show()

# --- Figure 2b: average day per season (2x2 grid) ---
# Meteorological seasons by month. Each panel is the average day for that season,
# over all its days (not only full-battery days), matching the figure above.
seasons = [
    ("Winter (DJF)", [12, 1, 2]),
    ("Spring (MAM)", [3, 4, 5]),
    ("Summer (JJA)", [6, 7, 8]),
    ("Autumn (SON)", [9, 10, 11]),
]

# Build each season's hourly profile first, and track the tallest PV so all four
# panels can share one power axis and stay directly comparable.
season_profiles = []
season_max = 0.0
for name, months in seasons:
    sub = df[df["month"].isin(months)]
    prof = sub.groupby("hour").agg(
        pv_kw=("pv_kw", "mean"),
        load_kw=("load_kw", "mean"),
        curtailed_kw=("curtailed_kw", "mean"),
        soc_kwh=("soc_kwh", "mean"),
    )
    season_profiles.append((name, months, prof))
    if len(prof):
        season_max = max(season_max, prof["pv_kw"].max())

fig_season, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
axes = axes.flatten()             # turn the 2x2 grid of axes into a flat list of four to loop over
twin_axes = []                    # keep the state-of-charge axes so we can label the right column

for ax_s, (name, months, prof) in zip(axes, season_profiles):   # zip pairs each panel with its season
    hrs = prof.index
    ax_s.fill_between(hrs, prof["pv_kw"], color=PV_COLOUR, alpha=0.25)
    ax_s.plot(hrs, prof["pv_kw"], color=PV_COLOUR, linewidth=1.3, label="PV production")
    ax_s.plot(hrs, prof["load_kw"], color=LOAD_COLOUR, linewidth=1.3, label="Load")
    ax_s.fill_between(hrs, prof["curtailed_kw"], color=CURT_COLOUR, alpha=0.6, label="Curtailed (headroom)")
    ax_s.set_xlim(0, 23)
    ax_s.set_xticks(range(0, 24, 4))
    ax_s.set_ylim(0, season_max * 1.05)
    ax_s.set_title(name, fontsize=10)
    ax_s.grid(axis="y", alpha=0.3)

    # Battery state of charge on a twin axis in percent, with full and floor lines.
    ax_s2 = ax_s.twinx()
    soc_pct_s = prof["soc_kwh"] / STATIONARY_BATTERY_KWH * 100
    ax_s2.plot(hrs, soc_pct_s, color=SOC_COLOUR, linewidth=1.3, linestyle="--", label="Battery SoC")
    ax_s2.set_ylim(0, 105)
    ax_s2.axhline(SOC_max * 100, color=SOC_COLOUR, linewidth=0.7, alpha=0.35)
    ax_s2.axhline(SOC_min * 100, color=SOC_COLOUR, linewidth=0.7, alpha=0.35)
    twin_axes.append(ax_s2)

    # Median fill time for this season, over the days that reach full.
    season_full = full_times[full_times["month"].isin(months)]["hour_decimal"]
    if len(season_full):
        ax_s.axvline(season_full.median(), color="grey", linewidth=0.9, linestyle=":")

# Outer labels only, to keep the grid clean.
axes[0].set_ylabel("Power (kW)")
axes[2].set_ylabel("Power (kW)")
axes[2].set_xlabel("Hour of day (local)")
axes[3].set_xlabel("Hour of day (local)")
twin_axes[1].set_ylabel("Battery state of charge (%)")
twin_axes[3].set_ylabel("Battery state of charge (%)")

# One shared legend along the bottom.
h1, l1 = axes[0].get_legend_handles_labels()
h2, l2 = twin_axes[0].get_legend_handles_labels()
fig_season.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4, fontsize=9, framealpha=0.95)
fig_season.suptitle("Average day by season: PV, load, battery and curtailment", fontsize=12)
fig_season.tight_layout(rect=[0, 0.05, 1, 0.96])   # leave room for the suptitle and the bottom legend
fig_season.savefig(FIGURES_DIR / "fig_sq45_seasonal_day.png", dpi=150)
plt.show()

# --- Figure 3: mobile-battery capture curve ---
fig3, ax_cap = plt.subplots(figsize=(11, 5))

ax_cap.plot(MOBILE_SIZES, captured_share, color=CURT_COLOUR, linewidth=1.8,
            label="Share of curtailment captured")
ax_cap.set_xlabel("Mobile battery size (kWh)")
ax_cap.set_ylabel("Captured share of annual curtailment (%)")
ax_cap.set_ylim(0, 100)
ax_cap.grid(alpha=0.3)

ax_fill = ax_cap.twinx()
ax_fill.plot(MOBILE_SIZES, focus_full_frac, color=SOC_COLOUR, linewidth=1.8,
             linestyle="--", label=f"{FOCUS_LABEL.capitalize()} days pack fills fully")
ax_fill.set_ylabel(f"{FOCUS_LABEL.capitalize()} days fully charged (%)")
ax_fill.set_ylim(0, 100)

lines1, labels1 = ax_cap.get_legend_handles_labels()
lines2, labels2 = ax_fill.get_legend_handles_labels()
ax_cap.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9, framealpha=0.95)
ax_cap.set_title("Mobile-battery capture curve")

fig3.tight_layout()
fig3.savefig(FIGURES_DIR / "fig_sq45_capture_curve.png", dpi=150)
plt.show()

# --- Figure 4: unserved load per month (reliability cross-check) ---
fig4, ax_un = plt.subplots(figsize=(12, 4))
unmet_vals = [monthly_unmet_kwh.get(m, 0.0) for m in x]
max_unmet = max(unmet_vals)
ax_un.bar(x, unmet_vals, color=UNMET_COLOUR, edgecolor="white", linewidth=0.5)
for xi, v in zip(x, unmet_vals):
    if v > 0.5:
        ax_un.text(xi, v + max(max_unmet * 0.01, 0.05), f"{v:.0f}",
                   ha="center", va="bottom", fontsize=8)
ax_un.set_xticks(x)
ax_un.set_xticklabels(month_labels)
ax_un.set_ylabel("Unserved load (kWh)")
ax_un.set_title(f"Unserved load per month, PV only, {STATIONARY_BATTERY_KWH:.0f} kWh pooled battery")
ax_un.grid(axis="y", alpha=0.3)
if max_unmet < 1:                                          # keep a readable axis when there is essentially no unmet load
    ax_un.set_ylim(0, 1)
    ax_un.text(0.5, 0.85, "no meaningful unserved load in this window",
               transform=ax_un.transAxes, ha="center", fontsize=9, style="italic", color="grey")

fig4.tight_layout()
fig4.savefig(FIGURES_DIR / "fig_sq45_unserved_month.png", dpi=150)
plt.show()


# ============================================================================
#  8. CSV exports
# ============================================================================

monthly_out = pd.DataFrame({
    "month": x,
    "mean_daily_curtailment_kwh": curt_vals,
    "unserved_load_kwh": unmet_vals,
})
monthly_out.to_csv(OUTPUTS_DIR / "sq45_monthly.csv", index=False)

capture_out = pd.DataFrame({
    "mobile_size_kwh": MOBILE_SIZES,
    "captured_kwh_year": captured_kwh,
    "captured_share_pct": captured_share,
    "days_full_pct": focus_full_frac,
})
capture_out.to_csv(OUTPUTS_DIR / "sq45_capture_curve.csv", index=False)

print(f"Saved figures to {FIGURES_DIR}/ and CSVs to {OUTPUTS_DIR}/")