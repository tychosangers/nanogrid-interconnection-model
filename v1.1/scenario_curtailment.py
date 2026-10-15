# ============================================================================
#  scenario_curtailment.py
#  Curtailed PV of the interconnected cluster at the installed 14 kWh battery,
#  and how often that surplus could fully charge a mobile battery pack.
#  The cluster is pooled as in run_interconnected and dispatched once with
#  PV only. Each day a mobile pack starts empty and absorbs at most that day's
#  curtailment, so the result is an upper bound. Unserved load is reported as
#  a reliability check. Uses the modelling window set in configuration.py.
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from configuration import TIME_STEP_MIN, SOC_MAX, MODEL_WINDOW_START, MODEL_WINDOW_END, SOLAR_FILE
from load_profile import prepare_model_loads
from pv_profile import build_pv_for_window
from operation import pool_cluster_residual
from simulation import run_simulation


# ============================================================================
#  Settings
# ============================================================================

STATIONARY_BATTERY_KWH = 14.0


# Months used for the battery fill time and fill fraction in the summary
FOCUS_MONTHS = [6, 7, 8]
FOCUS_LABEL  = "summer"                # short word shown in the summary

SUMMER_MONTHS = [6, 7, 8]
WINTER_MONTHS = [12, 1, 2]

MOBILE_SIZES = np.arange(0.0, 20.5, 0.5)
EXAMPLE_SIZES = [2.0, 5.0, 10.0]
SUMMER_EXAMPLE = 17.0                       # size marked on the summer line [kWh]

FIGURES_DIR = Path("outputs")
OUTPUTS_DIR = Path("outputs")


# ============================================================================
#  1. Run the pooled PV-only dispatch once at the fixed battery
# ============================================================================

hours_per_step = TIME_STEP_MIN / 60.0                        # 0.25 h per 15-min step

loads, time_index = prepare_model_loads(start=MODEL_WINDOW_START, end=MODEL_WINDOW_END)   # (n_steps, n_nanogrids)
n_nanogrids = loads.shape[1]
pv = build_pv_for_window(time_index, n_nanogrids, SOLAR_FILE, loads)                # (n_steps, 4), aligned to the load window

pooled_pv, pooled_load = pool_cluster_residual(pv, loads)    # collapse the cluster to one residual nanogrid
sim = run_simulation(pooled_pv, pooled_load, STATIONARY_BATTERY_KWH, use_diesel=False)   # PV only, no diesel

# Whole-site PV and load
total_pv   = pv.sum(axis=1)
total_load = loads.sum(axis=1)


# ============================================================================
#  2. Tidy dataframe on the time index
# ============================================================================

# One row per 15-minute step; energies per step in kWh
df = pd.DataFrame({
    "pv_kw":         total_pv,
    "load_kw":       total_load,
    "curtailed_kw":  sim["curtailed"],
    "unmet_kw":      sim["unmet_kw"],
    "soc_kwh":       sim["soc_kWh"],
}, index=time_index)

df["curtailed_kwh"] = df["curtailed_kw"] * hours_per_step
df["unmet_kwh"]     = df["unmet_kw"] * hours_per_step

# Date, month and hour in local time
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

# Daily curtailment with its month
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

soc_full_kwh = SOC_MAX * STATIONARY_BATTERY_KWH             # top of the usable window [kWh]

# True where the battery is full
df["is_full"] = df["soc_kwh"] >= soc_full_kwh - 0.01

# Decimal hour of day (11:15 is 11.25)
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
focus_daily_vals = focus_daily.values

captured_kwh   = []
captured_share = []                                        # that energy as a share of all curtailment
focus_full_frac = []

for c in MOBILE_SIZES:
    # Per day the smaller of the curtailment and the pack size, summed over the year
    captured = np.minimum(all_daily, c).sum()
    captured_kwh.append(captured)
    captured_share.append(captured / annual_curt_kwh * 100 if annual_curt_kwh else 0.0)

    if c == 0:
        focus_full_frac.append(0.0)
    else:
        # Share of days with curtailment of at least the pack size
        focus_full_frac.append((focus_daily_vals >= c).mean() * 100)

captured_kwh   = np.array(captured_kwh)
captured_share = np.array(captured_share)
focus_full_frac = np.array(focus_full_frac)

# Share of days a pack of each size is filled completely: whole year and summer
full_year_days_full = np.array([(all_daily >= c).mean() * 100 for c in MOBILE_SIZES])
summer_days_full    = np.array([(summer_daily.values >= c).mean() * 100 for c in MOBILE_SIZES])


# ============================================================================
#  6. Printed summary
# ============================================================================

print("=" * 62)
print("Curtailment and mobile battery headroom (PV only, interconnected cluster)")
print("=" * 62)
print(f"  Window              : {MODEL_WINDOW_START} to {MODEL_WINDOW_END}")
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
print("  Mobile pack: days it could be filled for free (whole year | summer):")
for c in EXAMPLE_SIZES:
    yr  = (all_daily >= c).mean() * 100
    smr = (summer_daily.values >= c).mean() * 100
    print(f"    {c:5.1f} kWh pack : {yr:3.0f}% of days  |  {smr:3.0f}% of summer days")
smr17 = (summer_daily.values >= SUMMER_EXAMPLE).mean() * 100
print(f"    {SUMMER_EXAMPLE:5.1f} kWh pack : summer {smr17:3.0f}% of days")
print("=" * 62)


# ============================================================================
#  7. Figures
# ============================================================================

FIGURES_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

CURT_COLOUR = "#c0392b"

FULLYEAR_COLOUR = "#4c72b0"   # blue, whole-year capture line
SUMMER_COLOUR   = "#f5a800"   # gold, summer capture line

# --- Daily curtailment over the window and mean per month ---
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
fig1.savefig(FIGURES_DIR / "fig_curtailment_year.png", dpi=150)
plt.show()


# --- Share of days a mobile pack is fully charged from curtailed PV ---
fig_cap, ax_cap = plt.subplots(figsize=(8, 5))
ax_cap.plot(MOBILE_SIZES, full_year_days_full, color=FULLYEAR_COLOUR, lw=1.8, label="Full year")
ax_cap.plot(MOBILE_SIZES, summer_days_full,    color=SUMMER_COLOUR,   lw=1.8, label="Summer (Jun-Aug)")

# Example pack sizes on the full-year line
for c in EXAMPLE_SIZES:
    y = (all_daily >= c).mean() * 100
    ax_cap.plot(c, y, "o", color=FULLYEAR_COLOUR, ms=5)
    ax_cap.annotate(f"{c:.0f} kWh: {y:.0f}%", (c, y),
                    textcoords="offset points", xytext=(-8, -10),
                    ha="right", va="top", fontsize=8, color=FULLYEAR_COLOUR)

# Example pack size on the summer line
y_summer = (summer_daily.values >= SUMMER_EXAMPLE).mean() * 100
ax_cap.plot(SUMMER_EXAMPLE, y_summer, "o", color=SUMMER_COLOUR, ms=5)
ax_cap.annotate(f"{SUMMER_EXAMPLE:.0f} kWh: {y_summer:.0f}%", (SUMMER_EXAMPLE, y_summer),
                textcoords="offset points", xytext=(6, -14),
                fontsize=8, color=SUMMER_COLOUR)

ax_cap.set_xlabel("Mobile battery size (kWh)")
ax_cap.set_ylabel("Days the pack fills fully for free (%)")
ax_cap.set_xlim(0, MOBILE_SIZES.max())
ax_cap.set_ylim(0, 102)
ax_cap.grid(alpha=0.3)
ax_cap.legend(frameon=False, loc="lower left")

fig_cap.tight_layout()
fig_cap.savefig(FIGURES_DIR / "fig_days_fully_charged.png", dpi=150)
plt.show()

unmet_vals = [monthly_unmet_kwh.get(m, 0.0) for m in x]


# ============================================================================
#  8. CSV exports
# ============================================================================

monthly_out = pd.DataFrame({
    "month": x,
    "mean_daily_curtailment_kwh": curt_vals,
    "unserved_load_kwh": unmet_vals,
})
monthly_out.to_csv(OUTPUTS_DIR / "curtailment_monthly.csv", index=False)

capture_out = pd.DataFrame({
    "mobile_size_kwh": MOBILE_SIZES,
    "captured_kwh_year": captured_kwh,
    "captured_share_pct": captured_share,
    "full_year_days_full_pct": full_year_days_full,
    "summer_days_full_pct": summer_days_full,
})
capture_out.to_csv(OUTPUTS_DIR / "curtailment_capture_curve.csv", index=False)

print(f"Saved figures to {FIGURES_DIR}/ and CSVs to {OUTPUTS_DIR}/")
