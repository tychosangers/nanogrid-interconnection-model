# ============================================================================
#  airco_visualisation.py
#  Shows the air-conditioning load of the Small Lodge by comparing the same
#  months with and without the unit: February to May 2025 (no A/C) and
#  February to May 2026 (with A/C), on a shared y-axis. Prints day and night
#  mean loads for both periods. Uses the gap-filled load data of the model.
# ============================================================================

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from configuration import NANOGRIDS
from load_profile import load_consumption_even, clean_consumption, to_local, CONSUMPTION_FILE


# ============================================================================
#  Settings
# ============================================================================

BUILDING = "Small house"

# Two periods with the same months, without and with the A/C unit.
BASELINE_LABEL = "Feb-May 2025 (no A/C)"
BASELINE_START = "2025-02-01"
BASELINE_END   = "2025-05-31"

AC_LABEL = "Feb-May 2026 (with A/C)"
AC_START = "2026-02-01"
AC_END   = "2026-05-31"

# Night window used for the day/night averages in the printout [hour of day].
NIGHT_START = 20      # 20:00
NIGHT_END   = 6       # 06:00

FIGURES_DIR = Path("outputs")
FIGURE_NAME = "fig_airco_load_signature.png"


# ============================================================================
#  1. Load data
# ============================================================================

hourly = clean_consumption(load_consumption_even(CONSUMPTION_FILE))
hourly = to_local(hourly)                                             # relabel to Athens local time
sh = hourly[BUILDING]


# ============================================================================
#  2. Helpers
# ============================================================================

def day_night_means(series, start, end):
    """Mean load in the day hours and in the night hours over a date range."""
    s = series.loc[start:end]
    # Night wraps midnight, so it is hours >= NIGHT_START or < NIGHT_END.
    is_night = (s.index.hour >= NIGHT_START) | (s.index.hour < NIGHT_END)
    return s[~is_night].mean(), s[is_night].mean()


base_series = sh.loc[BASELINE_START:BASELINE_END]     # hourly load, no airco
ac_series   = sh.loc[AC_START:AC_END]                 # hourly load, with airco

base_day, base_night = day_night_means(sh, BASELINE_START, BASELINE_END)
ac_day,   ac_night   = day_night_means(sh, AC_START, AC_END)


# ============================================================================
#  3. Printed summary
# ============================================================================

print("=" * 58)
print(f"Small-house load signature of the air conditioner  ({BUILDING})")
print("=" * 58)
print(f"  Night window        : {NIGHT_START:02d}:00 to {NIGHT_END:02d}:00")
print("  " + "-" * 52)
print(f"  {BASELINE_LABEL}")
print(f"    peak              : {base_series.max():6.2f} kW")
print(f"    total energy      : {base_series.sum():6.0f} kWh")
print(f"    day mean          : {base_day:6.3f} kW")
print(f"    night mean        : {base_night:6.3f} kW")
print(f"  {AC_LABEL}")
print(f"    peak              : {ac_series.max():6.2f} kW")
print(f"    total energy      : {ac_series.sum():6.0f} kWh")
print(f"    day mean          : {ac_day:6.3f} kW")
print(f"    night mean        : {ac_night:6.3f} kW")
print("  " + "-" * 52)
print(f"  Peak grows by       : x{ac_series.max() / base_series.max():5.1f}")
print(f"  Energy grows by     : x{ac_series.sum() / base_series.sum():5.1f}")
print(f"  A/C adds by day     : {ac_day - base_day:+6.3f} kW")
print(f"  A/C adds by night   : {ac_night - base_night:+6.3f} kW")
print("=" * 58)


# ============================================================================
#  4. Figure
# ============================================================================

FIGURES_DIR.mkdir(exist_ok=True)

BUILD_COLOUR = NANOGRIDS[BUILDING]["colour"]
BASE_COLOUR  = "#7f7f7f"

# Two panels with a shared y-axis, so the difference in load level is visible.
fig, (ax_base, ax_ac) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)


def plot_window(ax, series, label, colour):
    """One window of hourly load, with its peak annotated."""
    ax.plot(series.index, series.values, color=colour, linewidth=0.8)

    t_peak = series.idxmax()
    v_peak = series.max()
    ax.annotate(
        f"peak {v_peak:.1f} kW",
        xy=(t_peak, v_peak),
        xytext=(6, -2), textcoords="offset points",
        fontsize=8, color=colour, va="top",
    )

    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Local time")
    ax.grid(alpha=0.3)

    # Ticks on the 1st and 15th of each month
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=[1, 15]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlim(series.index[0], series.index[-1])


plot_window(ax_base, base_series, BASELINE_LABEL, BASE_COLOUR)
plot_window(ax_ac,   ac_series,   AC_LABEL,       BUILD_COLOUR)

ax_base.set_ylabel("Load (kW)")
ax_base.set_ylim(bottom=0)

fig.tight_layout()
fig.savefig(FIGURES_DIR / FIGURE_NAME, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure to {FIGURES_DIR}/{FIGURE_NAME}")
