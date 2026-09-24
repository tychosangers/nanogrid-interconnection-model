"""
airco_night_draw.py
Isolate and show the small-house air conditioner in the measured load.

The small house has one meter, so the air conditioner is not recorded on its
own. Its signature is isolated by comparing the same months with and without
it: Feb-May 2026 has the unit, Feb-May 2025 does not, and both sit on
well-covered data away from the long winter gap. The load is read through
tinos_data's even-fill and cleaning, so this is exactly the small-house load
the model replays.

One panel: hourly load in the two matched windows, side by side on a SHARED
y-axis. The shared axis is the point; without it each panel autoscales and the
step change is hidden in the tick labels.

The console summary still reports the day and night means, so the finding that
the unit is a daytime load rather than an overnight one is recorded there even
though the average-day panel is no longer drawn.

Press run. Toggles are at the top.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from load_profile import load_consumption_even, clean_consumption, to_local, COLOURS, CONSUMPTION_FILE


# ============================================================================
#  Toggles
# ============================================================================

BUILDING = "Small house"

# The two matched windows. Same months, one year with the airco, one without.
# Feb-Mar is the safe choice: both years are on well-covered data. Widen to
# 05-31 if you want the full spring run of the unit, but check the coverage
# figure first, because the early-2025 side is partly bridged by the even fill.
BASELINE_LABEL = "Feb-May 2025 (no A/C)"
BASELINE_START = "2025-02-01"
BASELINE_END   = "2025-05-31"

AC_LABEL = "Feb-May 2026 (with A/C)"
AC_START = "2026-02-01"
AC_END   = "2026-05-31"

# Night window used for the day/night averages in the printout [hour of day].
NIGHT_START = 20      # 20:00
NIGHT_END   = 6       # 06:00

FIGURES_DIR = Path("figures")
FIGURE_NAME = "fig_airco_load_signature.png"


# ============================================================================
#  1. Load the small-house load exactly as the model sees it
# ============================================================================

hourly = clean_consumption(load_consumption_even(CONSUMPTION_FILE))   # UTC hourly kWh, gaps evenly filled, negatives clipped
hourly = to_local(hourly)                                             # relabel to Athens local time
sh = hourly[BUILDING]                                                 # one hourly kWh series (= average kW over each hour)


# ============================================================================
#  2. Helpers
# ============================================================================

def day_night_means(series, start, end):
    """Mean load in the day hours and in the night hours over a date range."""
    s = series.loc[start:end]
    # Night wraps midnight, so it is hours >= NIGHT_START OR < NIGHT_END.
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
#  4. Figure: the two matched windows, shared y-axis
# ============================================================================

FIGURES_DIR.mkdir(exist_ok=True)

BUILD_COLOUR = COLOURS.get(BUILDING, "#ff7f0e")
BASE_COLOUR  = "#7f7f7f"

# Two panels side by side, sharing one y-axis. sharey=True does two jobs here:
# it forces both panels onto identical y limits (so the step change is visible
# rather than hidden by each panel autoscaling), and it automatically removes
# the duplicate y tick labels from the right-hand panel.
fig, (ax_base, ax_ac) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)


def plot_window(ax, series, label, colour):
    """One window of hourly load, with its peak annotated."""
    ax.plot(series.index, series.values, color=colour, linewidth=0.8)

    # idxmax gives the timestamp of the highest value, max gives the value.
    t_peak = series.idxmax()
    v_peak = series.max()
    # annotate draws text at a point; xytext offsets the label so it does not
    # sit on top of the spike, and textcoords="offset points" means that offset
    # is measured in screen points rather than data units.
    ax.annotate(
        f"peak {v_peak:.1f} kW",
        xy=(t_peak, v_peak),
        xytext=(6, -2), textcoords="offset points",
        fontsize=8, color=colour, va="top",
    )

    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Local time")
    ax.grid(alpha=0.3)

    # Tick on the 1st and 15th of each month so the window is readable.
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