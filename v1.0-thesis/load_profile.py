"""
load_profile.py
Load and visualise Tinos Ecolodge consumption data.

This version reads the extended 2024-2026 export and produces three figures
over the whole record:

  1. Data coverage per meter   (raw gaps, nothing filled)
  2. Monthly energy per meter  (gaps filled by even spreading)
  3. Full hourly per building  (gaps filled by even spreading)

The meters are cumulative kWh. After a data gap the logger reports the gap's
missed energy as a lump over the first hour or two back online, which appears
as a large spike in the hourly difference. The even-spread fill removes both
the gap and that catch-up spike, so the monthly and yearly figures are not
dominated by recovery needles. The coverage figure deliberately stays raw, so
you can still see exactly where readings were missing.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path

from configuration import (MODEL_WINDOW_START, MODEL_WINDOW_END, LOAD_COLUMNS, Timezone,
                           LODGE_NAMES, LODGE_COLOURS)


# --- Config -----------------------------------------------------------

CONSUMPTION_FILE = "data/load/Ecolodge_energy consumption2024-26 now.csv"
FIGURES_DIR = Path("figures")
LOCAL_TZ = Timezone
GAP_THRESHOLD = pd.Timedelta("2h")

# The meters are cumulative. After a data gap the logger releases the gap's
# missed energy over the first hour or two back online (a spike in the hourly
# difference), and the very first reading after a gap is still the pre-catch-up
# value. To remove the spike we blank the gap AND this many hours after it,
# then interpolate the cumulative meter straight through. A straight line on a
# cumulative meter is a constant rate, i.e. the missed energy spread evenly.
# 3 covers the longest catch-up seen in the data (two hours) with a margin.
LUMP_HOURS = 3

METER_LABELS = {
    "bh_energy":           "Big house",
    "sh_energy":           "Small house",
    "uh_energy":           "Utility house",
    "gh_energy":           "Greenhouse",
    "pump_bh_rain":        "Pump (BH rain)",
    "sewage_pumps_energy": "Sewage pumps",
    "main_energy":         "Main (aggregate)",
}

COLOURS = {
    "Big house":        "#d62728",
    "Small house":      "#ff7f0e",
    "Utility house":    "#bcbd22",
    "Greenhouse":       "#2ca02c",
    "Pump (BH rain)":   "#1f77b4",
    "Sewage pumps":     "#9467bd",
    "Main (aggregate)": "#7f7f7f",
}

PLOT_METERS = [
    "Big house", "Small house", "Utility house",
    "Greenhouse", "Pump (BH rain)", "Sewage pumps",
]

# Which consumption columns become nanogrids, taken from configuration.py
# (load_file_1..N). This is what prepare_model_loads selects and tiles to the
# cluster, so adding a load_file there adds a nanogrid here.
MODEL_BUILDINGS = list(LOAD_COLUMNS)


# --- Loader -----------------------------------------------------------

def load_cumulative(path=CONSUMPTION_FILE):
    """Load the raw cumulative meter readings (kWh), wide format,
    one column per meter, on the timestamps present in the file."""
    # Flux export: 3 metadata header rows before the real header
    raw = pd.read_csv(path, skiprows=3, low_memory=False)
    raw.columns = raw.columns.str.strip()

    # Parse forgivingly; junk rows become NaN and get dropped
    raw["t"] = pd.to_datetime(raw["_time"], errors="coerce", utc=True)
    raw["v"] = pd.to_numeric(raw["_value"], errors="coerce")
    df = raw.dropna(subset=["t", "v"])[["t", "_field", "v"]].copy()
    df = df[df["_field"].isin(METER_LABELS.keys())]

    # Long -> wide: one column per meter
    cum = df.pivot_table(
        index="t", columns="_field", values="v", aggfunc="last"
    ).sort_index()
    cum = cum.reindex(columns=list(METER_LABELS.keys()))
    cum = cum.rename(columns=METER_LABELS)
    return cum


def load_consumption(path=CONSUMPTION_FILE):
    # Cumulative kWh -> hourly interval kWh (raw, gaps and spikes left in)
    return load_cumulative(path).diff()


def load_consumption_even(path=CONSUMPTION_FILE, lump_hours=LUMP_HOURS):
    """Hourly kWh with every gap filled by even spreading.

    Works on the cumulative meter. For each gap it blanks the missing hours
    plus the few catch-up hours after them, then interpolates the cumulative
    reading along a straight line from the last value before the gap to the
    first settled value after it. Differencing that straight line gives a
    constant load across the whole span, so the gap energy is spread evenly
    and the post-gap spike disappears. Total energy is preserved, because the
    line still ends on the real settled cumulative reading.
    """
    cum = load_cumulative(path).asfreq("h")        # cumulative kWh, NaN inside gaps

    # The gap pattern is shared across meters, so one column defines it.
    is_gap = cum.iloc[:, 0].isna()

    # Blank the gap itself and the catch-up hours that follow it. shift(k)
    # moves the gap mask k hours forward, so this also marks the hours just
    # after each gap where the lump is released.
    blank = is_gap.copy()
    for k in range(1, lump_hours + 1):
        blank |= is_gap.shift(k, fill_value=False)

    cum_filled = cum.mask(blank).interpolate(method="time")   # straight line across blanks
    return cum_filled.diff()


def clean_consumption(hourly_kWh):
    return hourly_kWh.clip(lower=0).copy()


# --- Diagnostics ------------------------------------------------------

def report_quality(hourly_kWh):
    print("=" * 60)
    print("Data quality report: Tinos consumption data")
    print("=" * 60)

    t_min, t_max = hourly_kWh.index.min(), hourly_kWh.index.max()
    print(f"\nPeriod : {t_min}  ->  {t_max}")
    print(f"Length : {t_max - t_min}")
    print(f"Rows   : {len(hourly_kWh)}")

    dt = hourly_kWh.index.to_series().diff()
    big = dt[dt > GAP_THRESHOLD]
    print(f"\nTimestamp gaps > {GAP_THRESHOLD}: {len(big)}")
    if len(big) > 0:
        print(f"  Largest        : {big.max()}")
        print(f"  Total gap time : {big.sum()}")
        print("  Five largest gaps:")
        for ts, g in big.nlargest(5).items():
            i = hourly_kWh.index.get_loc(ts)
            t_before = hourly_kWh.index[i - 1]
            print(f"    {t_before}  ->  {ts}   ({g})")

    print("\nTotal energy per meter over covered period (kWh):")
    totals = hourly_kWh.clip(lower=0).sum().round(0)
    for label, v in totals.items():
        print(f"  {label:25s}: {v:8.0f}")
    print("=" * 60)


def report_negatives(hourly_kWh):

    print("=" * 60)
    print("Negative-value analysis: Tinos consumption data")
    print("=" * 60)

    # Mark the first reading after each gap; negatives there are
    # expected differencing artifacts.
    gap_after = hourly_kWh.index.to_series().diff() > GAP_THRESHOLD

    for col in hourly_kWh.columns:
        series = hourly_kWh[col]
        is_neg = series < -0.001          # ignore floating-point noise
        n_neg = is_neg.sum()

        if n_neg == 0:
            print(f"  {col:25s}: no negatives")
            continue

        # Split: negatives at a gap edge vs in a clean period.
        at_gap = (is_neg & gap_after).sum()
        in_clean = n_neg - at_gap
        print(f"  {col:25s}: {n_neg:4d} neg  "
              f"({at_gap} at gap, {in_clean} in clean period), "
              f"worst={series.min():.3f}")

    print("=" * 60)


# --- Helpers ----------------------------------------------------------

def to_local(hourly_kWh):
    out = hourly_kWh.copy()
    out.index = out.index.tz_convert(LOCAL_TZ)
    return out


def month_bounds(index):
    """First-of-month before the data starts and first-of-month after it
    ends, in local time. Used to frame the x-axis on whole months."""
    start = index.min().normalize().replace(day=1)
    end = (index.max().normalize().replace(day=1) + pd.offsets.MonthBegin(1))
    return start, end


# --- Figure 1: data coverage map (RAW, nothing filled) ----------------

def plot_coverage_map(cum_grid, save_path):
    """Coverage strip per meter, drawn from the RAW readings (nothing filled).

    cum_grid is the cumulative meter on a regular UTC hourly grid; a meter
    counts as present wherever it has a reading. Each present run is drawn as
    one clean rectangle with broken_barh, so the bands have flat tops and
    bottoms (the old fill_between version left slightly uneven edges).
    """
    # Draw on a UTC grid so daylight-saving changes do not distort spacing,
    # then label the axis in local time (a whole-hour shift, invisible here).
    disp_index = cum_grid.index.tz_convert(LOCAL_TZ).tz_localize(None)
    xnum = mdates.date2num(disp_index)
    hour = 1.0 / 24.0                         # one hour expressed in day units

    fig, ax = plt.subplots(figsize=(13, 4))

    for row, meter in enumerate(PLOT_METERS):
        present = cum_grid[meter].notna().to_numpy().astype(int)
        # Run starts where present goes 0 -> 1, run ends where it goes 1 -> 0.
        change = np.diff(present)
        starts = np.where(change == 1)[0] + 1
        ends = np.where(change == -1)[0] + 1
        if present[0]:                       # a run open at the very start
            starts = np.r_[0, starts]
        if present[-1]:                      # a run still open at the very end
            ends = np.r_[ends, len(present)]
        # One (x_start, width) rectangle per present run; width = run length.
        bars = [(xnum[s], (e - s) * hour) for s, e in zip(starts, ends)]
        ax.broken_barh(bars, (row, 0.8), facecolors=COLOURS[meter])

    ax.set_yticks([r + 0.4 for r in range(len(PLOT_METERS))])
    ax.set_yticklabels(PLOT_METERS, fontsize=9)
    ax.set_ylim(0, len(PLOT_METERS))
    lo, hi = month_bounds(disp_index)
    ax.set_xlim(mdates.date2num(lo), mdates.date2num(hi))
    ax.xaxis_date()                          # x values are dates, format ticks as such
    ax.set_xlabel("Local time")
    ax.set_title(
        "Data coverage per meter, Tinos Ecolodge 2024 to 2026 "
        "(coloured = reading present, blank = missing)"
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def plot_coverage_map_nanogrids(cum_grid, save_path,
                                start="2025-01-01", end="2026-05-31"):
    """Coverage strip for the four model nanogrids only, over the model window.

    Same idea as plot_coverage_map, but restricted to the four buildings the
    model uses (relabelled Lodge 1, Lodge 2, Utility House, Greenhouse) and
    clipped to the January 2025 to May 2026 window, so the figure shows only
    the data that actually feeds the model. The pumps, the sewage meter and the
    earlier 2024 readings are left out. No title: the report caption describes
    the figure.
    """
    # buildings picks the data columns; labels are the matching display names,
    # in the same order (Big house -> Lodge 1, and so on).
    buildings = ["Big house", "Small house", "Utility house", "Greenhouse"]
    labels    = ["Big Lodge", "Small Lodge", "Utility House", "Greenhouse"]

    # Keep only the model window. The grid index is tz-aware UTC and the date
    # strings are read on that same index, so this includes all of 31 May 2026.
    window = cum_grid.loc[start:end]

    # Draw on a UTC grid so daylight-saving changes do not distort spacing,
    # then label the axis in local time (a whole-hour shift, invisible here).
    disp_index = window.index.tz_convert(LOCAL_TZ).tz_localize(None)
    xnum = mdates.date2num(disp_index)
    hour = 1.0 / 24.0                         # one hour expressed in day units

    fig, ax = plt.subplots(figsize=(13, 3))

    for row, meter in enumerate(buildings):
        present = window[meter].notna().to_numpy().astype(int)
        # Run starts where present goes 0 -> 1, run ends where it goes 1 -> 0.
        change = np.diff(present)
        starts = np.where(change == 1)[0] + 1
        ends = np.where(change == -1)[0] + 1
        if present[0]:                       # a run open at the very start
            starts = np.r_[0, starts]
        if present[-1]:                      # a run still open at the very end
            ends = np.r_[ends, len(present)]
        # One (x_start, width) rectangle per present run; width = run length.
        bars = [(xnum[s], (e - s) * hour) for s, e in zip(starts, ends)]
        ax.broken_barh(bars, (row, 0.8), facecolors=COLOURS[meter])

    ax.set_yticks([r + 0.4 for r in range(len(buildings))])
    ax.set_yticklabels(labels, fontsize=9)   # labels line up with buildings above
    ax.set_ylim(0, len(buildings))

    # Frame the x-axis on whole months, exactly Jan 2025 to end of May 2026.
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.offsets.MonthBegin(1)
    ax.set_xlim(mdates.date2num(lo), mdates.date2num(hi))
    ax.xaxis_date()                          # x values are dates, format ticks as such
    ax.set_xlabel("Local time")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")

# --- Figure 2: monthly totals (EVEN-FILLED) ---------------------------

def plot_monthly_totals(hourly_kWh, save_path):
    df = to_local(hourly_kWh)[PLOT_METERS]
    monthly = df.resample("MS").sum(min_count=1)

    x_labels = [t.strftime("%b %Y") for t in monthly.index]
    x_pos = np.arange(len(monthly))

    fig, ax = plt.subplots(figsize=(14, 6))

    bottom = np.zeros(len(monthly))
    for meter in PLOT_METERS:
        vals = monthly[meter].fillna(0).values
        ax.bar(
            x_pos, vals, bottom=bottom,
            label=meter, color=COLOURS[meter],
            edgecolor="white", linewidth=0.5,
        )
        bottom += vals

    for xi, total in zip(x_pos, bottom):
        if total > 0:
            ax.text(xi, total + 3, f"{total:.0f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Energy consumption (kWh)")
    ax.set_title("Monthly energy consumption per meter, "
                 "Tinos Ecolodge 2024 to 2026 (gaps filled by even spreading)")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.text(
        0.99, 0.97,
        "Gaps filled by even spreading; see coverage figure for\n"
        "where readings were missing (filled months are less certain)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, style="italic", color="grey",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")

def plot_monthly_nanogrids(hourly_kWh, save_path,
                           start="2025-01-01", end="2026-05-31"):
    """Monthly stacked consumption for the four model nanogrids only.

    Same style as plot_monthly_totals, but restricted to the four buildings the
    model treats as nanogrids (relabelled Lodge 1, Lodge 2, Utility House,
    Greenhouse) and clipped to the January 2025 to May 2026 span, so it lines up
    with the full-file PV and wind figures. The pumps and sewage meters are left
    out. No title: the report caption describes the figure.
    """
    # buildings picks the data columns; labels are the matching display names,
    # in the same order (Big house -> Lodge 1, and so on).
    buildings = ["Big house", "Small house", "Utility house", "Greenhouse"]
    labels    = ["Big Lodge", "Small Lodge", "Utility House", "Greenhouse"]

    df = to_local(hourly_kWh)[buildings]                 # keep only the four model buildings, in display order
    monthly = df.resample("MS").sum(min_count=1)
    monthly = monthly.loc[start:end]                     # clip to Jan 2025 to May 2026 (no stray June 2026 bar)

    x_labels = [t.strftime("%b %Y") for t in monthly.index]
    x_pos = np.arange(len(monthly))

    fig, ax = plt.subplots(figsize=(14, 6))

    bottom = np.zeros(len(monthly))
    for building, label in zip(buildings, labels):       # walk the two lists together: building for the data, label for the legend
        vals = monthly[building].fillna(0).values
        ax.bar(
            x_pos, vals, bottom=bottom,
            label=label, color=COLOURS[building],        # reuse each building's existing colour
            edgecolor="white", linewidth=0.5,
        )
        bottom += vals

    for xi, total in zip(x_pos, bottom):
        if total > 0:
            ax.text(xi, total + 3, f"{total:.0f}",
                    ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Energy consumption (kWh)")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")

# --- Figure 3: full time series per building (EVEN-FILLED) ------------

def plot_time_series_per_building(hourly_kWh, save_path):
    df = to_local(hourly_kWh)
    buildings = ["Big house", "Small house",
                 "Utility house", "Greenhouse"]

    fig, axes = plt.subplots(
        len(buildings), 1, figsize=(13, 8), sharex=True
    )

    for ax, meter in zip(axes, buildings):
        ax.plot(
            df.index, df[meter],
            color=COLOURS[meter], linewidth=0.6,
        )
        ax.set_ylabel(meter, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(*month_bounds(df.index))

    axes[-1].set_xlabel("Local time")
    fig.suptitle(
        "Full hourly consumption per building, "
        "Tinos Ecolodge 2024 to 2026 (gaps filled by even spreading)"
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")
    
# --- Figure: mean daily load shape per lodge, basic window vs heavy window ---

def plot_daily_load_shapes(save_path,
                           windows=(("Basic load case (2025)",                        "2025-01-01", "2025-12-31"),
                                    ("Heavy load case with A/C (Jun 2025 to May 2026)", "2025-06-01", "2026-05-31"))):
    """Average daily load shape per lodge, one panel per modelling window.

    For each window the measured load is averaged by time of day, giving the mean
    24-hour shape of each lodge. Putting the basic window next to the heavy window
    shows how the air-con reshapes the day, which is the justification for running
    two load cases. Each lodge keeps its name and colour from the LODGES table in
    configuration.py, so the lines match the monthly and time-series figures.
    """
    fig, axes = plt.subplots(1, len(windows), figsize=(13, 5), sharey=True)
    if len(windows) == 1:
        axes = [axes]                                   # make a one-panel figure iterable too

    for ax, (label, start, end) in zip(axes, windows):
        load_kw, time_index = prepare_model_loads(start=start, end=end)   # (n_steps, 4) kW, columns in LOAD_COLUMNS order

        # Average day: tag every 15-min step with its minute-of-day, then mean each slot across all days.
        minute_of_day = time_index.hour * 60 + time_index.minute          # 0..1425
        daily = pd.DataFrame(load_kw, index=minute_of_day).groupby(level=0).mean()
        hours = daily.index / 60.0                                        # x-axis in hours

        for col in range(load_kw.shape[1]):                              # one line per lodge, LOAD_COLUMNS order
            ax.plot(hours, daily.iloc[:, col], linewidth=1.8,
                    color=LODGE_COLOURS[col], label=LODGE_NAMES[col])
        ax.set_title(label)                                             # functional panel title naming the window
        ax.set_xlabel("Hour of day")
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean load (kW)")
    axes[-1].legend()                                                   # one legend serves both panels (same lodges)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


# --- Preparation for the simulation model -----------------------------
# Turns the cleaned hourly building data into a 15-minute load array in the
# shape the cluster model expects: (n_steps, 3) for Big/Small/Utility house,
# optionally windowed to the modelling period set in configuration.py.

def prepare_model_loads(path=CONSUMPTION_FILE,
                        start=MODEL_WINDOW_START,
                        end=MODEL_WINDOW_END):
    """Cleaned hourly building loads -> 15-minute kW array, optionally windowed.

    Returns (load_kw, time_index): load_kw has shape (n_steps, 3) with columns
    in MODEL_BUILDINGS order, time_index is the matching 15-minute stamps used
    to align PV and wind to the same window.

    start and end come from configuration.py as "YYYY-MM-DD" strings, or None.
    With a window set the data is cut to that period; with both None the whole
    record is used (for data that already covers exactly the period you want).

    The even fill is applied to the WHOLE record first, so every gap (including
    the long winter outage inside the window) is bridged using real readings on
    both sides before any cut, leaving no edge gap unfilled.
    """
    even = load_consumption_even(path)                  # UTC hourly kWh, gaps spread evenly
    even = clean_consumption(even)                      # clip negatives to zero
    df = even[MODEL_BUILDINGS]

    # Optional windowing: slice only the side that is given. The bounds are
    # tz-aware UTC to match the index, and the end includes the whole last day.
    if start is not None:
        df = df.loc[pd.Timestamp(start, tz="UTC"):]
    if end is not None:
        df = df.loc[:pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)]
    df = df.dropna()                                    # trim any unfilled edge

    # Hourly kWh == average kW over that hour, so the value carries over
    # directly. Upsample to 15-min by holding each hour's value across its
    # four quarter-hour steps (model stays at 15-min, energy is unchanged).
    idx_15 = pd.date_range(df.index[0], df.index[-1], freq="15min")
    kw_15 = df.reindex(idx_15, method="ffill")
    
    # Express the timeline in Tinos local time so PV, load, plots and binding
    # dates all read in the island clock. This relabels the instants, it does
    # not move them, so the energy values are unchanged.
    kw_15.index = kw_15.index.tz_convert(LOCAL_TZ)

    load_kw = kw_15.to_numpy()                          # shape (n_steps, 3), columns = MODEL_BUILDINGS
    time_index = kw_15.index                            # 15-min timestamps, used to align PV/wind to the same window

    return load_kw, time_index


# --- Entry point ------------------------------------------------------

if __name__ == "__main__":
    FIGURES_DIR.mkdir(exist_ok=True)

    # Raw hourly (gaps and spikes left in) for the quality reports.
    hourly_raw = load_consumption()
    report_quality(hourly_raw)
    report_negatives(hourly_raw)

    # Coverage map uses raw reading presence on a UTC hourly grid.
    cum_grid = load_cumulative().asfreq("h")
    plot_coverage_map(
        cum_grid,
        save_path=FIGURES_DIR / "fig_coverage_map.png",
    )

    # Even-filled series for the monthly and yearly figures: gaps spread
    # evenly, post-gap spikes removed.
    hourly_even = clean_consumption(load_consumption_even())

    plot_monthly_totals(
        hourly_even,
        save_path=FIGURES_DIR / "fig_monthly_totals.png",
    )
    
    plot_monthly_nanogrids(
        hourly_even,
        save_path=FIGURES_DIR / "fig_monthly_nanogrids.png",
    )
    
    plot_time_series_per_building(
        hourly_even,
        save_path=FIGURES_DIR / "fig_time_series_per_building.png",
    )
    
    plot_daily_load_shapes(
    save_path=FIGURES_DIR / "fig_daily_load_shapes.png",
    )
    
    plot_coverage_map_nanogrids(
    cum_grid,
    save_path=FIGURES_DIR / "fig_coverage_map_nanogrids.png",
)