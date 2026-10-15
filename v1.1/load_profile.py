# ============================================================================
#  load_profile.py
#  Reads the cumulative meter export of the Tinos Ecolodge, fills measurement
#  gaps and prepares the 15-minute load array of the nanogrids for the model.
#  Run this file for data-quality reports and the load input figures.
#
#  After a logging gap, the meters release the missed energy as a lump in the
#  first hours back online. The even fill spreads that energy over the whole
#  gap, which preserves the total and removes the spike.
# ============================================================================

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path

from configuration import (MODEL_WINDOW_START, MODEL_WINDOW_END, LOAD_COLUMNS, TIMEZONE,
                           NANOGRID_NAMES, NANOGRID_COLOURS, NANOGRIDS, save_figure)


# --- Settings ----------------------------------------------------------

CONSUMPTION_FILE = "data/load/Ecolodge_energy consumption2024-26 now.csv"
FIGURES_DIR = Path("outputs")
LOCAL_TZ = TIMEZONE
GAP_THRESHOLD = pd.Timedelta("2h")

# Hours after each gap that are blanked before interpolation. The longest
# catch-up in the data is two hours.
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


# Data columns used as nanogrids (LOAD_COLUMNS in configuration.py).
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

    # Long to wide format: one column per meter
    cum = df.pivot_table(
        index="t", columns="_field", values="v", aggfunc="last"
    ).sort_index()
    cum = cum.reindex(columns=list(METER_LABELS.keys()))
    cum = cum.rename(columns=METER_LABELS)
    return cum


def load_consumption(path=CONSUMPTION_FILE):
    """Hourly energy per meter [kWh], differenced from the cumulative readings (gaps and spikes left in)."""
    return load_cumulative(path).diff()


def load_consumption_even(path=CONSUMPTION_FILE, lump_hours=LUMP_HOURS):
    """Hourly energy per meter [kWh] with gaps filled by even spreading.

    The gap and the catch-up hours after it are blanked and the cumulative
    reading is interpolated linearly across them. This spreads the missed
    energy evenly and preserves the total.
    """
    cum = load_cumulative(path).asfreq("h")        # cumulative kWh, NaN inside gaps

    # The gap pattern is shared across meters, so one column defines it.
    is_gap = cum.iloc[:, 0].isna()

    # Blank the gap and the catch-up hours after it.
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

    # Negatives at the first reading after a gap are differencing artefacts.
    gap_after = hourly_kWh.index.to_series().diff() > GAP_THRESHOLD

    for col in hourly_kWh.columns:
        series = hourly_kWh[col]
        is_neg = series < -0.001          # ignore floating-point noise
        n_neg = is_neg.sum()

        if n_neg == 0:
            print(f"  {col:25s}: no negatives")
            continue

        # Negatives at a gap edge versus in a clean period
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


# --- Figures ----------------------------------------------------------

def plot_coverage_map_nanogrids(cum_grid, save_path,
                                start="2025-01-01", end="2026-05-31"):
    """Data coverage per nanogrid meter, January 2025 to May 2026, from the raw readings.

    Each period with readings is drawn as one bar; gaps stay white.
    """
    buildings = LOAD_COLUMNS
    labels    = NANOGRID_NAMES

    # Keep January 2025 to May 2026 (index in UTC).
    window = cum_grid.loc[start:end]

    # Drawn on a UTC grid, so daylight-saving changes do not distort the spacing.
    disp_index = window.index.tz_convert(LOCAL_TZ).tz_localize(None)
    xnum = mdates.date2num(disp_index)
    hour = 1.0 / 24.0                         # one hour expressed in day units

    fig, ax = plt.subplots(figsize=(13, 3))

    for row, meter in enumerate(buildings):
        present = window[meter].notna().to_numpy().astype(int)
        # Start and end of each period with readings
        change = np.diff(present)
        starts = np.where(change == 1)[0] + 1
        ends = np.where(change == -1)[0] + 1
        if present[0]:                       # a run open at the very start
            starts = np.r_[0, starts]
        if present[-1]:                      # a run still open at the very end
            ends = np.r_[ends, len(present)]
        # One (x_start, width) bar per period
        bars = [(xnum[s], (e - s) * hour) for s, e in zip(starts, ends)]
        ax.broken_barh(bars, (row, 0.8), facecolors=NANOGRIDS[meter]["colour"])

    ax.set_yticks([r + 0.4 for r in range(len(buildings))])
    ax.set_yticklabels(labels, fontsize=9)   # labels line up with buildings above
    ax.set_ylim(0, len(buildings))

    # x-axis on whole months
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.offsets.MonthBegin(1)
    ax.set_xlim(mdates.date2num(lo), mdates.date2num(hi))
    ax.xaxis_date()                          # x values are dates, format ticks as such
    ax.set_xlabel("Local time")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def plot_monthly_nanogrids(hourly_kWh, save_path,
                           start="2025-01-01", end="2026-05-31"):
    """Monthly consumption per nanogrid, stacked, January 2025 to May 2026 (gaps filled)."""
    buildings = LOAD_COLUMNS
    labels    = NANOGRID_NAMES

    df = to_local(hourly_kWh)[buildings]
    monthly = df.resample("MS").sum(min_count=1)
    monthly = monthly.loc[start:end]

    x_labels = [t.strftime("%b %Y") for t in monthly.index]
    x_pos = np.arange(len(monthly))

    fig, ax = plt.subplots(figsize=(14, 6))

    bottom = np.zeros(len(monthly))
    for building, label in zip(buildings, labels):
        vals = monthly[building].fillna(0).values
        ax.bar(
            x_pos, vals, bottom=bottom,
            label=label, color=NANOGRIDS[building]["colour"],
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
    ax.set_ylim(0, bottom.max() * 1.10)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure(fig, save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")

    


# --- Model input ------------------------------------------------------

def prepare_model_loads(path=CONSUMPTION_FILE,
                        start=MODEL_WINDOW_START,
                        end=MODEL_WINDOW_END):
    """Load array of the nanogrids for the model.

    Returns load_kw [kW], shape (n_steps, n_nanogrids) in MODEL_BUILDINGS
    order, and the matching 15-minute time index in local time. start and end
    are "YYYY-MM-DD" strings; None uses the whole record. Gaps are filled on
    the whole record before the window is cut.
    """
    even = load_consumption_even(path)                  # UTC hourly kWh, gaps spread evenly
    even = clean_consumption(even)                      # clip negatives to zero
    df = even[MODEL_BUILDINGS]

    # Cut to the window (UTC bounds; the end date is included).
    if start is not None:
        df = df.loc[pd.Timestamp(start, tz="UTC"):]
    if end is not None:
        df = df.loc[:pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)]
    df = df.dropna()                                    # trim any unfilled edge

    # Hourly kWh equals the mean kW of that hour; hold it over the four 15-minute steps.
    idx_15 = pd.date_range(df.index[0], df.index[-1], freq="15min")
    kw_15 = df.reindex(idx_15, method="ffill")
    
    # Express the time index in local time.
    kw_15.index = kw_15.index.tz_convert(LOCAL_TZ)

    load_kw = kw_15.to_numpy()                          # shape (n_steps, 3), columns = MODEL_BUILDINGS
    time_index = kw_15.index

    return load_kw, time_index


# --- Entry point ------------------------------------------------------

if __name__ == "__main__":
    FIGURES_DIR.mkdir(exist_ok=True)

    # Data-quality reports on the raw hourly data
    hourly_raw = load_consumption()
    report_quality(hourly_raw)
    report_negatives(hourly_raw)

    # Coverage figure from the raw readings
    cum_grid = load_cumulative().asfreq("h")
    plot_coverage_map_nanogrids(cum_grid, save_path=FIGURES_DIR / "fig_coverage_map_nanogrids.png")

    # Monthly figure from the gap-filled data
    hourly_even = clean_consumption(load_consumption_even())
    plot_monthly_nanogrids(hourly_even, save_path=FIGURES_DIR / "fig_monthly_nanogrids.png")
