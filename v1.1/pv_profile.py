# ============================================================================
#  pv_profile.py
#  PV generation per nanogrid from Solcast irradiance data, aligned with the
#  load time index. Run this file for a summary and figures of the PV input.
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configuration import *
import datetime as dt

SOLCAST_TZ = dt.timezone(dt.timedelta(hours=SOLCAST_UTC_OFFSET_H))
TINOS_TZ   = TIMEZONE


def load_solcast_data(filepath):
    """Read one Solcast CSV, or a list of CSVs combined into one series.

    Duplicate timestamps from overlapping files are dropped.
    """
    paths = [filepath] if isinstance(filepath, str) else list(filepath)
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df['period_end'] = pd.to_datetime(df['period_end'])
    df = df.drop_duplicates(subset='period_end', keep='first')
    df = df.sort_values('period_end').reset_index(drop=True)

    return df


def calculate_pv_generation(df):
    """PV output of one array of PV_CAPACITY_KW [kW].

    Output scales linearly with global horizontal irradiance (GHI) relative to
    1000 W/m², times the performance ratio PV_PR. All losses, and the absence
    of a tilt and orientation transposition, are included in PV_PR.
    """
    ghi = df['ghi'].values
    pv_kw = (ghi / 1000) * PV_CAPACITY_KW * PV_PR

    return pv_kw


def capacity_per_grid(load_kw, avg_capacity_kW, allocation="load"):
    """Split the cluster capacity of a generator over the nanogrids [kW per nanogrid].

    Used for PV and wind. The cluster total is avg_capacity_kW * n_grids.
    allocation is "load" (proportional to annual consumption), "equal", or a
    list of capacities in LOAD_COLUMNS order that must sum to the cluster total.
    """
    n_grids = load_kw.shape[1]
    total   = avg_capacity_kW * n_grids

    if not isinstance(allocation, str):
        sizes = np.asarray(allocation, dtype=float)
        if sizes.shape != (n_grids,):
            raise ValueError(f"ALLOCATION list has {sizes.size} value(s) but there are {n_grids} nanogrids")
        if not np.isclose(sizes.sum(), total):
            raise ValueError(
                f"ALLOCATION list sums to {sizes.sum():.3f} kW but the costed total is "
                f"{total:.3f} kW (average {avg_capacity_kW:.4f} x {n_grids} nanogrids). "
                f"Set the average to {sizes.sum() / n_grids:.4f} so the two agree."
            )
        return sizes

    if allocation == "equal":
        return np.full(n_grids, avg_capacity_kW)

    if allocation == "load":
        annual = load_kw.sum(axis=0)
        return total * annual / annual.sum()

    raise ValueError(f"ALLOCATION must be 'load', 'equal', or a list of sizes, not {allocation!r}")


def build_pv_for_window(time_index, n_nanogrids, solar_file, load_kw=None):
    """PV generation for every nanogrid on the load time index [kW], shape (n_steps, n_nanogrids).

    Solcast timestamps (UTC+1) and the load timestamps are both converted to
    local time before matching; each load step takes the nearest PV value.
    With load_kw given, each nanogrid's capacity follows ALLOCATION; without
    it, every nanogrid gets PV_CAPACITY_KW.
    """
    df_solar = load_solcast_data(solar_file)
    pv_1d = calculate_pv_generation(df_solar)

    pv_index = pd.DatetimeIndex(pd.to_datetime(df_solar["period_end"]))
    if pv_index.tz is None:
        pv_index = pv_index.tz_localize(SOLCAST_TZ)
    pv_index = pv_index.tz_convert(TINOS_TZ)
    pv_series = pd.Series(pv_1d, index=pv_index)

    load_idx = time_index.tz_convert(TINOS_TZ)

    tol = pd.Timedelta(hours=1)
    if load_idx.min() < pv_series.index.min() - tol or load_idx.max() > pv_series.index.max() + tol:
        raise ValueError(
            f"Solar data covers {pv_series.index.min()} to {pv_series.index.max()}, "
            f"but the load window is {load_idx.min()} to {load_idx.max()}. "
            f"Make sure SOLAR_FILE covers the whole modelling period "
            f"(combine the Solcast files into a list if needed)."
        )

    pv_window = pv_series.reindex(load_idx, method="nearest")
    pv_col = pv_window.to_numpy().reshape(-1, 1)          # output of one PV_CAPACITY_KW array, shape (n_steps, 1)

    if load_kw is None:
        return np.tile(pv_col, (1, n_nanogrids))

    # Scale the average array to each nanogrid's allocated capacity.
    sizes = capacity_per_grid(load_kw, PV_CAPACITY_KW, ALLOCATION)
    scale = sizes / PV_CAPACITY_KW
    return pv_col * scale


def clip_to_window(series):
    """Keep only the modelling window (MODEL_WINDOW_START to MODEL_WINDOW_END)."""
    win_start = pd.Timestamp(MODEL_WINDOW_START)
    win_end   = pd.Timestamp(MODEL_WINDOW_END)
    if series.index.tz is not None:
        win_start = win_start.tz_localize(series.index.tz)
        win_end   = win_end.tz_localize(series.index.tz)
    return series.loc[win_start:win_end]


SOLAR_FILES_FULL = [
    "data/solar/Solar_Data_TinosEcoLodge_2025_Solcast.csv",
    "data/solar/Solar_Wind_Data_TinosEcoLodge_2025_2026_Solcast.csv",
]


def combine_solcast_files(paths):
    """Combine Solcast files with naive and timezone-aware timestamps into one DataFrame.

    All timestamps are put on the Solcast clock without timezone; duplicates
    from overlapping files are dropped.
    """
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        ts = pd.to_datetime(d["period_end"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(SOLCAST_TZ).dt.tz_localize(None)
        d["period_end"] = ts
        frames.append(d)
    df_full = pd.concat(frames, ignore_index=True)
    return df_full.drop_duplicates(subset="period_end", keep="first").sort_values("period_end").reset_index(drop=True)


if __name__ == "__main__":
    # Summary and figures of the PV input for the whole installed system.

    PV_COLOUR = "#f5a800"

    df = load_solcast_data(SOLAR_FILE)
    pv_kw = calculate_pv_generation(df)

    system_kWp = PV_CAPACITY_KW * N_NANOGRIDS
    pv_system = pv_kw * N_NANOGRIDS                             # whole-system output [kW]

    # Energy per step [kWh], labelled by the start of each Solcast interval.
    t = pd.DatetimeIndex(df["period_end"]) - pd.Timedelta(minutes=TIME_STEP_MIN)
    energy = pd.Series(pv_system * TIME_STEP_MIN / 60.0, index=t)

    n_days = (t[-1] - t[0]).total_seconds() / 86400.0
    total_kWh = energy.sum()
    annual_kWh = total_kWh / n_days * 365.0
    specific_yield = annual_kWh / system_kWp                    # [kWh/kWp/year]
    peak_kw = pv_system.max()
    capacity_factor = pv_system.mean() / system_kWp
    mean_daily = total_kWh / n_days

    print("=" * 58)
    print("PV generation check (whole system, full Solcast coverage)")
    print("=" * 58)
    print(f"  Solcast file       : {SOLAR_FILE}")
    print(f"  Coverage           : {t[0].date()} to {t[-1].date()}  ({n_days:.0f} days)")
    print(f"  Installed PV       : {system_kWp:6.2f} kWp   ({PV_CAPACITY_KW:.2f} kWp x {N_NANOGRIDS} nanogrids)")
    print(f"  Performance ratio  : {PV_PR:6.2f}")
    print("  " + "-" * 44)
    print(f"  Total production   : {total_kWh:8.0f} kWh   (over the coverage)")
    print(f"  Equivalent annual  : {annual_kWh:8.0f} kWh/year")
    print(f"  Specific yield     : {specific_yield:8.0f} kWh/kWp/year")
    print(f"  Mean daily         : {mean_daily:8.1f} kWh/day")
    print(f"  Peak output        : {peak_kw:8.2f} kW")
    print(f"  Capacity factor    : {capacity_factor:8.1%}")
    print("=" * 58)

    # Daily production over the modelling window
    daily = clip_to_window(energy.resample("D").sum())

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(daily.index, daily.values, color=PV_COLOUR, alpha=0.55)
    ax.plot(daily.index, daily.values, color=PV_COLOUR, linewidth=0.8)
    ax.set_ylabel("PV production (kWh/day)")
    ax.set_xlabel("Date")
    ax.set_title(f"Daily PV production, {MODEL_WINDOW_START} to {MODEL_WINDOW_END} ({system_kWp:.1f} kWp system)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.set_xlim(daily.index[0], daily.index[-1])
    plt.tight_layout()
    plt.show()

    # Monthly production over the modelling window
    monthly = clip_to_window(energy.resample("MS").sum())

    labels = [d.strftime("%b %Y") for d in monthly.index]
    x = np.arange(len(monthly))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, monthly.values, color=PV_COLOUR, edgecolor="white", linewidth=0.5)
    headroom = monthly.values.max() * 0.01
    for xi, v in zip(x, monthly.values):
        ax.text(xi, v + headroom, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("PV production (kWh)")
    ax.set_title(f"Monthly PV production, modelling window ({system_kWp:.1f} kWp system)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Both Solcast files combined, 1 January 2025 to 31 May 2026
    df_full = combine_solcast_files(SOLAR_FILES_FULL)

    pv_full = calculate_pv_generation(df_full) * N_NANOGRIDS
    t_full = pd.DatetimeIndex(df_full["period_end"]) - pd.Timedelta(minutes=TIME_STEP_MIN)
    energy_full = pd.Series(pv_full * TIME_STEP_MIN / 60.0, index=t_full)

    # Daily production over both files
    daily_full = energy_full.resample("D").sum()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(daily_full.index, daily_full.values, color=PV_COLOUR, alpha=0.55)
    ax.plot(daily_full.index, daily_full.values, color=PV_COLOUR, linewidth=0.8)
    ax.set_ylabel("PV production (kWh/day)")
    ax.set_xlabel("Date")
    ax.set_title(f"Daily PV production, full Solcast file ({system_kWp:.1f} kWp system)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.set_xlim(daily_full.index[0], daily_full.index[-1] + pd.Timedelta(days=1))
    plt.tight_layout()
    plt.show()

    # Monthly production over both files (ends on 31 May 2026)
    monthly_full = energy_full.loc[:"2026-05-31"].resample("MS").sum()
    labels = [d.strftime("%b %Y") for d in monthly_full.index]
    x = np.arange(len(monthly_full))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, monthly_full.values, color=PV_COLOUR, edgecolor="white", linewidth=0.5)
    headroom = monthly_full.values.max() * 0.01
    for xi, v in zip(x, monthly_full.values):
        ax.text(xi, v + headroom, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("PV production (kWh)")
    ax.set_title(f"Monthly PV production, full Solcast file ({system_kWp:.1f} kWp system)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
