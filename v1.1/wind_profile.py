# ============================================================================
#  wind_profile.py
#  Wind generation per nanogrid from the Solcast 10 m wind speed, aligned with
#  the load time index. Run this file for a summary and figures of the wind input.
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configuration import *
from pv_profile import (load_solcast_data, capacity_per_grid, clip_to_window,
                        combine_solcast_files, SOLAR_FILES_FULL, SOLCAST_TZ, TINOS_TZ)


def calculate_wind_generation(df):
    """Output of one turbine of WIND_CAPACITY_KW [kW].

    The 10 m wind speed is extrapolated to hub height with the power law, then
    converted with a power curve: zero below cut-in, cubic between cut-in and
    rated speed, rated power up to cut-out, and zero above cut-out.
    Furling of the turbine above rated speed is not modelled, so output above
    rated speed is an upper bound.
    """
    v10 = df['wind_speed_10m'].values
    v = v10 * (WIND_HUB_HEIGHT_M / WIND_DATA_HEIGHT_M) ** WIND_FRICTION_COEFF

    wind_kw = np.zeros_like(v)
    ramp = (v >= WIND_CUT_IN_SPEED) & (v < WIND_RATED_SPEED)
    flat = (v >= WIND_RATED_SPEED) & (v <= WIND_CUT_OUT_SPEED)

    wind_kw[ramp] = WIND_CAPACITY_KW * (v[ramp] ** 3 - WIND_CUT_IN_SPEED ** 3) / (WIND_RATED_SPEED ** 3 - WIND_CUT_IN_SPEED ** 3)
    wind_kw[flat] = WIND_CAPACITY_KW

    return wind_kw


def build_wind_for_window(time_index, n_nanogrids, solar_file, load_kw=None):
    """Wind generation for every nanogrid on the load time index [kW], shape (n_steps, n_nanogrids).

    Timestamps are matched as in build_pv_for_window. With load_kw given, each
    nanogrid's turbine capacity follows ALLOCATION (load shares when ALLOCATION
    is an explicit PV list); without it, every nanogrid gets WIND_CAPACITY_KW.
    """
    df_solar = load_solcast_data(solar_file)
    wind_1d = calculate_wind_generation(df_solar)

    wind_index = pd.DatetimeIndex(pd.to_datetime(df_solar["period_end"]))
    if wind_index.tz is None:
        wind_index = wind_index.tz_localize(SOLCAST_TZ)
    wind_index = wind_index.tz_convert(TINOS_TZ)
    wind_series = pd.Series(wind_1d, index=wind_index)

    load_idx = time_index.tz_convert(TINOS_TZ)

    tol = pd.Timedelta(hours=1)
    if load_idx.min() < wind_series.index.min() - tol or load_idx.max() > wind_series.index.max() + tol:
        raise ValueError(
            f"Wind data covers {wind_series.index.min()} to {wind_series.index.max()}, "
            f"but the load window is {load_idx.min()} to {load_idx.max()}. "
            f"Make sure SOLAR_FILE covers the whole modelling period "
            f"(combine the Solcast files into a list if needed)."
        )

    wind_window = wind_series.reindex(load_idx, method="nearest")
    wind_col = wind_window.to_numpy().reshape(-1, 1)      # output of one WIND_CAPACITY_KW turbine, shape (n_steps, 1)

    if load_kw is None:
        return np.tile(wind_col, (1, n_nanogrids))

    wind_alloc = ALLOCATION if isinstance(ALLOCATION, str) else "load"
    sizes = capacity_per_grid(load_kw, WIND_CAPACITY_KW, wind_alloc)
    scale = sizes / WIND_CAPACITY_KW
    return wind_col * scale


if __name__ == "__main__":
    # Summary and figures of the wind input for the whole installed system.

    WIND_COLOUR = "#4e9fd1"

    df = load_solcast_data(SOLAR_FILE)
    wind_kw = calculate_wind_generation(df)

    system_kW = WIND_CAPACITY_KW * N_NANOGRIDS
    wind_system = wind_kw * N_NANOGRIDS                        # whole-system output [kW]

    # Energy per step [kWh], labelled by the start of each Solcast interval.
    t = pd.DatetimeIndex(df["period_end"]) - pd.Timedelta(minutes=TIME_STEP_MIN)
    energy = pd.Series(wind_system * TIME_STEP_MIN / 60.0, index=t)

    n_days = (t[-1] - t[0]).total_seconds() / 86400.0
    total_kWh = energy.sum()
    annual_kWh = total_kWh / n_days * 365.0
    specific_yield = annual_kWh / system_kW                    # [kWh/kW/year]
    peak_kw = wind_system.max()
    capacity_factor = wind_system.mean() / system_kW
    mean_daily = total_kWh / n_days

    print("=" * 58)
    print("Wind generation check (whole system, full Solcast coverage)")
    print("=" * 58)
    print(f"  Solcast file       : {SOLAR_FILE}")
    print(f"  Coverage           : {t[0].date()} to {t[-1].date()}  ({n_days:.0f} days)")
    print(f"  Installed wind     : {system_kW:6.2f} kW    ({WIND_CAPACITY_KW:.2f} kW x {N_NANOGRIDS} nanogrids)")
    print(f"  Hub height         : {WIND_HUB_HEIGHT_M:.0f} m  (data at {WIND_DATA_HEIGHT_M:.0f} m, shear {WIND_FRICTION_COEFF:.2f})")
    print(f"  Power curve        : cut-in {WIND_CUT_IN_SPEED:.0f}, rated {WIND_RATED_SPEED:.0f}, cut-out {WIND_CUT_OUT_SPEED:.0f} m/s")
    print("  " + "-" * 44)
    print(f"  Total generation   : {total_kWh:8.0f} kWh   (over the coverage)")
    print(f"  Equivalent annual  : {annual_kWh:8.0f} kWh/year")
    print(f"  Specific yield     : {specific_yield:8.0f} kWh/kW/year")
    print(f"  Mean daily         : {mean_daily:8.1f} kWh/day")
    print(f"  Peak output        : {peak_kw:8.2f} kW")
    print(f"  Capacity factor    : {capacity_factor:8.1%}")
    print("=" * 58)

    # Daily generation over the modelling window
    daily = clip_to_window(energy.resample("D").sum())

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(daily.index, daily.values, color=WIND_COLOUR, alpha=0.55)
    ax.plot(daily.index, daily.values, color=WIND_COLOUR, linewidth=0.8)
    ax.set_ylabel("Wind generation (kWh/day)")
    ax.set_xlabel("Date")
    ax.set_title(f"Daily wind generation, {MODEL_WINDOW_START} to {MODEL_WINDOW_END} ({system_kW:.1f} kW system)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.set_xlim(daily.index[0], daily.index[-1])
    plt.tight_layout()
    plt.show()

    # Monthly generation over the modelling window
    monthly = clip_to_window(energy.resample("MS").sum())

    labels = [d.strftime("%b %Y") for d in monthly.index]
    x = np.arange(len(monthly))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, monthly.values, color=WIND_COLOUR, edgecolor="white", linewidth=0.5)
    headroom = monthly.values.max() * 0.01
    for xi, v in zip(x, monthly.values):
        ax.text(xi, v + headroom, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Wind generation (kWh)")
    ax.set_title(f"Monthly wind generation, modelling window ({system_kW:.1f} kW system)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Both Solcast files combined, 1 January 2025 to 31 May 2026
    df_full = combine_solcast_files(SOLAR_FILES_FULL)
    wind_full = calculate_wind_generation(df_full) * N_NANOGRIDS
    t_full = pd.DatetimeIndex(df_full["period_end"]) - pd.Timedelta(minutes=TIME_STEP_MIN)
    energy_full = pd.Series(wind_full * TIME_STEP_MIN / 60.0, index=t_full)

    # Daily generation over both files
    daily_full = energy_full.resample("D").sum()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(daily_full.index, daily_full.values, color=WIND_COLOUR, alpha=0.55)
    ax.plot(daily_full.index, daily_full.values, color=WIND_COLOUR, linewidth=0.8)
    ax.set_ylabel("Wind generation (kWh/day)")
    ax.set_xlabel("Date")
    ax.set_title(f"Daily wind generation, full Solcast file ({system_kW:.1f} kW system)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.set_xlim(daily_full.index[0], daily_full.index[-1] + pd.Timedelta(days=1))
    plt.tight_layout()
    plt.show()

    # Monthly generation over both files (ends on 31 May 2026)
    monthly_full = energy_full.loc[:"2026-05-31"].resample("MS").sum()
    labels = [d.strftime("%b %Y") for d in monthly_full.index]
    x = np.arange(len(monthly_full))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, monthly_full.values, color=WIND_COLOUR, edgecolor="white", linewidth=0.5)
    headroom = monthly_full.values.max() * 0.01
    for xi, v in zip(x, monthly_full.values):
        ax.text(xi, v + headroom, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Wind generation (kWh)")
    ax.set_title(f"Monthly wind generation, full Solcast file ({system_kW:.1f} kW system)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
