import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configuration import *
from pv_profile import load_solcast_data, capacity_per_grid   # shared Solcast loader and capacity split (wind lives in the same data file as PV)
import datetime as dt

SOLCAST_TZ = dt.timezone(dt.timedelta(hours=SOLCAST_UTC_OFFSET_H))              # from configuration
TINOS_TZ   = Timezone


def calculate_wind_generation(df):                       # Wind power [kW] from the 10 m wind speed in the Solcast data
    """Core wind model. The turbine output is built in two steps: correct the
    measured 10 m wind speed up to hub height, then map hub-height speed onto a
    linearised power curve scaled to the rated power. Aerodynamic and electrical
    losses are left implicit in the rated power rather than modelled separately."""
    v10 = df['wind_speed_10m'].values                    # measured wind speed at 10 m [m/s]

    # Step 1: lift the measured speed to hub height. Wind speeds up with height, so a turbine whose hub is
    # above the 10 m sensor sees more wind. Power law: v_hub = v10 * (hub_height / data_height) ** friction.
    v = v10 * (Wind_hub_height_m / Wind_data_height_m) ** Wind_friction_coeff

    # Step 2: turn hub-height wind speed into power with a simple power curve, scaled to the rated power:
    #   below cut-in        -> 0      (too little wind to turn the rotor)
    #   cut-in  to rated     -> rises with the cube of wind speed (the energy in wind grows as v^3)
    #   rated   to cut-out   -> flat at rated power (the turbine caps its output)
    #   above cut-out        -> 0      (it shuts down to protect itself in a storm)
    wind_kw = np.zeros_like(v)                            # start every timestep at zero

    ramp = (v >= Wind_cut_in_speed) & (v < Wind_rated_speed)     # timesteps on the rising part of the curve
    flat = (v >= Wind_rated_speed) & (v <= Wind_cut_out_speed)   # timesteps at rated power

    wind_kw[ramp] = Wind_capacity_kW * (v[ramp] ** 3 - Wind_cut_in_speed ** 3) / (Wind_rated_speed ** 3 - Wind_cut_in_speed ** 3)
    wind_kw[flat] = Wind_capacity_kW
    # everything else (below cut-in or above cut-out) stays at the zero we started with

    # CAVEAT: the real Bergey XL1 furls (turns out of the wind) above its rated
    # speed and sheds power, so holding output flat at rated from rated to cut-out
    # is an upper bound; modelled generation above roughly 11 to 13 m/s overstates
    # the true turbine output. This is flagged as a limitation, not corrected here.

    return wind_kw


def build_wind_for_window(time_index, n_households, solar_file, load_kw=None):   # Wind aligned to the load's 15-min timestamps
    """Slice the wind series to the load window, then size it per nanogrid.

    Wind and load are matched on the real instant, not the wall-clock label, in
    exactly the same way as PV. The Solcast stamps are on a UTC+1 clock and the
    load arrives tz-aware, so both are put on the Tinos clock before matching.
    Without this, each load step would be paired with wind from a different hour.

    Pass the load array as load_kw to size each nanogrid's turbine by the same
    ALLOCATION rule as PV (load-proportional by default). Leave it None to give
    every nanogrid the same Wind_capacity_kW turbine (the original equal split,
    handy for quick standalone checks).
    """
    df_solar = load_solcast_data(solar_file)
    wind_1d = calculate_wind_generation(df_solar)

    wind_index = pd.DatetimeIndex(pd.to_datetime(df_solar["period_end"]))
    if wind_index.tz is None:
        wind_index = wind_index.tz_localize(SOLCAST_TZ)   # tell pandas these stamps are UTC+1
    wind_index = wind_index.tz_convert(TINOS_TZ)          # restate on the Tinos clock
    wind_series = pd.Series(wind_1d, index=wind_index)

    load_idx = time_index.tz_convert(TINOS_TZ)            # load is tz-aware; put it on the same clock

    # Guard: the wind data must cover the load window (compared on the real instant).
    tol = pd.Timedelta(hours=1)
    if load_idx.min() < wind_series.index.min() - tol or load_idx.max() > wind_series.index.max() + tol:
        raise ValueError(
            f"Wind data covers {wind_series.index.min()} to {wind_series.index.max()}, "
            f"but the load window is {load_idx.min()} to {load_idx.max()}. "
            f"Make sure SOLAR_FILE covers the whole modelling period "
            f"(combine the Solcast files into a list if needed)."
        )

    # Put wind onto the load's timestamps: each load step takes the wind reading
    # closest in time (method="nearest"). This is the actual wind-to-load match,
    # not a reorder, so the 15-min wind series lines up with the 15-min load series.
    wind_window = wind_series.reindex(load_idx, method="nearest")
    wind_col = wind_window.to_numpy().reshape(-1, 1)               # (n_steps, 1): output of one Wind_capacity_kW turbine

    if load_kw is None:
        return np.tile(wind_col, (1, n_households))               # same turbine at every nanogrid (original behaviour)

    # Wind follows the same ALLOCATION rule as PV. An explicit PV kW list is not a
    # wind rating list, so if ALLOCATION is a list we fall back to load-share for
    # wind (the sensible default); "load" and "equal" pass straight through.
    wind_alloc = ALLOCATION if isinstance(ALLOCATION, str) else "load"
    sizes = capacity_per_grid(load_kw, Wind_capacity_kW, wind_alloc)   # kW per lodge
    scale = sizes / Wind_capacity_kW                                   # one multiplier per lodge
    return wind_col * scale                                            # (n_steps,1) * (n,) broadcasts to (n_steps,n)


if __name__ == "__main__":
    # Standalone check of the wind input over the WHOLE Solcast file (not the
    # modelling window). Shows the real system turbine (Wind_capacity_kW per
    # nanogrid, tiled to N_nanogrids = the full rated power on site).

    WIND_COLOUR = "#4e9fd1"

    df = load_solcast_data(SOLAR_FILE)
    wind_kw = calculate_wind_generation(df)                    # kW per nanogrid turbine

    system_kW = Wind_capacity_kW * N_nanogrids                 # full installed wind on site
    wind_system = wind_kw * N_nanogrids                        # whole-system output [kW]

    # kWh per 15-min step. Solcast stamps each interval by its END, so we shift
    # the index back one step to label each interval by its START (see pv_profile).
    t = pd.DatetimeIndex(df["period_end"]) - pd.Timedelta(minutes=Time_step_min)
    energy = pd.Series(wind_system * Time_step_min / 60.0, index=t)

    # --- Key values (whole system) ---
    n_days = (t[-1] - t[0]).total_seconds() / 86400.0
    total_kWh = energy.sum()
    annual_kWh = total_kWh / n_days * 365.0                    # scaled to a full year
    specific_yield = annual_kWh / system_kW                    # kWh per kW per year (capacity-independent)
    peak_kw = wind_system.max()
    capacity_factor = wind_system.mean() / system_kW
    mean_daily = total_kWh / n_days

    print("=" * 58)
    print("Wind generation check (whole system, full Solcast coverage)")
    print("=" * 58)
    print(f"  Solcast file       : {SOLAR_FILE}")
    print(f"  Coverage           : {t[0].date()} to {t[-1].date()}  ({n_days:.0f} days)")
    print(f"  Installed wind     : {system_kW:6.2f} kW    ({Wind_capacity_kW:.2f} kW x {N_nanogrids} nanogrids)")
    print(f"  Hub height         : {Wind_hub_height_m:.0f} m  (data at {Wind_data_height_m:.0f} m, shear {Wind_friction_coeff:.2f})")
    print(f"  Power curve        : cut-in {Wind_cut_in_speed:.0f}, rated {Wind_rated_speed:.0f}, cut-out {Wind_cut_out_speed:.0f} m/s")
    print("  " + "-" * 44)
    print(f"  Total generation   : {total_kWh:8.0f} kWh   (over the coverage)")
    print(f"  Equivalent annual  : {annual_kWh:8.0f} kWh/year")
    print(f"  Specific yield     : {specific_yield:8.0f} kWh/kW/year")
    print(f"  Mean daily         : {mean_daily:8.1f} kWh/day")
    print(f"  Peak output        : {peak_kw:8.2f} kW")
    print(f"  Capacity factor    : {capacity_factor:8.1%}")
    print("=" * 58)

    # --- Figure 1: daily generation over the modelling window ---
    # Clip to the same window the model runs on (MODEL_WINDOW_* from
    # configuration.py, so this follows USE_2025_2026_WINDOW automatically).
    daily = energy.resample("D").sum()
    win_start = pd.Timestamp(MODEL_WINDOW_START)           # window bounds as timestamps
    win_end   = pd.Timestamp(MODEL_WINDOW_END)
    if daily.index.tz is not None:                         # match the data's timezone if it has one
        win_start = win_start.tz_localize(daily.index.tz)
        win_end   = win_end.tz_localize(daily.index.tz)
    daily = daily.loc[win_start:win_end]                   # keep only the modelling window

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(daily.index, daily.values, color=WIND_COLOUR, alpha=0.55)
    ax.plot(daily.index, daily.values, color=WIND_COLOUR, linewidth=0.8)
    ax.set_ylabel("Wind generation (kWh/day)")
    ax.set_xlabel("Date")
    ax.set_title(f"Daily wind generation 2025 ({system_kW:.1f} kW system)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)                                  # y-axis starts exactly at 0: no gap under the fill
    ax.set_xlim(daily.index[0], daily.index[-1])           # x-axis flush to the first and last day: no side whitespace
    plt.tight_layout()
    plt.show()

    # --- Figure 2: monthly generation over the modelling window ---
    monthly = energy.resample("MS").sum()
    win_start = pd.Timestamp(MODEL_WINDOW_START)           # window bounds as timestamps
    win_end   = pd.Timestamp(MODEL_WINDOW_END)
    if monthly.index.tz is not None:                       # match the data's timezone if it has one
        win_start = win_start.tz_localize(monthly.index.tz)
        win_end   = win_end.tz_localize(monthly.index.tz)
    monthly = monthly.loc[win_start:win_end]               # keep only the modelling window

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

    # --- Full-range series for Figures 3 and 4 ---
    # Stitch the calendar-2025 file (Jan to Apr 2025) onto the 2025-2026 file
    # (May 2025 onward) so these figures span 01-01-2025 to 31-05-2026. The two
    # files stamp period_end differently (one naive, one tz-aware), so parse each
    # on its own and put both on the same Solcast UTC+1 clock before combining,
    # else the join fails and the overlapping months do not deduplicate. Overlapping
    # May to Dec 2025 rows are dropped, keeping the first file's copy.
    SOLAR_FILES_FULL = [
        "data/solar/Solar_Data_TinosEcoLodge_2025_Solcast.csv",
        "data/solar/Solar_Wind_Data_TinosEcoLodge_2025_2026_Solcast.csv",
    ]
    frames = []
    for p in SOLAR_FILES_FULL:
        d = pd.read_csv(p)
        ts = pd.to_datetime(d["period_end"])                       # naive in one file, tz-aware in the other
        if ts.dt.tz is not None:                                   # tz-aware file: move it onto the Solcast UTC+1 clock, then drop the tz
            ts = ts.dt.tz_convert(SOLCAST_TZ).dt.tz_localize(None)
        d["period_end"] = ts                                       # both files now naive on the same clock, so they line up
        frames.append(d)
    df_full = pd.concat(frames, ignore_index=True)
    df_full = df_full.drop_duplicates(subset="period_end", keep="first").sort_values("period_end").reset_index(drop=True)

    wind_full = calculate_wind_generation(df_full) * N_nanogrids   # whole-system output [kW]
    t_full = pd.DatetimeIndex(df_full["period_end"]) - pd.Timedelta(minutes=Time_step_min)
    energy_full = pd.Series(wind_full * Time_step_min / 60.0, index=t_full)

    # --- Figure 3: daily generation over the whole Solcast file (no window clip) ---
    daily_full = energy_full.resample("D").sum()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(daily_full.index, daily_full.values, color=WIND_COLOUR, alpha=0.55)
    ax.plot(daily_full.index, daily_full.values, color=WIND_COLOUR, linewidth=0.8)
    ax.set_ylabel("Wind generation (kWh/day)")
    ax.set_xlabel("Date")
    ax.set_title(f"Daily wind generation, full Solcast file ({system_kW:.1f} kW system)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)                                             # y-axis flush to 0
    ax.set_xlim(daily_full.index[0], daily_full.index[-1] + pd.Timedelta(days=1))   # close on the day after the last day
    plt.tight_layout()
    plt.show()

    # --- Figure 4: monthly generation over the whole Solcast file (no window clip) ---
    monthly_full = energy_full.loc[:"2026-05-31"].resample("MS").sum()   # drop any 01-06 boundary stamp so no empty June bar appears
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