import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configuration import *
import datetime as dt

SOLCAST_TZ = dt.timezone(dt.timedelta(hours=SOLCAST_UTC_OFFSET_H))              # from configuration
TINOS_TZ   = Timezone

def load_solcast_data(filepath):                                                # Function to make dataframe of csv (one path, or several stitched)
    """Read one PV generation CSV, or a list of them stitched into one series.
    Possible to pass a list of paths to combine downloads that cover different date ranges
    (for example calendar 2025 plus the mid-2025 to mid-2026 year), so a single
    df covers every window you might model and switching window needs no data
    swap. Overlapping timestamps are dropped so the join is seamless.
    """
    paths = [filepath] if isinstance(filepath, str) else list(filepath)
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)          # stack the files
    df['period_end'] = pd.to_datetime(df['period_end'])
    df = df.drop_duplicates(subset='period_end', keep='first')                  # remove overlap between files
    df = df.sort_values('period_end').reset_index(drop=True)

    return df

def calculate_pv_generation(df):                                                # Function to get PV generation from GHI and PR
    """ Core PV model. Scale the global horizontal irradiance (GHI) linearly
     against the standard-test-condition irradiance of 1000 W/m², then
     multiply by the installed capacity and a flat performance ratio. At
     1000 W/m² the array produces PV_capacity_kW * PV_PR; at half that
     irradiance, half as much. Every real-world loss (cell temperature,
     wiring, soiling, inverter conversion and low-light response) is folded
     into the single PV_PR term rather than modelled separately. GHI is used
     directly, so no tilt or orientation transposition is applied; that
     simplification also sits inside PR."""
    ghi = df['ghi'].values
    pv_kw = (ghi / 1000) * PV_capacity_kW * PV_PR                               # kW per nanogrid array

    return pv_kw

def capacity_per_grid(load_kw, avg_capacity_kW, allocation="load"):
    """Split a generator's cluster total across the nanogrids [capacity per grid].

    Shared by PV and wind. avg_capacity_kW is the average size per nanogrid, so
    the cluster total is avg_capacity_kW * n_grids (for PV the real 4.2 kWp over
    four lodges; for wind the four quarter-turbines). allocation decides the split:

        "load"  -> proportional to each lodge's share of annual consumption, so a
                   bigger lodge hosts more capacity. This is the default.
        "equal" -> the same capacity on every lodge (the original SQ1 assumption).
        a list  -> an explicit capacity per lodge in kW, in LOAD_COLUMNS order,
                   for example [1.6, 0.8, 0.9, 0.9] for real installed sizes.

    "load" and "equal" keep the cluster total at exactly avg_capacity_kW * n_grids,
    so the total capex and O&M (costed from that total in optimiser.py) do not
    move; only the per-lodge generation shape changes. An explicit list must sum
    to that same total, so the cost side stays consistent with the sizes used
    here; if it does not, we stop with a message telling you which average to set.
    """
    n_grids = load_kw.shape[1]
    total   = avg_capacity_kW * n_grids                 # cluster total the cost side assumes

    # Explicit per-lodge sizes: one value per lodge, used as given.
    if not isinstance(allocation, str):                 # a list/array was passed instead of "load"/"equal"
        sizes = np.asarray(allocation, dtype=float)     # turn the list into a numeric array
        if sizes.shape != (n_grids,):
            raise ValueError(f"ALLOCATION list has {sizes.size} value(s) but there are {n_grids} nanogrids")
        if not np.isclose(sizes.sum(), total):          # keep generation and cost in step
            raise ValueError(
                f"ALLOCATION list sums to {sizes.sum():.3f} kW but the costed total is "
                f"{total:.3f} kW (average {avg_capacity_kW:.4f} x {n_grids} nanogrids). "
                f"Set the average to {sizes.sum() / n_grids:.4f} so the two agree."
            )
        return sizes

    if allocation == "equal":
        return np.full(n_grids, avg_capacity_kW)        # the average on every lodge

    if allocation == "load":
        annual = load_kw.sum(axis=0)                    # per-lodge consumption over the window
        return total * annual / annual.sum()            # kW per lodge, summing back to total

    raise ValueError(f"ALLOCATION must be 'load', 'equal', or a list of sizes, not {allocation!r}")


def build_pv_for_window(time_index, n_households, solar_file, load_kw=None):
    """Slice the Solcast PV series to the load window, then tile to all nanogrids.

    PV and load are matched on the real instant, not the wall-clock label. The
    Solcast stamps are on a UTC+1 clock and the load arrives tz-aware, so both are
    put on the Tinos clock before matching. Without this, each load step would be
    paired with PV from a different hour.

    Pass the load array as load_kw to size each nanogrid's PV by the ALLOCATION
    rule in configuration.py (load-proportional by default). Leave it None to give
    every nanogrid the same PV_capacity_kW array (the original equal split, handy
    for quick standalone checks).
    """
    df_solar = load_solcast_data(solar_file)
    pv_1d = calculate_pv_generation(df_solar)

    pv_index = pd.DatetimeIndex(pd.to_datetime(df_solar["period_end"]))
    if pv_index.tz is None:
        pv_index = pv_index.tz_localize(SOLCAST_TZ)   # tell pandas these stamps are UTC+1
    pv_index = pv_index.tz_convert(TINOS_TZ)          # restate on the Tinos clock
    pv_series = pd.Series(pv_1d, index=pv_index)

    load_idx = time_index.tz_convert(TINOS_TZ)        # load is tz-aware; put it on the same clock

    # Guard: the solar data must cover the load window (compared on the real instant).
    tol = pd.Timedelta(hours=1)
    if load_idx.min() < pv_series.index.min() - tol or load_idx.max() > pv_series.index.max() + tol:
        raise ValueError(
            f"Solar data covers {pv_series.index.min()} to {pv_series.index.max()}, "
            f"but the load window is {load_idx.min()} to {load_idx.max()}. "
            f"Make sure SOLAR_FILE covers the whole modelling period "
            f"(combine the Solcast files into a list if needed)."
        )

    # Put PV onto the load's timestamps: each load step takes the PV reading
    # closest in time (method="nearest"). This is the actual PV-to-load match,
    # not a reorder, so the 15-min PV series lines up with the 15-min load series.
    pv_window = pv_series.reindex(load_idx, method="nearest")
    pv_col = pv_window.to_numpy().reshape(-1, 1)          # (n_steps, 1): output of one PV_capacity_kW array

    if load_kw is None:
        return np.tile(pv_col, (1, n_households))         # every lodge identical (original behaviour)

    # Give each lodge its own size. calculate_pv_generation already used the
    # average PV_capacity_kW, so we scale each column by size / average. That
    # ratio is exactly 1.0 for the equal split, so this reproduces the tile above.
    sizes = capacity_per_grid(load_kw, PV_capacity_kW, ALLOCATION)   # kWp per lodge
    scale = sizes / PV_capacity_kW                                   # one multiplier per lodge
    return pv_col * scale                                            # (n_steps,1) * (n,) broadcasts to (n_steps,n)


if __name__ == "__main__":
    # Standalone check of the PV input over the WHOLE Solcast file (not the
    # modelling window). Shows the real system array (PV_capacity_kW per
    # nanogrid, tiled to N_nanogrids = the full kWp on site).

    PV_COLOUR = "#f5a800"

    df = load_solcast_data(SOLAR_FILE)
    pv_kw = calculate_pv_generation(df)                         # kW per nanogrid array

    system_kWp = PV_capacity_kW * N_nanogrids                   # full installed PV on site
    pv_system = pv_kw * N_nanogrids                             # whole-system output [kW]

    # kWh per 15-min step. Solcast stamps each interval by its END, so we shift
    # the index back one step to label each interval by its START. That puts the
    # energy in the month and day it actually occurred (and drops the midnight
    # boundary reading that would otherwise show as a stray next-month sliver).
    t = pd.DatetimeIndex(df["period_end"]) - pd.Timedelta(minutes=Time_step_min)
    energy = pd.Series(pv_system * Time_step_min / 60.0, index=t)

    # --- Key values (whole system) ---
    n_days = (t[-1] - t[0]).total_seconds() / 86400.0
    total_kWh = energy.sum()
    annual_kWh = total_kWh / n_days * 365.0                     # scaled to a full year
    specific_yield = annual_kWh / system_kWp                    # kWh per kWp per year (capacity-independent)
    peak_kw = pv_system.max()
    capacity_factor = pv_system.mean() / system_kWp
    mean_daily = total_kWh / n_days

    print("=" * 58)
    print("PV generation check (whole system, full Solcast coverage)")
    print("=" * 58)
    print(f"  Solcast file       : {SOLAR_FILE}")
    print(f"  Coverage           : {t[0].date()} to {t[-1].date()}  ({n_days:.0f} days)")
    print(f"  Installed PV       : {system_kWp:6.2f} kWp   ({PV_capacity_kW:.2f} kWp x {N_nanogrids} nanogrids)")
    print(f"  Performance ratio  : {PV_PR:6.2f}")
    print("  " + "-" * 44)
    print(f"  Total production   : {total_kWh:8.0f} kWh   (over the coverage)")
    print(f"  Equivalent annual  : {annual_kWh:8.0f} kWh/year")
    print(f"  Specific yield     : {specific_yield:8.0f} kWh/kWp/year")
    print(f"  Mean daily         : {mean_daily:8.1f} kWh/day")
    print(f"  Peak output        : {peak_kw:8.2f} kW")
    print(f"  Capacity factor    : {capacity_factor:8.1%}")
    print("=" * 58)

    # --- Figure 1: daily production over the modelling window ---
    # Clip to the same window the model runs on (MODEL_WINDOW_* come from
    # configuration.py, so this follows USE_2025_2026_WINDOW automatically)
    # instead of the whole file, so the figure matches the modelled period.
    daily = energy.resample("D").sum()
    win_start = pd.Timestamp(MODEL_WINDOW_START)            # window bounds as timestamps
    win_end   = pd.Timestamp(MODEL_WINDOW_END)
    if daily.index.tz is not None:                          # give the bounds the same timezone as the data, if it has one
        win_start = win_start.tz_localize(daily.index.tz)
        win_end   = win_end.tz_localize(daily.index.tz)
    daily = daily.loc[win_start:win_end]                    # keep only the modelling window

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(daily.index, daily.values, color=PV_COLOUR, alpha=0.55)
    ax.plot(daily.index, daily.values, color=PV_COLOUR, linewidth=0.8)
    ax.set_ylabel("PV production (kWh/day)")
    ax.set_xlabel("Date")
    ax.set_title(f"Daily PV production, 01-01-2025 to 31-12-2025 ({system_kWp:.1f} kWp system)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)                                   # y-axis starts exactly at 0: no gap under the fill
    ax.set_xlim(daily.index[0], daily.index[-1])           # x-axis flush to the first and last day: no side whitespace
    plt.tight_layout()
    plt.show()

    # --- Figure 2: monthly production over the modelling window ---
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
    
    # --- Full-range series for Figures 3 and 4 ---
    # Stitch the calendar-2025 file (Jan to Apr 2025) onto the 2025-2026 file
    # (May 2025 onward) so these figures span 01-01-2025 to 31-05-2026. The two
    # files stamp period_end differently (one naive, one tz-aware), so parse each
    # on its own and put both on the same Solcast UTC+1 clock before combining,
    # otherwise the join fails and the overlapping months do not deduplicate.
    # Overlapping May to Dec 2025 rows are dropped, keeping the first file's copy.
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

    pv_full = calculate_pv_generation(df_full) * N_nanogrids       # whole-system output [kW]
    t_full = pd.DatetimeIndex(df_full["period_end"]) - pd.Timedelta(minutes=Time_step_min)
    energy_full = pd.Series(pv_full * Time_step_min / 60.0, index=t_full)
    
    # --- Figure 3: daily production over the whole Solcast file (no window clip) ---
    # Shows the full span of the loaded file: 01-01-2025 to 31-05-2026 for the
    # combined 2025-2026 file, or all of 2025 for the calendar-2025 file.
    daily_full = energy_full.resample("D").sum()      # Figure 3
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(daily_full.index, daily_full.values, color=PV_COLOUR, alpha=0.55)
    ax.plot(daily_full.index, daily_full.values, color=PV_COLOUR, linewidth=0.8)
    ax.set_ylabel("PV production (kWh/day)")
    ax.set_xlabel("Date")
    ax.set_title(f"Daily PV production, full Solcast file ({system_kWp:.1f} kWp system)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)                                             # y-axis flush to 0
    ax.set_xlim(daily_full.index[0], daily_full.index[-1] + pd.Timedelta(days=1))   # close on the day after the last day
    plt.tight_layout()
    plt.show()

    # --- Figure 4: monthly production over the whole Solcast file (no window clip) ---
    monthly_full = energy_full.loc[:"2026-05-31"].resample("MS").sum()   # drop any 01-06 boundary stamp so no empty June bar appears
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