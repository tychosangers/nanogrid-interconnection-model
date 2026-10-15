# ============================================================================
#  pv_validation.py
#  Compares modelled PV generation (Solcast GHI, installed capacity and PR)
#  with the measured PV generation of the Tinos Ecolodge, on overlapping
#  timestamps only. Includes two checks: the ratio of measured to modelled
#  energy per battery state-of-charge band (curtailment), and the summer and
#  winter average day (time zone handling).
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configuration import SOLAR_FILE, PV_PR, TIME_STEP_MIN, TIMEZONE, PV_CAPACITY_KW, N_NANOGRIDS
from pv_profile import load_solcast_data, SOLCAST_TZ


# --- Settings ---
MEASURED_PV_FILE = "data/load/Ecolodge_PV_energy.csv"
PV_KWP           = PV_CAPACITY_KW * N_NANOGRIDS     # installed capacity [kWp]

SUMMER_MONTHS   = [6, 7, 8]
WINTER_MONTHS   = [12, 1, 2]


def load_field(path, field):
    # InfluxDB export: three annotation lines before the header row
    df = pd.read_csv(path, skiprows=3)
    df = df[df["_field"] == field].copy()
    df["_time"]  = pd.to_datetime(df["_time"], utc=True)   # measured stamps are UTC
    df["_value"] = pd.to_numeric(df["_value"])
    s = df.set_index("_time")["_value"].sort_index()
    return s[~s.index.duplicated(keep="first")]            # guard against repeated stamps


def build_modelled_pv():
    df_solar = load_solcast_data(SOLAR_FILE)
    ghi = df_solar["ghi"].to_numpy()
    pv_kw    = (ghi / 1000.0) * PV_KWP * PV_PR     # modelled output at 4.2 kWp and PR
    ideal_kw = (ghi / 1000.0) * PV_KWP             # PR = 1 reference, for the empirical-PR backout

    # Solcast timestamps (UTC+1, interval end) to interval start in UTC
    t = pd.DatetimeIndex(pd.to_datetime(df_solar["period_end"]))
    t = t.tz_localize(SOLCAST_TZ) if t.tz is None else t.tz_convert(SOLCAST_TZ)
    t = (t - pd.Timedelta(minutes=TIME_STEP_MIN)).tz_convert("UTC")

    step_h = TIME_STEP_MIN / 60.0
    model_kwh = pd.Series(pv_kw    * step_h, index=t).resample("h").sum()   # 15-min energy: hourly kWh
    ideal_kwh = pd.Series(ideal_kw * step_h, index=t).resample("h").sum()
    return model_kwh, ideal_kwh


def run_validation():
    measured_kwh = (load_field(MEASURED_PV_FILE, "pv_power") / 1000.0).resample("h").mean()   # W: kW, kWh/hour
    soc          = load_field(MEASURED_PV_FILE, "battery_soc").resample("h").mean()
    if soc.max() <= 1.5:
        soc = soc * 100.0
    model_kwh, ideal_kwh = build_modelled_pv()

    # Overlapping timestamps only, then local time
    df = pd.concat([measured_kwh.rename("measured"),
                    model_kwh.rename("model"),
                    ideal_kwh.rename("ideal"),
                    soc.rename("soc")], axis=1, join="inner").dropna()
    df.index = df.index.tz_convert(TIMEZONE)

    m_tot, p_tot, i_tot = df["measured"].sum(), df["model"].sum(), df["ideal"].sum()
    pct    = (p_tot - m_tot) / m_tot * 100
    pr_eff = m_tot / i_tot
    err    = df["model"] - df["measured"]
    rmse   = np.sqrt((err ** 2).mean())

    print("=" * 70)
    print("  PV validation: modelled (Solcast x PR) vs measured (pv_power)")
    print("=" * 70)
    print(f"  Overlapping coverage : {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}"
          f"  ({len(df)} h, {len(df) / 24:.0f} days)  [Tinos local time]")
    print(f"  Assumed capacity     : {PV_KWP:.2f} kWp     assumed PR: {PV_PR:.2f}")
    print("  " + "-" * 50)
    print(f"  Measured total       : {m_tot:8.0f} kWh")
    print(f"  Modelled total       : {p_tot:8.0f} kWh   ({pct:+.1f} % vs measured)")
    print(f"  Measured yield       : {m_tot / PV_KWP:8.0f} kWh/kWp  (over the coverage)")
    print(f"  Modelled yield       : {p_tot / PV_KWP:8.0f} kWh/kWp")
    print("  " + "-" * 50)
    print(f"  Empirical PR (data)  : {pr_eff:8.2f}   vs assumed {PV_PR:.2f}")
    print(f"  Hourly RMSE          : {rmse:8.3f} kWh")
    print("=" * 70)

    plot_main(df)
    plot_diagnostics(df)
    return df


def plot_main(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    monthly = df.resample("MS")[["measured", "model"]].sum()
    x = np.arange(len(monthly))
    axes[0].bar(x - 0.2, monthly["measured"], width=0.4, label="Measured",  color="tab:green")
    axes[0].bar(x + 0.2, monthly["model"],    width=0.4, label="Modelled",  color="tab:orange")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([d.strftime("%b") for d in monthly.index])
    axes[0].set_ylabel("PV energy (kWh)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    daily = df.resample("D")[["measured", "model"]].sum()
    mx = max(daily["measured"].max(), daily["model"].max())
    axes[1].scatter(daily["measured"], daily["model"], s=18, alpha=0.5, color="tab:blue")
    axes[1].plot([0, mx], [0, mx], color="black", linestyle="--", linewidth=1, label="1:1")
    slope, intercept = np.polyfit(daily["measured"], daily["model"], 1)
    axes[1].plot([0, mx], [intercept, slope * mx + intercept],
                 color="red", linewidth=1, label=f"Fit, slope {slope:.2f}")
    axes[1].set_xlabel("Measured (kWh/day)")
    axes[1].set_ylabel("Modelled (kWh/day)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    diurnal = df.groupby(df.index.hour)[["measured", "model"]].mean()
    axes[2].plot(diurnal.index, diurnal["measured"], color="tab:green",  label="Measured")
    axes[2].plot(diurnal.index, diurnal["model"],    color="tab:orange", label="Modelled")
    axes[2].set_xlabel("Hour of day (Tinos local)")
    axes[2].set_ylabel("Mean PV (kWh/h)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig("outputs/pv_validation.png", dpi=150)
    plt.show()
    print("Saved: outputs/pv_validation.png")


def plot_diagnostics(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- Panel 1: curtailment by battery state of charge ---
    # Daytime hours only (model > 0.1 kWh). Ratio of measured to modelled energy
    # per 10% state-of-charge band; curtailment lowers the ratio as the battery fills.
    day = df[df["model"] > 0.1].copy()
    edges = np.arange(0, 101, 10)
    day["soc_bin"] = pd.cut(day["soc"], bins=edges, right=False)
    grp = day.groupby("soc_bin", observed=False)[["measured", "model"]].sum()
    ratio = grp["measured"] / grp["model"]
    centres = [(iv.left + iv.right) / 2 for iv in ratio.index]

    axes[0].bar(centres, ratio.to_numpy(), width=8, color="tab:purple", edgecolor="black")
    axes[0].axhline(1.0, color="grey", linestyle="--", linewidth=1, label="Full harvest")
    axes[0].set_xlabel("Battery state of charge (%)")
    axes[0].set_ylabel("Measured / modelled energy")
    finite = ratio.to_numpy()[np.isfinite(ratio.to_numpy())]
    axes[0].set_ylim(0, max(1.1, finite.max() * 1.1) if finite.size else 1.1)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # --- Panel 2: summer vs winter average day (timezone/DST check) ---
    # Mean per local hour and season. The one-hour shift of the peak between
    # summer and winter is daylight saving time.
    is_summer = df.index.month.isin(SUMMER_MONTHS)
    is_winter = df.index.month.isin(WINTER_MONTHS)
    s = df[is_summer].groupby(df[is_summer].index.hour)[["measured", "model"]].mean()
    w = df[is_winter].groupby(df[is_winter].index.hour)[["measured", "model"]].mean()

    axes[1].plot(s.index, s["model"], color="tab:orange", label="Modelled, summer")
    axes[1].plot(w.index, w["model"], color="tab:blue",   label="Modelled, winter")
    axes[1].plot(s.index, s["measured"], color="tab:orange", linestyle="--", alpha=0.7, label="Measured, summer")
    axes[1].plot(w.index, w["measured"], color="tab:blue",   linestyle="--", alpha=0.7, label="Measured, winter")
    axes[1].set_xlabel("Hour of day (Tinos local)")
    axes[1].set_ylabel("Mean PV (kWh/h)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig("outputs/pv_diagnostics.png", dpi=150)
    plt.show()
    print("Saved: outputs/pv_diagnostics.png")


if __name__ == "__main__":
    df_validation = run_validation()
