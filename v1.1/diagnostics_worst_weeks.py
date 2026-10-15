# ============================================================================
#  diagnostics_worst_weeks.py
#  For each isolated nanogrid: sizes its battery, simulates its dispatch and
#  plots the week around its lowest state of charge, next to its generation
#  and load. Shows which night or period sizes each battery.
#  Uses the CONFIGURATION set in configuration.py.
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt

from configuration import CONFIGURATION, TIME_STEP_MIN, SOC_MIN, SOC_MAX, NANOGRID_NAMES, save_figure
from run_model import load_inputs
from optimiser import run_battery_sweep
from simulation import run_simulation


def plot_isolated_worst_weeks():
    """Plot the week around the lowest state of charge of each isolated nanogrid."""
    base_loads, pv_cluster, wind_cluster, n_nanogrids, horizon_note, time_index = load_inputs()
    gen = pv_cluster if wind_cluster is None else pv_cluster + wind_cluster

    steps_per_day = int(24 * 60 / TIME_STEP_MIN)
    half = int(3.5 * steps_per_day)            # half a week on either side of the lowest point

    # One row per nanogrid: state of charge (left) and generation and load (right).
    fig, axes = plt.subplots(n_nanogrids, 2, figsize=(13, 2.6 * n_nanogrids), squeeze=False)

    print("\n" + "=" * 78)
    print("  Week with the lowest state of charge per isolated nanogrid")
    print(f"  ({horizon_note})")
    print("=" * 78)
    print(f"  {'nanogrid':16} {'battery kWh':>11} {'low SoC kWh':>12} {'binding date':>14}")

    for h in range(n_nanogrids):
        _, optimal_row = run_battery_sweep(gen[:, h], base_loads[:, h], verbose=False, config=CONFIGURATION)
        battery_kWh = float(optimal_row["battery_kWh"])

        sim = run_simulation(gen[:, h], base_loads[:, h], battery_kWh, use_diesel=CONFIGURATION["use_diesel"])
        soc = sim["soc_kWh"]
        soc_pct = soc / battery_kWh * 100.0
        low_i = int(np.argmin(soc))

        a = max(0, low_i - half)
        b = min(len(soc), low_i + half)
        week = time_index[a:b]
        name = NANOGRID_NAMES[h]

        print(f"  {name:16} {battery_kWh:>11.1f} {soc[low_i]:>12.2f} "
              f"{time_index[low_i].strftime('%Y-%m-%d'):>14}")

        ax_soc = axes[h][0]
        ax_soc.plot(week, soc_pct[a:b], color="tab:blue", linewidth=1.3)
        ax_soc.axhline(SOC_MAX * 100, color="tab:red", linestyle="--", linewidth=1, label="SoC maximum")
        ax_soc.axhline(SOC_MIN * 100, color="tab:red", linestyle="--", linewidth=1, label="SoC minimum")
        ax_soc.scatter([time_index[low_i]], [soc_pct[low_i]], color="black", zorder=5)
        ax_soc.set_ylim(0, 100)
        ax_soc.set_ylabel("SoC (%)")
        ax_soc.set_title(f"{name}: {battery_kWh:.1f} kWh, lowest {soc_pct[low_i]:.0f}% on "
                         f"{time_index[low_i].strftime('%d %b %Y')}", fontsize=9, x=1.1)   # centred over both panels
        ax_soc.legend(fontsize=7, loc="lower right")
        ax_soc.grid(alpha=0.3)
        ax_soc.set_axisbelow(True)
        ax_soc.tick_params(axis="x", labelrotation=30)

        ax_gl = axes[h][1]
        ax_gl.plot(week, gen[a:b, h], color="tab:orange", linewidth=0.9, label="Generation")
        ax_gl.plot(week, base_loads[a:b, h], color="tab:green", linewidth=0.9, label="Load")
        ax_gl.set_ylim(bottom=0)
        ax_gl.set_ylabel("Power (kW)")
        ax_gl.legend(fontsize=7, loc="upper right")
        ax_gl.grid(alpha=0.3)
        ax_gl.set_axisbelow(True)
        ax_gl.tick_params(axis="x", labelrotation=30)

    print("=" * 78)
    axes[-1][0].set_xlabel("Date")
    axes[-1][1].set_xlabel("Date")
    plt.tight_layout()
    save_figure(fig, "outputs/isolated_worst_weeks.png", scale=1.6, dpi=150)
    plt.show()
    print(f"Saved: outputs/isolated_worst_weeks.png  ({CONFIGURATION['name']})")


if __name__ == "__main__":
    plot_isolated_worst_weeks()
