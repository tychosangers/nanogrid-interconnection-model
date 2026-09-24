# ============================================================================
#  diagnostic_worst_weeks.py
#  Diagnostic, not a sub-question result. For each lodge on its OWN, this sizes its
#  battery, replays its dispatch, and plots the week around its lowest state of
#  charge. It shows whether each standalone lodge is sized by a routine night
#  (diurnal cycling) or by a real multi-day generation shortfall, and on which
#  date. This is the evidence behind the discussion point that, in a strongly
#  PV-oversized system, the battery is set by a single night rather than a seasonal
#  deficit, which is why a year-long load-variety measure could not predict the
#  interconnection benefit.
#
#  It runs on whatever CONFIGURATION is set in configuration.py. Press run.
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt

from configuration import CONFIGURATION, Time_step_min, SOC_min, SOC_max
from load_profile import MODEL_BUILDINGS
from run_model import load_inputs
from optimiser import run_battery_sweep
from simulation import run_simulation


def plot_isolated_worst_weeks():
    base_loads, pv_cluster, wind_cluster, n_households, horizon_note, time_index = load_inputs()
    gen = pv_cluster if wind_cluster is None else pv_cluster + wind_cluster   # same supply the dispatch uses

    steps_per_day = int(24 * 60 / Time_step_min)
    half = int(3.5 * steps_per_day)            # half a week of timesteps either side of the worst point

    # n_households rows, 2 columns (SoC | generation-vs-load). squeeze=False keeps
    # axes as a 2-D grid even if a dimension is 1, so axes[h][0] always works.
    fig, axes = plt.subplots(n_households, 2, figsize=(13, 2.6 * n_households), squeeze=False)

    print("\n" + "=" * 78)
    print("  Isolated worst-SoC week per lodge")
    print(f"  ({horizon_note})")
    print("=" * 78)
    print(f"  {'lodge':16} {'battery kWh':>11} {'low SoC kWh':>12} {'binding date':>14}")

    for h in range(n_households):
        # Each lodge sized and dispatched on its OWN PV and load (no pooling).
        _, optimal_row = run_battery_sweep(gen[:, h], base_loads[:, h], verbose=False, config=CONFIGURATION)
        battery_kWh = float(optimal_row["battery_kWh"])

        sim = run_simulation(gen[:, h], base_loads[:, h], battery_kWh, use_diesel=CONFIGURATION["use_diesel"])
        soc = sim["soc_kWh"]
        soc_pct = soc / battery_kWh * 100.0    # SoC as % of this lodge's nominal capacity, so every panel shares the 10-90% window
        low_i = int(np.argmin(soc))            # the timestep that sizes this lodge's battery

        a = max(0, low_i - half)               # window edges, clipped to the data
        b = min(len(soc), low_i + half)
        week = time_index[a:b]
        name = MODEL_BUILDINGS[h]

        print(f"  {name:16} {battery_kWh:>11.1f} {soc[low_i]:>12.2f} "
              f"{time_index[low_i].strftime('%Y-%m-%d'):>14}")

        # Left: state of charge across the worst week, as % of capacity, with the SoC
        # limits and the lowest point.
        ax_soc = axes[h][0]
        ax_soc.plot(week, soc_pct[a:b], color="tab:blue", linewidth=1.3)
        ax_soc.axhline(SOC_max * 100, color="tab:red", linestyle="--", linewidth=1, label="SoC maximum")
        ax_soc.axhline(SOC_min * 100, color="tab:red", linestyle="--", linewidth=1, label="SoC minimum")
        ax_soc.scatter([time_index[low_i]], [soc_pct[low_i]], color="black", zorder=5)
        ax_soc.set_ylim(0, 100)
        ax_soc.set_ylabel("SoC (%)")
        ax_soc.set_title(f"{name}: {battery_kWh:.1f} kWh, lowest {soc_pct[low_i]:.0f}% on "
                 f"{time_index[low_i].strftime('%d %b %Y')}", fontsize=9, x=1.1)   # x>1 centres the title across both panels in the row
        ax_soc.legend(fontsize=7, loc="lower right")
        ax_soc.grid(alpha=0.3)
        ax_soc.set_axisbelow(True)
        ax_soc.tick_params(axis="x", labelrotation=30)

        # Right: generation vs load over the same week. Daily generation peaks with a
        # nightly load gap means diurnal cycling; a multi-day generation collapse with
        # load on top would mean a real deficit event.
        ax_gl = axes[h][1]
        ax_gl.plot(week, gen[a:b, h], color="tab:orange", linewidth=0.9, label="Generation")
        ax_gl.plot(week, base_loads[a:b, h], color="tab:green", linewidth=0.9, label="Load")
        ax_gl.set_ylim(bottom=0)               # start the power axis at zero
        ax_gl.set_ylabel("Power (kW)")
        ax_gl.legend(fontsize=7, loc="upper right")
        ax_gl.grid(alpha=0.3)
        ax_gl.set_axisbelow(True)
        ax_gl.tick_params(axis="x", labelrotation=30)

    print("=" * 78)
    axes[-1][0].set_xlabel("Date")
    axes[-1][1].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig("outputs/isolated_worst_weeks.png", dpi=150)
    plt.show()
    print(f"Saved: outputs/isolated_worst_weeks.png  ({CONFIGURATION['name']})")


if __name__ == "__main__":
    plot_isolated_worst_weeks()