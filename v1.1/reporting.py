# ============================================================================
#  reporting.py
#  Figures and console summaries of a model run (called by run_model.py):
#  battery sizes, LCOE composition, the lowest state-of-charge week, cable
#  and grid-extension break-even, and the carbon price comparison.
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from configuration import *
from optimiser import (decompose_lcoe_gap, cable_breakeven_length, grid_extension_breakeven,
                       sweep_cost_components, lcoe_breakdown)
from cabling import cable_length_routed


def plot_battery_distribution(iso_result, inter_optimal, n_nanogrids, names=None):
    sizes       = np.asarray(iso_result["optimal_batteries"], dtype=float)   # per-nanogrid isolated battery [kWh]
    iso_total   = iso_result["battery_kWh_total"]                            # cluster total = sizes.sum()
    inter_total = inter_optimal["battery_kWh_total"]                         # single pooled battery [kWh]
    saving_pct  = (1 - inter_total / iso_total) * 100                        # printed to the console for the figure caption

    # Names and colours from NANOGRIDS in configuration.py
    labels  = [NANOGRID_NAMES[i]   if i < len(NANOGRID_NAMES)   else f"Nanogrid {i+1}"
               for i in range(n_nanogrids)]
    colours = [NANOGRID_COLOURS[i] if i < len(NANOGRID_COLOURS) else plt.cm.tab10.colors[i]
               for i in range(n_nanogrids)]

    x_iso, x_inter = 0, 1                                                   # the two bar positions
    fig, ax = plt.subplots(figsize=(8, 6))

    # Isolated: one segment per nanogrid
    bottom = 0.0                                                            # running height where the next segment starts
    for size, label, colour in zip(sizes, labels, colours):
        ax.bar(x_iso, size, bottom=bottom, width=0.6, color=colour,
               edgecolor="black", linewidth=0.6, label=label)
        if size > iso_total * 0.03:                                         # only label segments tall enough to read
            ax.text(x_iso, bottom + size / 2, f"{size:.1f}",
                    ha="center", va="center", fontsize=9)
        bottom += size                                                     # lift the base for the next nanogrid

    # Interconnected: one shared battery
    ax.bar(x_inter, inter_total, width=0.6, color=COLOUR_INTERCONNECTED,
           edgecolor="black", linewidth=0.6, label="Interconnected cluster")

    # Totals above the bars
    ax.text(x_iso,   iso_total,   f"{iso_total:.1f}",   ha="center", va="bottom", fontsize=10)
    ax.text(x_inter, inter_total, f"{inter_total:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks([x_iso, x_inter])
    ax.set_xticklabels(["Isolated", "Interconnected"])
    ax.set_ylabel("BESS capacity (kWh)")
    ax.set_ylim(0, iso_total * 1.12)                                        # headroom so the top total label is not clipped
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)                                                  # gridlines behind the bars
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.95)

    plt.tight_layout()
    plt.savefig("outputs/scenario_battery_distribution.png", dpi=150)
    plt.show()
    print(f"Saved: outputs/scenario_battery_distribution.png "
          f"(interconnection saves {saving_pct:.1f}% on cluster BESS)")


def plot_lcoe_components(df_sweep, inter_optimal, config, n_nanogrids):
    """LCOE of the interconnected cluster against shared battery size, split into
    the fixed part, battery cost, diesel fuel and carbon price; marks the selected size."""
    x = df_sweep["battery_kWh_total"]
    lcoe = df_sweep["lcoe_eur_kWh"]
    batt, fuel, carbon = sweep_cost_components(df_sweep, config)
    fuel_carbon = fuel + carbon
    fixed = lcoe - batt - fuel_carbon

    fuel_label = "Diesel fuel" + (" + carbon" if (APPLY_CARBON_TAX and config["use_diesel"]) else "")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.stackplot(x, fixed, batt, fuel_carbon,
                 labels=["Fixed (PV, diesel O&M, cable)", "BESS", fuel_label],
                 colors=["lightgrey", COST_COLOURS["Battery capex"], COST_COLOURS["Diesel fuel"]], alpha=0.85)
    ax.plot(x, lcoe, color="black", linewidth=2, label="Total LCOE (interconnected)")

    sel_x = inter_optimal["battery_kWh_total"]
    ax.axvline(sel_x, color="black", linestyle=":", linewidth=1.2,
               label=f"Selected shared BESS, interconnected ({sel_x:.1f} kWh)")

    ax.set_xlabel("Shared BESS capacity, interconnected cluster (kWh)")
    ax.set_ylabel("LCOE contribution (EUR/kWh)")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", fontsize=9, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig("outputs/lcoe_components.png", dpi=150)
    plt.show()
    print(f"Saved: outputs/lcoe_components.png  (interconnected selected BESS {sel_x:.1f} kWh)")


def plot_lcoe_breakdown(iso_result, inter_optimal, config, n_nanogrids):
    """Stacked LCOE of isolated and interconnected operation, split into cost terms [EUR/kWh]."""
    iso_terms   = lcoe_breakdown(iso_result,    config)
    inter_terms = lcoe_breakdown(inter_optimal, config)

    # Leave out terms that are zero in both bars
    labels = [name for name in iso_terms
              if abs(iso_terms[name]) > 1e-12 or abs(inter_terms[name]) > 1e-12]

    totals = np.array([sum(iso_terms.values()), sum(inter_terms.values())])   # bar heights = the two LCOEs
    drop_pct = (1 - totals[1] / totals[0]) * 100

    x = np.array([0.0, 1.0])
    bottoms = np.zeros(2)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for name in labels:
        values = np.array([iso_terms[name], inter_terms[name]])
        ax.bar(x, values, bottom=bottoms, width=0.5, color=COST_COLOURS.get(name, "lightgrey"),
               edgecolor="black", linewidth=0.5, label=COST_LABELS.get(name, name))
        for i in (0, 1):
            if values[i] > 0.05 * totals.max():
                ax.text(x[i], bottoms[i] + values[i] / 2, f"{values[i]:.2f}",
                        ha="center", va="center", fontsize=8)
        bottoms = bottoms + values                              # raise the floor for the next segment

    for i in (0, 1):                                            # total LCOE on top of each bar
        ax.annotate(f"{totals[i]:.3f}",
                    xy=(x[i], totals[i]), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(["Isolated", "Interconnected"])
    ax.set_ylabel("LCOE contribution (EUR/kWh)")
    ax.set_ylim(0, totals.max() * 1.15)                         # headroom for the total labels
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)                                      # gridlines behind the bars

    handles, legend_labels = ax.get_legend_handles_labels()     # reverse so the legend reads top-of-bar first
    ax.legend(handles[::-1], legend_labels[::-1], loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=9)

    plt.tight_layout()
    plt.savefig("outputs/lcoe_breakdown.png", dpi=150)
    plt.show()
    print(f"Saved: outputs/lcoe_breakdown.png (interconnection reduces LCOE by {drop_pct:.1f}%)")
    print(f"  check: stacked terms sum to {totals[0]:.4f} vs LCOE {iso_result['lcoe_eur_kWh']:.4f} (isolated), "
          f"{totals[1]:.4f} vs {inter_optimal['lcoe_eur_kWh']:.4f} (interconnected)")


def plot_soc_lowest_week(gen_cluster, load_cluster, inter_optimal, time_index, config):
    """Week around the lowest state of charge of the shared battery, with cluster generation and load.

    The simulation starts at 50% state of charge; a lowest point on the first
    day of the window therefore reflects the initial charge, not the sizing.
    """
    from simulation import run_simulation

    battery_kWh = float(inter_optimal["battery_kWh_total"])

    # Same pooling as run_interconnected
    direct = np.minimum(gen_cluster, load_cluster)
    pooled_gen  = (gen_cluster  - direct).sum(axis=1)
    pooled_load = (load_cluster - direct).sum(axis=1)

    sim = run_simulation(pooled_gen, pooled_load, battery_kWh, use_diesel=config["use_diesel"])
    soc = sim["soc_kWh"]
    soc_pct = soc / battery_kWh * 100.0

    low_i = int(np.argmin(soc))                                   # timestep of lowest SoC = the binding moment
    steps_per_day = int(24 * 60 / TIME_STEP_MIN)
    half = int(3.5 * steps_per_day)                              # half a week each side
    a = max(0, low_i - half)
    b = min(len(soc), low_i + half)
    week = time_index[a:b]

    total_gen  = gen_cluster.sum(axis=1)                          # whole-cluster context for the lower panel
    total_load = load_cluster.sum(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(week, soc_pct[a:b], color="tab:blue", linewidth=1.4, label="Shared battery SoC")
    ax1.axhline(SOC_MAX * 100, color="tab:red", linestyle="--", linewidth=1, label="SoC maximum")
    ax1.axhline(SOC_MIN * 100, color="tab:red", linestyle="--", linewidth=1, label="SoC minimum")
    ax1.scatter([time_index[low_i]], [soc_pct[low_i]], color="black", zorder=5,
                label=f"Lowest {soc_pct[low_i]:.0f}% ({soc[low_i]:.2f} kWh), {time_index[low_i].strftime('%d %b %Y %H:%M')}")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("State of charge (%)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_axisbelow(True)

    ax2.plot(week, total_gen[a:b], color="tab:orange", linewidth=1.0, label="Cluster generation")
    ax2.plot(week, total_load[a:b], color="tab:green", linewidth=1.0, label="Cluster load")
    ax2.set_ylim(bottom=0)                                        # start the power axis at zero
    ax2.set_ylabel("Power (kW)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_axisbelow(True)

    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig("outputs/soc_lowest_week.png", dpi=150)
    plt.show()
    print(f"Saved: outputs/soc_lowest_week.png  (shared BESS {battery_kWh:.1f} kWh, "
          f"lowest SoC {soc[low_i]:.2f} kWh = {soc_pct[low_i]:.0f}% on {time_index[low_i].date()})")

def _annotate_bars(ax, bars, fmt="{:.1f}"):
    """Value label above each bar."""
    for bar in bars:
        ax.annotate(fmt.format(bar.get_height()),
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9)


def print_carbon_tax_comparison(iso_base, inter_base, iso_carbon, inter_carbon, config):
    """Print LCOE, benefit and renewable fraction without and with the carbon price."""
    price_tonne = CARBON_PRICE_EUR_KG * 1000.0                    # 0.08 EUR/kg = 80 EUR/tonne
    cost_per_L  = DIESEL_EMISSIONS_L * CARBON_PRICE_EUR_KG        # extra fuel cost from the tax [EUR/L]
    pct_fuel    = cost_per_L / DIESEL_PRICE_EUR_L * 100           # that, as a % of the base diesel price
    eff_price   = DIESEL_PRICE_EUR_L + cost_per_L                 # effective diesel price under the tax [EUR/L]

    gap_base   = iso_base["lcoe_eur_kWh"]   - inter_base["lcoe_eur_kWh"]
    gap_carbon = iso_carbon["lcoe_eur_kWh"] - inter_carbon["lcoe_eur_kWh"]
    red_base   = (1 - inter_base["lcoe_eur_kWh"]   / iso_base["lcoe_eur_kWh"])   * 100
    red_carbon = (1 - inter_carbon["lcoe_eur_kWh"] / iso_carbon["lcoe_eur_kWh"]) * 100
    re_iso_b,  re_iso_c   = iso_base["renewable_fraction"]*100,   iso_carbon["renewable_fraction"]*100
    re_int_b,  re_int_c   = inter_base["renewable_fraction"]*100, inter_carbon["renewable_fraction"]*100

    print("\n" + "=" * 64)
    print(f"  CARBON TAX OVERLAY  (config = {config['name']})")
    print(f"  single EU ETS price: {price_tonne:.0f} EUR/tonne CO2 ({CARBON_PRICE_EUR_KG:.2f} EUR/kg)")
    print("=" * 64)
    print(f"  effective diesel price : {DIESEL_PRICE_EUR_L:.2f} -> {eff_price:.2f} EUR/L "
          f"(+{cost_per_L:.2f} EUR/L, +{pct_fuel:.1f}%)")
    print("-" * 64)
    print(f"  {'':<26}{'base':>11}{'+carbon':>11}{'change':>11}")
    print(f"  {'iso LCOE [EUR/kWh]':<26}{iso_base['lcoe_eur_kWh']:>11.4f}{iso_carbon['lcoe_eur_kWh']:>11.4f}"
          f"{iso_carbon['lcoe_eur_kWh']-iso_base['lcoe_eur_kWh']:>+11.4f}")
    print(f"  {'inter LCOE [EUR/kWh]':<26}{inter_base['lcoe_eur_kWh']:>11.4f}{inter_carbon['lcoe_eur_kWh']:>11.4f}"
          f"{inter_carbon['lcoe_eur_kWh']-inter_base['lcoe_eur_kWh']:>+11.4f}")
    print(f"  {'benefit gap [EUR/kWh]':<26}{gap_base:>11.4f}{gap_carbon:>11.4f}{gap_carbon-gap_base:>+11.4f}")
    print(f"  {'benefit [% LCOE reduc.]':<26}{red_base:>10.1f}%{red_carbon:>10.1f}%{red_carbon-red_base:>+9.1f}pp")
    print(f"  {'renew. frac, isolated':<26}{re_iso_b:>10.1f}%{re_iso_c:>10.1f}%{re_iso_c-re_iso_b:>+9.1f}pp")
    print(f"  {'renew. frac, interconn.':<26}{re_int_b:>10.1f}%{re_int_c:>10.1f}%{re_int_c-re_int_b:>+9.1f}pp")
    print(f"  {'diesel, isolated [kWh]':<26}{iso_base['annual_diesel_kWh']:>11.0f}{iso_carbon['annual_diesel_kWh']:>11.0f}"
          f"{iso_carbon['annual_diesel_kWh']-iso_base['annual_diesel_kWh']:>+11.0f}")
    print(f"  {'diesel, interconn. [kWh]':<26}{inter_base['annual_diesel_kWh']:>11.0f}{inter_carbon['annual_diesel_kWh']:>11.0f}"
          f"{inter_carbon['annual_diesel_kWh']-inter_base['annual_diesel_kWh']:>+11.0f}")
    print("=" * 64)
    if gap_base != 0:
        print(f"  benefit gap change: {(gap_carbon/gap_base - 1)*100:+.1f}% relative")
    print("=" * 64)


def plot_carbon_tax_comparison(iso_base, inter_base, iso_carbon, inter_carbon, config):
    """Renewable fraction (left) and interconnection benefit (right), without and with the carbon price."""
    price_tonne   = CARBON_PRICE_EUR_KG * 1000.0
    base_colour   = "#B0B0B0"        # neutral grey for the no-tax case
    carbon_colour = "indianred"

    fig, (ax_re, ax_ben) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: renewable fraction per topology
    groups    = ["Isolated", "Interconnected"]
    x         = range(len(groups))                   # group centres at 0 and 1
    width     = 0.38
    re_base   = [iso_base["renewable_fraction"]*100,   inter_base["renewable_fraction"]*100]
    re_carbon = [iso_carbon["renewable_fraction"]*100, inter_carbon["renewable_fraction"]*100]

    bars_b = ax_re.bar([xi - width/2 for xi in x], re_base,   width,
                       color=base_colour,   edgecolor="black", label="No carbon tax")
    bars_c = ax_re.bar([xi + width/2 for xi in x], re_carbon, width,
                       color=carbon_colour, edgecolor="black", label=f"+{price_tonne:.0f} EUR/tonne")
    ax_re.set_xticks(list(x))                        # put one tick per group...
    ax_re.set_xticklabels(groups)                    # ...and label them with the topology names
    ax_re.set_ylabel("Renewable fraction (%)")
    ax_re.set_title("Share of load served by PV and BESS")
    ax_re.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax_re.set_axisbelow(True)
    ax_re.set_ylim(0, max(re_base + re_carbon) + 18)   # headroom so the legend clears the value labels
    ax_re.legend(fontsize=8, loc="upper left")
    _annotate_bars(ax_re, bars_b)
    _annotate_bars(ax_re, bars_c)

    # Right: interconnection benefit
    red_base   = (1 - inter_base["lcoe_eur_kWh"]   / iso_base["lcoe_eur_kWh"])   * 100
    red_carbon = (1 - inter_carbon["lcoe_eur_kWh"] / iso_carbon["lcoe_eur_kWh"]) * 100
    bars_ben = ax_ben.bar(["No carbon tax", f"+{price_tonne:.0f} EUR/tonne"],
                          [red_base, red_carbon],
                          color=[base_colour, carbon_colour], width=0.5, edgecolor="black")
    ax_ben.set_ylabel("LCOE reduction from interconnecting (%)")
    ax_ben.set_title("Interconnection benefit")
    ax_ben.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax_ben.set_axisbelow(True)
    _annotate_bars(ax_ben, bars_ben)

    plt.tight_layout()
    fname = "outputs/carbon_tax_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")

# ============================================================================
#  Console summaries
# ============================================================================


def print_comparison(iso_result, inter_optimal, n_nanogrids, config, horizon_note=""):
    """Print the isolated and interconnected results side by side, with the LCOE gap decomposition."""

    lcoe_iso   = iso_result["lcoe_eur_kWh"]
    lcoe_inter = inter_optimal["lcoe_eur_kWh"]
    lcoe_drop_pct = (1 - lcoe_inter / lcoe_iso) * 100                          # PRIMARY: economic benefit of interconnecting

    batt_iso   = iso_result["battery_kWh_total"]
    batt_inter = inter_optimal["battery_kWh_total"]
    batt_change_pct = (batt_inter / batt_iso - 1) * 100

    batt_part, fuel_part, cable_part, carbon_part, total_gap = decompose_lcoe_gap(iso_result, inter_optimal, config)

    # Rows that apply to this configuration
    show_diesel = config["use_diesel"]
    show_carbon = APPLY_CARBON_TAX and config["use_diesel"]
    show_cable  = CABLING_COSTS_ON

    print("\n" + "=" * 60)
    print(f"  SCENARIO COMPARISON (N = {n_nanogrids}, config = {config['name']})")
    print(f"  battery sized per topology, metric = LCOE")
    if show_carbon:
        print(f"  carbon tax applied: {CARBON_PRICE_EUR_KG:.2f} EUR/kg CO2")
    if not CABLING_COSTS_ON:
        print(f"  cabling cost OFF: interconnection cable modelled as free")
    if horizon_note:
        print(f"  {horizon_note}")
    print("=" * 60)
    print(f"  {'':<28}{'isolated':>13}{'interconn.':>13}")
    print("-" * 60)
    print(f"  {'cluster LCOE [EUR/kWh]':<28}{lcoe_iso:>13.4f}{lcoe_inter:>13.4f}")
    print(f"  {'fleet battery [kWh]':<28}{batt_iso:>13.1f}{batt_inter:>13.1f}")
    print(f"  {'renewable fraction':<28}{iso_result['renewable_fraction']:>12.1%}{inter_optimal['renewable_fraction']:>13.1%}")
    if show_diesel:
        print(f"  {'diesel energy [kWh]':<28}{iso_result['annual_diesel_kWh']:>13.0f}{inter_optimal['annual_diesel_kWh']:>13.0f}")
        print(f"  {'diesel fuel [L]':<28}{iso_result['diesel_fuel_L']:>13.0f}{inter_optimal['diesel_fuel_L']:>13.0f}")
        print(f"  {'CO2 emissions [kg]':<28}{iso_result['diesel_emissions_kg']:>13.0f}{inter_optimal['diesel_emissions_kg']:>13.0f}")
    curtail_label = "curtailed PV+wind [kWh]" if config["use_wind"] else "curtailed PV [kWh]"
    print(f"  {curtail_label:<28}{iso_result['annual_curtailed_kWh']:>13.0f}{inter_optimal['annual_curtailed_kWh']:>13.0f}")
    print("-" * 60)
    print(f"  PRIMARY  LCOE reduction by interconnecting: {lcoe_drop_pct:+.1f}%")
    print(f"  descript. battery capacity change:          {batt_change_pct:+.1f}%  "
          f"({'smaller' if batt_change_pct < 0 else 'larger'})")
    print("-" * 60)
    print(f"  LCOE gap decomposition:")
    print(f"    from battery cost change : {batt_part:+.4f} EUR/kWh")
    if show_diesel:
        print(f"    from diesel fuel change  : {fuel_part:+.4f} EUR/kWh")
    if show_cable:
        print(f"    from cable cost (interc.): {cable_part:+.4f} EUR/kWh")
    if show_carbon:
        print(f"    from carbon tax change   : {carbon_part:+.4f} EUR/kWh")
    print(f"    sum (check vs total gap) : {batt_part + fuel_part + cable_part + carbon_part:+.4f} vs {total_gap:+.4f}")
    print("=" * 60)


def print_cable_breakeven(iso_result, inter_optimal, config, horizon_years):
    """Print the longest cable for which interconnection breaks even."""
    annual_benefit, length_disc, length_simple = cable_breakeven_length(iso_result, inter_optimal, config, horizon_years)
    routed = cable_length_routed(NANOGRID_COORDINATES)

    print(f"\n  CABLE TIPPING POINT (break even within {horizon_years} years)")
    if length_disc is None:
        print(f"    interconnection saves {annual_benefit:,.0f} EUR/yr before the cable, so no length breaks even")
        print("=" * 60)
        return

    print(f"    annual benefit before cable : {annual_benefit:>10,.0f} EUR/yr")
    print(f"    max cable length, discounted: {length_disc:>10,.0f} m   (amortised over {horizon_years} yr at {DISCOUNT_RATE:.0%})")
    print(f"    max cable length, simple    : {length_simple:>10,.0f} m   (undiscounted)")
    if routed <= length_disc:
        print(f"    actual routed cable length  : {routed:>10,.0f} m   -> breaks even ({length_disc / routed:.0f}x headroom)")
    else:
        print(f"    actual routed cable length  : {routed:>10,.0f} m   -> does NOT break even within {horizon_years} yr")
    print("=" * 60)


def print_grid_comparison(result, grid_price, lifetime_years):
    """Print the maximum grid extension cost at which the cluster remains cheaper."""
    nano_cost, grid_energy, breakeven_capex = grid_extension_breakeven(result, grid_price, lifetime_years)
    lcoe = result["lcoe_eur_kWh"]

    print(f"\n  GRID EXTENSION COMPARISON (over {lifetime_years} years, grid price {grid_price:.2f} EUR/kWh)")
    print(f"    interconnected LCOE        : {lcoe:>12.4f} EUR/kWh  (grid: {grid_price:.2f})")
    print(f"    nanogrid lifetime cost     : {nano_cost:>12,.0f} EUR")
    print(f"    grid energy over lifetime  : {grid_energy:>12,.0f} EUR")
    if breakeven_capex >= 0:
        print(f"    max grid-extension capex   : {breakeven_capex:>12,.0f} EUR   (a costlier extension means the nanogrid wins)")
    else:
        print(f"    nanogrid beats grid energy alone: even a free extension loses by {-breakeven_capex:,.0f} EUR")
    print("=" * 60)


def print_generation_split(pv_cluster, wind_cluster, n_nanogrids):
    hours = TIME_STEP_MIN / 60
    n_steps = pv_cluster.shape[0]
    pv_gen   = pv_cluster.sum()   * hours
    wind_gen = wind_cluster.sum() * hours
    pv_cf    = pv_cluster.sum()   / (PV_CAPACITY_KW   * n_nanogrids * n_steps)
    wind_cf  = wind_cluster.sum() / (WIND_CAPACITY_KW * n_nanogrids * n_steps)
    total    = pv_gen + wind_gen

    print("  generation split over the window (gross, before dispatch):")
    print(f"    PV   : {pv_gen:>10.0f} kWh  ({pv_gen/total:.0%} of supply, capacity factor {pv_cf:.0%})")
    print(f"    wind : {wind_gen:>10.0f} kWh  ({wind_gen/total:.0%} of supply, capacity factor {wind_cf:.0%})")
    print("=" * 60)
