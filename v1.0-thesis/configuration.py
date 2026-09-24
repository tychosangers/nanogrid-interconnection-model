# ============================================================================
#  configuration.py
#  The ONLY file a researcher edits to run the model on their own context.
#  Choose a technology configuration, set the parameter values, pick the
#  optional analyses, then press run on run_model.py.
#  Variable names follow config.py, so the rest of the model imports unchanged.
# ============================================================================


# ============================================================================
#  1. NECESSARY INPUT
# ============================================================================

# --- Site / location ---
Timezone  = "Europe/Athens"
SOLCAST_UTC_OFFSET_H = 1      # Solcast export clock vs UTC (fixed offset, no DST)

# --- Time resolution ---
Time_step_min = 15            # minutes

# --- Economics (core) ---
Discount_rate = 0.08

# --- Cluster size ---
N_nanogrids = 4               # number of nanogrids to analyse (4 = the four Tinos lodges)

# --- Load inputs ---
# For now these point at the four Tinos lodge load columns (see load_profile.py),
# not external files. Later they can become paths to a researcher's own CSVs.
load_file_1 = "Big house"
load_file_2 = "Small house"
load_file_3 = "Utility house"
load_file_4 = "Greenhouse"
LOAD_COLUMNS = [load_file_1, load_file_2, load_file_3, load_file_4]   # one column per nanogrid; this is what the model runs
# requirements load_files: one load per nanogrid, 15-minute kW series, aligned in time
# NOTE: a future option is to scale real data by a randomness or simultaneity factor

# --- Modelling period and matching resource file ---
# Two ready-made periods. Each selects its load window AND its own Solcast file,
# so the load dates and the solar/wind data always match (no file combining).
#   False -> calendar year 2025 (default)
#   True  -> June 2025 to May 2026 (more recent, includes the small-lodge air-con)
USE_2025_2026_WINDOW = False

if USE_2025_2026_WINDOW:
    MODEL_WINDOW_START = "2025-06-01"
    MODEL_WINDOW_END   = "2026-05-31"
    SOLAR_FILE = "data/solar/Solar_Wind_Data_TinosEcoLodge_2025_2026_Solcast.csv"
else:
    MODEL_WINDOW_START = "2025-01-01"
    MODEL_WINDOW_END   = "2025-12-31"
    SOLAR_FILE = "data/solar/Solar_Data_TinosEcoLodge_2025_Solcast.csv"



# ============================================================================
#  2. CONFIGURATION INPUT
# ============================================================================

# These dictionaries describe what each configuration includes (which generators
# are active). You normally leave them as they are and only edit the
# CONFIGURATION line below and the parameter values further down.

CONFIG_PV_ONLY = {                        # PV + battery only, no diesel
    "name":       "PV-BESS",
    "colour":     "goldenrod",            # config colour, equals its PV-capex cost colour
    "use_diesel": False,
    "use_wind":   False,
}

CONFIG_PV_DIESEL = {                      # PV + battery + diesel backup
    "name":       "PV-Diesel-BESS",
    "colour":     "dimgrey",              # config colour, equals its diesel-capex cost colour
    "use_diesel": True,
    "use_wind":   False,
}

CONFIG_PV_WIND = {                        # PV + wind + battery
    "name":       "PV-Wind-BESS",
    "colour":     "seagreen",             # config colour, equals its wind-capex cost colour
    "use_diesel": False,
    "use_wind":   True,
}

# >>> Choose your configuration here <<<
CONFIGURATION = CONFIG_PV_ONLY           # CONFIG_PV_ONLY, CONFIG_PV_DIESEL or CONFIG_PV_WIND
DEFAULT_CONFIG = CONFIGURATION            # engines fall back to this when called without an explicit config

# ============================================================================
#  FIGURE STYLE  (display names and colours shared by every figure)
#  Edit here once and all plots follow, because the plotting files do
#  `from configuration import *`. Two rules keep the figures honest:
#    1. one colour means one thing: purple is ALWAYS the interconnected
#       cluster, and nothing else is allowed to use it;
#    2. each figure uses colour for a SINGLE role (topology, or lodge, or
#       configuration, or cost term), so the four palettes below live on
#       different figures and are never mixed on the same axes.
# ============================================================================

# --- Lodge names and colours ---
# Keys are the raw data-column names (leave these as they are: the CSV and the
# model look them up). "display" is the label shown on figures; "colour" is the
# lodge's fixed colour wherever a figure splits a total per lodge.
LODGES = {
    "Big house":     {"display": "Big Lodge",     "tag": "BL", "colour": "#d62728"},   # red
    "Small house":   {"display": "Small Lodge",   "tag": "SL", "colour": "#ff7f0e"},   # orange
    "Utility house": {"display": "Utility House", "tag": "UH", "colour": "#bcbd22"},   # olive
    "Greenhouse":    {"display": "Greenhouse",    "tag": "GH", "colour": "#2ca02c"},   # green
}
# Ready-made lists in LOAD_COLUMNS order, so a plot can pair names, tags and colours directly.
LODGE_NAMES   = [LODGES[col]["display"] for col in LOAD_COLUMNS]
LODGE_TAGS    = [LODGES[col]["tag"]     for col in LOAD_COLUMNS]
LODGE_COLOURS = [LODGES[col]["colour"]  for col in LOAD_COLUMNS]

# --- Topology colours (isolated fleet vs interconnected cluster) ---
COLOUR_ISOLATED       = "#6E8298"   # slate blue-grey, the isolated / standalone case
COLOUR_INTERCONNECTED = "#9370DB"   # mediumpurple, ALWAYS the interconnected cluster (reserved)

# --- LCOE cost-term colours (stacked composition figures) ---
COST_COLOURS = {
    "PV capex":      "goldenrod",      "PV O&M":      "khaki",
    "Wind capex":    "seagreen",       "Wind O&M":    "darkseagreen",
    "Diesel capex":  "dimgrey",        "Diesel O&M":  "darkgrey",
    "Diesel fuel":   "firebrick",      "Carbon tax":  "indianred",
    "Battery O&M":   "lightsteelblue", "Battery capex": "steelblue",
    "Cable capex":   "#8C6D4F",
}

# Display names for the cost terms whose figure label differs from their internal
# key. The model keeps "Battery capex" and "Battery O&M" everywhere (values,
# stacking order, COST_COLOURS); only the legend text is swapped to BESS at draw
# time via COST_LABELS.get(name, name), so any term without an entry here keeps
# its own name unchanged.
COST_LABELS = {
    "Battery capex": "BESS capex",
    "Battery O&M":   "BESS O&M",
}

# --- Capacity allocation across nanogrids (governs BOTH PV and wind) ---
# How each generator's cluster total is split between the lodges:
#   "load"  -> proportional to each lodge's share of annual consumption (default, most realistic)
#   "equal" -> the same rated capacity on every lodge (the original SQ1 assumption)
#   a list  -> explicit kW per lodge in LOAD_COLUMNS order, e.g. [1.6, 0.8, 0.9, 0.9]
#              (must sum to PV_capacity_kW * N_nanogrids, so cost stays consistent)
# "load" and "equal" keep the cluster total at PV_capacity_kW * N_nanogrids, so total
# PV/wind capex and O&M are unchanged; only the per-lodge generation shape moves.
ALLOCATION = "load"

# --- PV + battery parameters (present in every configuration) ---
PV_capacity_kW = 4.2/4                                  # average kWp per nanogrid (4.2 kWp measured across the four lodges); cluster total = this * N_nanogrids
PV_PR = 0.9                                            # Normalises expected energy output to installed capacity and plane-of-array irradiation, aggregating all optical and electrical losses relative to STC: incident angle modifier, non-STC corrections, cable, inverter, clipping and transformer losses. 
PV_Lifetime = 40
PV_capex_eur_kW = 720                                  # €/kWp installed (already includes inverter and installation)
PV_opex_eur_kW_yr = 6.7

Battery_min_kWh = 0                                     # minimum size to test in sweep
Battery_max_kWh = 10                                   # maximum size to test (raise to ~300 for the PV and battery sweep scripts, else the sweep is capped)
Battery_step_kWh = 0.1                                   # step size for sweep
Battery_efficiency = 0.91                               # round-trip efficiency (DEA/NREL)
Battery_charge_eff = Battery_efficiency ** 0.5          # square root, so charge × discharge = round-trip
Battery_discharge_eff = Battery_efficiency ** 0.5       # ≈ 0.954
SOC_min = 0.10                                          # minimum state of charge to protect lifetime
SOC_max = 0.90                                          # maximum state of charge to protect lifetime
Battery_Lifetime = 15
Battery_capex_eur_kWh = 400                             # €/kWh installed (already includes inverter and installation)
Battery_opex_eur_kWh_yr = 10

# --- Diesel parameters (used by CONFIG_PV_DIESEL) ---
Diesel_capacity_kW = 3.0                                # sized to cover the peak cluster load
Diesel_efficiency = 0.35                                # fuel to electricity
Diesel_price_eur_L = 1.81                               # €/litre (higher on islands)
Diesel_kWh_L = 10.0                                     # kWh of fuel per litre, LHV, JRC
Diesel_Lifetime = 25                                    # years; small gensets are hours-limited, roughly 15000 to 20000 operating hours
Diesel_Emissions_L = 3.16                               # Tank-to-Wheel
Diesel_capex_eur_kW = 500                               # €/kW
Diesel_opex_eur_kW_yr = 30

# --- Wind parameters (used by CONFIG_PV_WIND) ---
Wind_capacity_kW = 1/4                                    # kW rated power of one turbine
Wind_data_height_m = 10                                 # height at which wind_speed_10m is measured in the data [m]
Wind_hub_height_m = 11                                  # turbine hub height [m]; wind is extrapolated from the data height to this
Wind_friction_coeff = 0.15                              # wind shear (power-law) exponent; WindEmpowerment
Wind_cut_in_speed = 3.0                                 # m/s, below this the turbine produces nothing
Wind_rated_speed = 11.0                                 # m/s, at and above this it produces rated power
Wind_cut_out_speed = 54.0                               # m/s, above this it shuts down

Wind_capex_eur_kW = 3765                                # EUR/kW installed; Bergey XL1
Wind_opex_eur_kW_yr = Wind_capex_eur_kW * 0.05          # EUR/kW/year O&M; WindEmpowerment
Wind_Lifetime = 15


# ============================================================================
#  3. OPTIONAL INPUTS
# ============================================================================

# --- Carbon tax (policy overlay on diesel emissions) ---
Apply_carbon_tax = False                                # True adds an EU ETS-style carbon cost of diesel across the whole analysis
Carbon_price_eur_kg = 0.08                             # €/kg CO2 (EU ETS order of magnitude, about 80 €/tonne)

# --- Cabling cost on/off (interconnection cable) ---
# True  = interconnection pays for the cable (realistic).
# False = cable modelled as free, which isolates the storage-pooling benefit from the cable penalty.
CABLING_COSTS_ON = False
Cable_capex_eur_m = 30        # €/m installed LV AC cable, buried on rocky island ground; sensitivity 20-85
Cable_Lifetime = 40           # years (LV cable is long-lived)
Breakeven_horizon_years = 20      # cable tipping point: longest interconnection cable that still breaks even within this many years
Cable_routing_factor = 1.20   # uplift on the straight-line MST length for vertical runs and ground obstacles

Tinos_coordinates = [         # (lat, lon) per nanogrid, used for the MST cable length
    (37.565552, 25.215728),   # Big house
    (37.565448, 25.215527),   # Small house
    (37.565451, 25.215634),   # Utility house
    (37.565689, 25.215215),   # Greenhouse
]

Grid_price_eur_kWh = 0.236          # average main-grid electricity price [EUR/kWh], for the grid-extension comparison
Grid_project_lifetime_years = 20   # horizon over which the nanogrid and a grid extension are compared