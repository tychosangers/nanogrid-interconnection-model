# ============================================================================
#  configuration.py
#  All model settings and parameter values. This is the only file to edit
#  when applying the model to another site; then run run_model.py.
# ============================================================================


# ============================================================================
#  1. SITE, TIME AND LOAD
# ============================================================================

TIMEZONE = "Europe/Athens"
SOLCAST_UTC_OFFSET_H = 1          # clock offset of the Solcast export relative to UTC [h], no DST

TIME_STEP_MIN = 15                # simulation time step [min]
DISCOUNT_RATE = 0.08
N_NANOGRIDS = 4                   # number of nanogrids in the cluster

# Load data columns, one per nanogrid (15-minute kW series, aligned in time).
LOAD_COLUMNS = ["Big house", "Small house", "Utility house", "Greenhouse"]

# Modelling window and the matching Solcast weather file.
#   False: calendar year 2025 (basic household load)
#   True:  June 2025 to May 2026 (includes the air-conditioning load in the Small Lodge)
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
#  2. TECHNOLOGICAL CONFIGURATION
# ============================================================================

CONFIG_PV_BESS = {
    "name":       "PV-BESS",
    "colour":     "goldenrod",
    "use_diesel": False,
    "use_wind":   False,
}

CONFIG_PV_DIESEL_BESS = {
    "name":       "PV-Diesel-BESS",
    "colour":     "dimgrey",
    "use_diesel": True,
    "use_wind":   False,
}

CONFIG_PV_WIND_BESS = {
    "name":       "PV-Wind-BESS",
    "colour":     "seagreen",
    "use_diesel": False,
    "use_wind":   True,
}

# Configuration used by run_model.py: CONFIG_PV_BESS, CONFIG_PV_DIESEL_BESS or CONFIG_PV_WIND_BESS.
CONFIGURATION = CONFIG_PV_BESS
DEFAULT_CONFIG = CONFIGURATION    # used by functions that are called without a config


# ============================================================================
#  3. FIGURE STYLE
#  Each colour has one meaning across all figures: purple is reserved for the
#  interconnected cluster, and each figure uses colour for one role only
#  (topology, nanogrid, configuration or cost term).
# ============================================================================

# Keys are the load data columns; "display" is the name shown on figures.
NANOGRIDS = {
    "Big house":     {"display": "Big Lodge",     "tag": "BL", "colour": "#d62728"},
    "Small house":   {"display": "Small Lodge",   "tag": "SL", "colour": "#ff7f0e"},
    "Utility house": {"display": "Utility House", "tag": "UH", "colour": "#bcbd22"},
    "Greenhouse":    {"display": "Greenhouse",    "tag": "GH", "colour": "#2ca02c"},
}
NANOGRID_NAMES   = [NANOGRIDS[col]["display"] for col in LOAD_COLUMNS]
NANOGRID_TAGS    = [NANOGRIDS[col]["tag"]     for col in LOAD_COLUMNS]
NANOGRID_COLOURS = [NANOGRIDS[col]["colour"]  for col in LOAD_COLUMNS]

COLOUR_ISOLATED       = "#6E8298"   # slate blue-grey
COLOUR_INTERCONNECTED = "#9370DB"   # mediumpurple

COST_COLOURS = {
    "PV capex":      "goldenrod",      "PV O&M":      "khaki",
    "Wind capex":    "seagreen",       "Wind O&M":    "darkseagreen",
    "Diesel capex":  "dimgrey",        "Diesel O&M":  "darkgrey",
    "Diesel fuel":   "firebrick",      "Carbon tax":  "indianred",
    "Battery O&M":   "lightsteelblue", "Battery capex": "steelblue",
    "Cable capex":   "#8C6D4F",
}

# Legend labels for cost terms whose display name differs from their key.
COST_LABELS = {
    "Battery capex": "BESS capex",
    "Battery O&M":   "BESS O&M",
}

# Optional presentation versions of the figures (larger text, saved in outputs/slides/).
SLIDE_FIGURES = False
SLIDE_SCALE   = 1.6       # text enlargement factor of the slide version
SLIDE_DPI     = 300


def save_figure(fig, fname, scale=None, hide_in_slides=(), show_in_slides=(), **kwargs):
    """Save a figure and, if SLIDE_FIGURES is True, also a slide version.

    hide_in_slides and show_in_slides are figure elements that are only
    hidden or only shown in the slide version.
    """
    import os
    fig.savefig(fname, **kwargs)
    if not SLIDE_FIGURES:
        return
    scale = scale or SLIDE_SCALE
    width, height = fig.get_size_inches()
    fig.set_size_inches(width / scale, height / scale)
    for item in hide_in_slides:
        item.set_visible(False)
    for item in show_in_slides:
        item.set_visible(True)
    fig.tight_layout()
    folder = os.path.join(os.path.dirname(str(fname)), "slides")
    os.makedirs(folder, exist_ok=True)
    slide_name = os.path.join(folder, os.path.basename(str(fname)))
    fig.savefig(slide_name, dpi=SLIDE_DPI, bbox_inches="tight")
    for item in hide_in_slides:
        item.set_visible(True)
    for item in show_in_slides:
        item.set_visible(False)
    fig.set_size_inches(width, height)
    print(f"Saved slide version: {slide_name}")


# ============================================================================
#  4. TECHNO-ECONOMIC PARAMETERS
# ============================================================================

# Split of the cluster PV and wind capacity over the nanogrids:
#   "load":  proportional to each nanogrid's share of annual consumption
#   "equal": the same capacity on every nanogrid
#   a list:  explicit kW per nanogrid in LOAD_COLUMNS order, summing to PV_CAPACITY_KW * N_NANOGRIDS
# The cluster total, and therefore the generation cost, is the same for every option.
ALLOCATION = "load"

# --- PV ---
PV_CAPACITY_KW = 4.2/4            # mean capacity per nanogrid [kWp]; cluster total 4.2 kWp
PV_PR = 0.9                       # performance ratio: all optical and electrical losses relative to STC
PV_LIFETIME = 40                  # [years]
PV_CAPEX_EUR_KW = 720             # installed, including inverter [EUR/kWp]
PV_OPEX_EUR_KW_YR = 6.7           # [EUR/kWp/year]

# --- Battery (BESS) ---
# Battery sweep [kWh]. Basic load scenario: 0 to 10 in steps of 0.1.
# Heavy load scenario: 0 to 250 in steps of 0.5. The upper bound must exceed the optimum.
BATTERY_MIN_KWH = 0
BATTERY_MAX_KWH = 10
BATTERY_STEP_KWH = 0.1
BATTERY_EFFICIENCY = 0.91         # round-trip efficiency
BATTERY_CHARGE_EFF = BATTERY_EFFICIENCY ** 0.5
BATTERY_DISCHARGE_EFF = BATTERY_EFFICIENCY ** 0.5
SOC_MIN = 0.10                    # minimum state of charge (fraction of capacity)
SOC_MAX = 0.90                    # maximum state of charge (fraction of capacity)
BATTERY_LIFETIME = 15             # [years]
BATTERY_CAPEX_EUR_KWH = 400       # installed, including inverter [EUR/kWh]
BATTERY_OPEX_EUR_KWH_YR = 10      # [EUR/kWh/year]

# --- Diesel generator (CONFIG_PV_DIESEL_BESS) ---
DIESEL_CAPACITY_KW = 3.0          # fallback nameplate [kW]; normally sized to each nanogrid's peak load
DIESEL_EFFICIENCY = 0.35          # fuel to electricity
DIESEL_PRICE_EUR_L = 1.81         # island price [EUR/L]
DIESEL_KWH_L = 10.0               # lower heating value [kWh/L]
DIESEL_LIFETIME = 25              # [years]
DIESEL_EMISSIONS_L = 3.16         # tank-to-wheel [kg CO2/L]
DIESEL_CAPEX_EUR_KW = 500         # [EUR/kW]
DIESEL_OPEX_EUR_KW_YR = 30        # [EUR/kW/year]

# --- Wind turbine (CONFIG_PV_WIND_BESS) ---
WIND_CAPACITY_KW = 1/4            # mean capacity per nanogrid [kW]; cluster total 1 kW
WIND_DATA_HEIGHT_M = 10           # measurement height of the wind data [m]
WIND_HUB_HEIGHT_M = 11            # turbine hub height [m]
WIND_FRICTION_COEFF = 0.15        # power-law wind shear exponent
WIND_CUT_IN_SPEED = 3.0           # [m/s]
WIND_RATED_SPEED = 11.0           # [m/s]
WIND_CUT_OUT_SPEED = 54.0         # [m/s]
WIND_CAPEX_EUR_KW = 3765          # Bergey XL.1, installed [EUR/kW]
WIND_OPEX_EUR_KW_YR = WIND_CAPEX_EUR_KW * 0.05   # [EUR/kW/year]
WIND_LIFETIME = 15                # [years]


# ============================================================================
#  5. OPTIONAL SETTINGS
# ============================================================================

# --- Carbon price on diesel emissions ---
APPLY_CARBON_TAX = False
CARBON_PRICE_EUR_KG = 0.08        # [EUR/kg CO2], 80 EUR/tonne

# --- Interconnection cable ---
# True: the interconnected cluster pays for the cable.
# False: the cable is free, which isolates the storage-pooling benefit.
CABLING_COSTS_ON = False
CABLE_CAPEX_EUR_M = 30            # buried LV cable [EUR/m], sensitivity range 20 to 85
CABLE_LIFETIME = 40               # [years]
BREAKEVEN_HORIZON_YEARS = 20      # horizon for the cable break-even length [years]
CABLE_ROUTING_FACTOR = 1.20       # routed length relative to the straight-line MST length

NANOGRID_COORDINATES = [          # (lat, lon) per nanogrid, in LOAD_COLUMNS order
    (37.565552, 25.215728),       # Big Lodge
    (37.565448, 25.215527),       # Small Lodge
    (37.565451, 25.215634),       # Utility House
    (37.565689, 25.215215),       # Greenhouse
]

# --- Grid extension comparison ---
GRID_PRICE_EUR_KWH = 0.236        # main-grid electricity price [EUR/kWh]
GRID_PROJECT_LIFETIME_YEARS = 20  # comparison horizon [years]
