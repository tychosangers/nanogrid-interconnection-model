# Changelog

## v1.1 (2026-10-15)

Code clarity and consistency release. The model results are identical to
`v1.0-thesis`: battery sizes, LCOE, renewable fraction, curtailment and diesel
fuel were compared for all three configurations in both load scenarios.

### Changed
- Constants in `configuration.py` are written in capitals (for example
  `DISCOUNT_RATE`, `BATTERY_CAPEX_EUR_KWH`); override keys follow the same names.
- Configurations are named as in the thesis: `CONFIG_PV_BESS`,
  `CONFIG_PV_WIND_BESS` and `CONFIG_PV_DIESEL_BESS`.
- "Nanogrid" is used throughout the code (`n_nanogrids`, `NANOGRID_NAMES`,
  `NANOGRID_COORDINATES`); figures and printouts show the display names.
- Headers, comments and docstrings are shortened and describe what the code does.
- Defaults reproduce the base case of the thesis: basic load scenario, battery
  sweep from 0 to 10 kWh in steps of 0.1 kWh, slide versions of figures off.
- All scripts write to `outputs/`.
- `scenario_curtailment.py` uses the modelling window of `configuration.py`.
- Shared code for the minimum spanning tree and for reading Solcast files is
  defined once.

### Removed
- `scenario_mobility.py`, replaced by `scenario_curtailment.py`.
- `scenario_carbon_tax.py`; the carbon price comparison is produced by
  `run_model.py` with `APPLY_CARBON_TAX = True`.
- Figure variants that are not part of the thesis: exploratory load figures,
  the sensitivity spider charts and related figures, the solid battery figure
  and grid-extension break-even figure in `scenario_comparison.py`, three
  diagnostic curtailment figures, and the battery total figure in `reporting.py`.

### Fixed
- Sub-question labels in the printouts of `scenario_clustersize.py`.
- Figure titles that showed a fixed year instead of the modelling window.
- The fallback cluster load in `scenario_grid_extension.py` (1303 kWh).
