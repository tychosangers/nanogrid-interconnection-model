# Nanogrid Interconnection Model

An open-source Python model that quantifies how much interconnecting a cluster of
off-grid nanogrids reduces the battery storage and the levelised cost of
electricity (LCOE) the cluster needs, compared with operating each nanogrid on
its own. The model isolates a single mechanism, the pooling of battery storage,
under full reliability and without any grid connection or energy market.

The model is built around a real case study, the Tinos Ecolodge, a cluster of
four buildings on the Greek island of Tinos, but it is written so that it can be
applied to another site by editing a single file.

This repository accompanies the MSc Energy Science thesis 
*Sharing Storage Between Off-Grid Nanogrids:The Techno-Economic Benefit of Interconnection for a Greek Island Cluster* 
by Tycho Sangers, Copernicus Institute of Sustainable Development, Utrecht University (2026).

> **Thesis version.** The release tagged `v1.0-thesis` is the exact version of
> the code that produced the results reported in the thesis. Later commits may
> contain improvements to the code and documentation.

## What the model does

For a given cluster, the model runs a 15-minute dispatch simulation over a full
year and sizes the battery with a parametric sweep: every battery size in a range
is simulated, its cost and reliability are computed, and the lowest-LCOE size
that serves the entire load is selected. It does this for two ways of operating
the same cluster:

- **Isolated**: every nanogrid has its own PV, its own optional wind turbine or
  diesel generator, and its own battery, each sized separately.
- **Interconnected**: local generation serves local load first, the remaining
  surpluses and deficits are pooled across the cluster, and one shared battery
  is sized for the pooled residual.

The difference between the two is the interconnection benefit. On top of this
comparison, the scenario scripts examine how the benefit changes with the
technology configuration (PV-BESS, PV-Wind-BESS, PV-Diesel-BESS), the load
scenario (a basic household load and a heavy air-conditioning load), the cluster
size, and the cost and distance of the interconnection cable.

## Requirements

- Python 3.10 or newer
- numpy, pandas and matplotlib (see `requirements.txt`)

```bash
pip install -r requirements.txt
```

The model was developed and run in Spyder (Anaconda), where these packages are
included by default.

## Repository structure

```
configuration.py            Set up a run here (the only file you normally edit)
run_model.py                Run the base case

pv_profile.py               Inputs: raw data to kW generation and load series
wind_profile.py
load_profile.py

simulation.py               Core engine: dispatch, cost and LCOE, battery sizing,
optimiser.py                cable geometry and the two operating modes
cabling.py
operation.py

reporting.py                Figures and printed tables

scenario_*.py               Analyses for the sub-questions, each run on its own
pv_validation.py            Validation and diagnostics
diagnostics_worst_weeks.py
airco_visualisation.py

data/load/                  Measured load and PV data of the case study
data/solar/                 Irradiance and wind data (see Data availability)
outputs/                    Figures and CSVs written by the scripts
```

## Quick start

1. Open `configuration.py` and set up the run: the technology configuration,
   the parameter values and the modelling window.
2. Run `run_model.py`. It runs the isolated and interconnected operation and
   writes figures and tables to `outputs/`.
3. To answer a specific sub-question, run the matching `scenario_*.py` script.
   Each script is self-contained and uses the settings in `configuration.py`.

**Restart the Python kernel after every change to `configuration.py`**, so the
new values are used rather than those loaded at the previous run.

## Reproducing the thesis results

The two load scenarios use different modelling windows and battery ranges, set in
`configuration.py`:

| Load scenario | `USE_2025_2026_WINDOW` | `Battery_max_kWh` | `Battery_step_kWh` |
|---|---|---|---|
| Basic household load (2025) | `False` | `10` | `0.1` |
| Heavy A/C load (June 2025 to May 2026) | `True` | `250` | `0.5` |

Keep `CABLING_COSTS_ON = False` for all runs; the cabling analysis adds the cable
cost itself.

| Thesis part | Script | Load scenario |
|---|---|---|
| Base case, isolated vs interconnected | `run_model.py` | basic |
| SQ1: technology configurations | `scenario_comparison.py` | basic and heavy |
| SQ1: parameter sensitivity | `scenario_sensitivity.py` | basic |
| SQ1: carbon price | `scenario_carbon_tax.py` | basic |
| SQ1: PV capacity sweep | `scenario_pv_sensitivity.py` | heavy |
| SQ2: cluster size and subsets | `scenario_clustersize.py` | basic |
| SQ3: cabling, full cluster (part A) and subsets (part C) | `scenario_cabling.py` | basic |
| SQ3: cabling at the lowest-LCOE PV capacity (part B) | `scenario_cabling.py` | heavy |
| Grid extension comparison (appendix) | `scenario_grid_extension.py` | basic |
| Electric mobility (appendix) | `scenario_mobility.py` | basic |
| PV validation, worst weeks, A/C load | `pv_validation.py`, `diagnostics_worst_weeks.py`, `airco_visualisation.py` | as set |

Run `scenario_clustersize.py` before part C of `scenario_cabling.py`, since part C
reads its saved results. The toggles at the top of each script switch individual
parts and figures on or off.

## Data availability

The measured load and PV generation data of the Tinos Ecolodge in `data/load/`
are published with the consent of the site owner. Please cite this repository
and the accompanying thesis when using them.

The irradiance and wind data in `data/solar/` were obtained from Solcast.
[Keep one of the two sentences below, depending on Solcast's terms of use.]
*They are included here in accordance with Solcast's terms of use.*
*They are not redistributed here, in line with Solcast's terms of use; to run the
model, obtain equivalent data from Solcast or another source (for example PVGIS
or ERA5), place it in `data/solar/` in the column format described in
`pv_profile.py`, and point `SOLAR_FILE` in `configuration.py` at it.*

## Citation

If you use this model or its data, please cite:

> T. Sangers, "Nanogrid Interconnection Model," version 1.0-thesis, 2026.
> [Online]. Available: https://github.com/username/repository-name

A `CITATION.cff` file is included, so GitHub also shows a "Cite this
repository" button.

## Licence

The code is released under the MIT Licence (see `LICENSE`).

## Acknowledgements

Developed as part of an MSc Energy Science thesis at Utrecht University. The
author thanks the owner of the Tinos Ecolodge for sharing and allowing the
publication of the measured data, and the supervisors and experts acknowledged
in the thesis.
