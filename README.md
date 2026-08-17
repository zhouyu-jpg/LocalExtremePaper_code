# Local Temperature Extremes

[![DOI](https://img.shields.io/badge/DOI-10.1088%2F1748--9326%2Fae9202-blue)](https://doi.org/10.1088/1748-9326/ae9202) [![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Research code for estimating probability-density curves for local daily maximum and minimum temperatures. The repository supports the paper ["Local temperature extremes have become more extreme across many U.S. weather stations"](https://doi.org/10.1088/1748-9326/ae9202).


## Repository layout

```text
.
|-- Arbitrary Window/           Configurable odd windows from 7 through 31 days
|-- Data/                       Downloaded station CSVs, excluded from Git
|-- download_figshare_data.py   Figshare data downloader
|-- check_sas_*.py              Five-day SHASH parameter fitting
|-- GHCN_batch_*.py             Five-day station and calendar-day drivers
|-- ghcn*_shash_cy.pyx          Five-day Cython density implementations
|-- setup_shash.py              Five-day Cython build script
`-- requirements.txt            Python dependencies
```

## Quick start

Python 3 and a C/C++ compiler supported by Cython are required.

```bash
git clone https://github.com/zhouyu-jpg/LocalExtremePaper_code.git
cd LocalExtremePaper_code

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download all 45 GHCNd station files from Figshare.

```bash
python download_figshare_data.py
```

Download one or more stations for a small test:

```bash
python download_figshare_data.py --station USW00013968
```

## Five-day workflow

Run these commands from the repository root.

### 1. Fit daily SHASH parameters

```bash
python check_sas_4param_window_TMAX.py
python check_sas_4param_window_TMIN.py
```

The scripts pool five calendar days and fit the 1961-1990 reference period. Station-level parameters are written to `SAS_4Param_Window_Results_TMAX/` and `SAS_4Param_Window_Results_TMIN/`.

### 2. Build the Cython extensions

```bash
python setup_shash.py build_ext --inplace
```

### 3. Calculate daily densities

`--start` is inclusive and `--end` is exclusive. A complete non-leap calendar uses indices 0 through 364.

```bash
python GHCN_batch_SHASH_Cython_TMAX.py --cityID USW00013968 --start 0 --end 365
python GHCN_batch_SHASH_Cython_TMIN.py --cityID USW00013968 --start 0 --end 365
```

Five-day density files are written below `GHCN-Res/TMAX_SHASH/<stationID>/` and `GHCN-Res/TMIN_SHASH/<stationID>/`.

## Configurable-window workflow

The configurable implementation accepts odd windows from 7 through 31 days. Build the extensions from their source directory:

```bash
cd "Arbitrary Window"
python setupTMAX_shash_windowN.py build_ext --inplace
python setupTMIN_shash_windowN.py build_ext --inplace
python GHCN_batch_SHASH_Cython_TMAX_windowN.py \
  --cityID USW00013968 --start 0 --end 365 --window 15 \
  --param-file ../path/to/tmax-15-day-parameters.csv

python GHCN_batch_SHASH_Cython_TMIN_windowN.py \
  --cityID USW00013968 --start 0 --end 365 --window 15 \
  --param-file ../path/to/tmin-15-day-parameters.csv
```

## Input data

The file must include `DATE`, `year`, `TMAX`, and `TMIN`. Temperature values are stored in tenths of a degree Celsius. February 29 is removed by the analysis scripts, and the reference period is 1961-1990.


## Citation

Yu, Z., Cheng, D., Blanken, P. D., & Cao, G. (2026). Local temperature extremes have become more extreme across many U.S. weather stations. *Environmental Research Letters, 21*(15), 154034. https://doi.org/10.1088/1748-9326/ae9202
