# DataPipeline for Regional HRRR Train Dataset — Complete Tutorial

> **Purpose**: This document is a comprehensive, self-contained guide to the
> CorrDiff regional training-data pipeline. It is written so that (1) a human
> can clone the repo on any machine and reproduce every step, and (2) an LLM
> can read this file and make informed edits to any part of the codebase.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [Repository Structure](#3-repository-structure)
4. [File-by-File Reference](#4-file-by-file-reference)
5. [Environment Setup](#5-environment-setup)
6. [CDS API Credentials](#6-cds-api-credentials)
7. [Quick Start — Smoke Test](#7-quick-start--smoke-test)
8. [Stage 1 — ERA5 Download](#8-stage-1--era5-download)
9. [Stage 2 — Dataset Build](#9-stage-2--dataset-build)
10. [Stage 3 — Compute Stats](#10-stage-3--compute-stats)
11. [Full Production Run](#11-full-production-run)
12. [Deploying on Yale Bouchet HPC](#12-deploying-on-yale-bouchet-hpc)
13. [Deploying on Other HPC / Cloud](#13-deploying-on-other-hpc--cloud)
14. [Output NetCDF Schema](#14-output-netcdf-schema)
15. [Variable Mapping Reference](#15-variable-mapping-reference)
16. [Known Limitations & Design Decisions](#16-known-limitations--design-decisions)
17. [Troubleshooting](#17-troubleshooting)
18. [Making Edits — LLM Context Guide](#18-making-edits--llm-context-guide)
19. [Testing](#19-testing)
20. [Storage Estimates](#20-storage-estimates)

---

## 1. Project Overview

This pipeline builds a regional CorrDiff training dataset that downscales
**ERA5 reanalysis** (0.25°, ~25 km) to **HRRR operational forecasts** (~3 km)
over the CONUS ocean domain. The output NetCDF matches the **HRRR-Mini schema**
used by NVIDIA PhysicsNeMo CorrDiff, so it drops directly into CorrDiff's
existing dataset loader with zero code changes.

**Key numbers:**
- 200,000 training samples (configurable)
- Input: 26-channel ERA5 patches (8×8 pixels)
- Output: 9-channel HRRR patches (64×64 pixels)
- Time range: 2015–2025
- Sampling: ocean pixels only (ERA5 land-sea mask < 0.2)

**GitHub repo:**
```
https://github.com/sranjbar94/DataPipeline_for_regional_hrrr_train_dataset.git
```

---

## 2. Architecture & Data Flow

```
ERA5 (0.25°, global)              HRRR (3 km, CONUS)
   8×8 patch → [input]               64×64 patch → [output]
         │                                   │
         └──────── sample (lat, lon, t) ─────┘
                          │
              regional_hrrr_train_dataset.nc
                          │
                    Transfer to HPC
                          │
                  CorrDiff Training
```

### Three Pipeline Stages

| Stage | Script Command | What It Does |
|-------|---------------|--------------|
| 1 — Download | `python run_pipeline.py download` | Downloads ERA5 daily NetCDF files from Copernicus CDS |
| 2 — Build | `python run_pipeline.py build` | Samples ERA5+HRRR patches, writes output NetCDF |
| 3 — Stats | `python run_pipeline.py stats` | Computes per-channel mean/std for CorrDiff normalization |

### Data Sources

| Source | Resolution | Access | Format |
|--------|-----------|--------|--------|
| ERA5 single-level | 0.25° (~25 km) | CDS API (requires free key) | NetCDF |
| ERA5 pressure-level | 0.25° (~25 km) | CDS API (requires free key) | NetCDF |
| HRRR surface | ~3 km | AWS S3 (anonymous, free) | GRIB2 |

---

## 3. Repository Structure

```
DataPipeline_for_regional_hrrr_train_dataset/
├── run_pipeline.py               # CLI entry point — dispatches all stages
├── environment.yml               # Conda environment specification
├── pyproject.toml                # Python package config (editable install)
├── configs/
│   └── pipeline_config.yaml      # ALL parameters — edit here, not in code
├── src/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── era5_downloader.py    # Stage 1: CDS API bulk downloader
│   │   ├── era5_reader.py        # Stage 2: ERA5 file reader + patch extractor
│   │   ├── hrrr_fetcher.py       # Stage 2: HRRR S3 streamer + patch extractor
│   │   ├── dataset_builder.py    # Stage 2: main sampling orchestrator
│   │   ├── nc_writer.py          # Stage 2: NetCDF writer (HRRR-Mini schema)
│   │   └── compute_stats.py      # Stage 3: stats.json generator
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # YAML config loader
│       ├── logger.py             # Shared logging setup
│       └── time_sampler.py       # Random timestamp sampler with resume
├── scripts/
│   ├── smoke_test.sh             # Quick validation (no downloads needed)
│   └── transfer_to_hpc.sh        # rsync to Yale Bouchet HPC
├── tests/
│   ├── test_nc_schema.py         # NetCDF schema compliance tests
│   └── test_time_sampler.py      # Timestamp sampler unit tests
├── notebooks/                    # Exploratory notebooks (not tracked)
├── docs/
│   └── TUTORIAL.md               # This file
└── data/                         # Created at runtime (gitignored)
    ├── era5_raw/
    │   ├── single_level/         # era5_sl_YYYYMMDD.nc files
    │   └── pressure_level/       # era5_pl_YYYYMMDD.nc files
    ├── output/
    │   └── regional_hrrr_train_dataset.nc
    └── checkpoints/
        └── build_checkpoint.json
```

---

## 4. File-by-File Reference

### `run_pipeline.py`
- **Role**: Single CLI entry point for all three stages
- **Usage**: `python run_pipeline.py {download,build,stats,all} [options]`
- **Key CLI flags**:
  - `--test_mode`: 10 days only (2015-01-01 to 2015-01-10)
  - `--dry_run`: synthetic HRRR data (no S3 calls)
  - `--workers N`: parallel CDS API workers
  - `--n_samples N`: override sample count
  - `--samples_per_ts N`: patches per timestamp
  - `--skip_pressure`: skip pressure-level ERA5 download
  - `--date_start / --date_end`: override date range

### `configs/pipeline_config.yaml`
- **Role**: Central configuration — all parameters live here
- **Key sections**:
  - `time`: date range (2015-01-01 to 2025-12-31)
  - `domain`: CONUS bounding box (lat 18–56, lon -140 to -55)
  - `sampling`: n_samples (200000), samples_per_ts (50), random_seed
  - `ocean`: lsm_threshold (0.2), min_ocean_frac (0.6)
  - `patches`: era5_patch_size (8), hrrr_patch_size (64)
  - `era5`: variable lists for single-level and pressure-level
  - `hrrr`: forecast_hour (1), variable mapping
  - `storage`: directory paths, checkpoint frequency

### `src/pipeline/era5_downloader.py`
- **Role**: Stage 1 — downloads ERA5 data from CDS API
- **Design**: One CDS request per day (avoids 2 GB volume cap)
- **Key feature**: Splits instantaneous and accumulated variables into
  separate CDS requests, then merges with xarray. This prevents CDS from
  returning ZIP files instead of NetCDF.
- **Accumulated vars**: ssrd, strd, tp, sf
- **Instantaneous vars**: t2m, d2m, sp, msl, u10, v10, lsm, tcwv
- **Resume**: Skips files that already exist on disk

### `src/pipeline/era5_reader.py`
- **Role**: Stage 2 — reads ERA5 daily files, extracts 8×8 patches
- **Key features**:
  - LRU cache (max 4 days in memory)
  - Auto-unzips if CDS returned a ZIP instead of NetCDF
  - Derives specific humidity `q` from `d2m` and `sp`
  - `get_ocean_patch_centers()`: scans LSM to find valid ocean patches
  - `extract_patch()`: returns dict of all ERA5 variables for one patch

### `src/pipeline/hrrr_fetcher.py`
- **Role**: Stage 2 — fetches HRRR GRIB2 from AWS S3, extracts 64×64 patches
- **Data source**: `s3://noaa-hrrr-bdp-pds` (anonymous access)
- **Key design**:
  - Downloads full GRIB2 to temp file, opens with cfgrib
  - Flattens all 2D variables into a lookup dict
  - Matches output channel names to HRRR variable names
  - Returns `None` if HRRR file unavailable (sample is skipped)
- **Variable mapping** (our name → HRRR cfgrib name):
  - `2t` → `t2m`, `10u` → `u10`, `10v` → `v10`
  - `sp` → `sp`, `q` → `sh2`, `ssrd` → `sdswrf`
  - `tp` → `tp`, `sf` → `sdwe`
  - `strd` → NOT IN HRRR (filled from ERA5)

### `src/pipeline/dataset_builder.py`
- **Role**: Stage 2 — main sampling loop orchestrator
- **Key features**:
  - Draws random timestamps, picks ocean patch centres
  - Extracts ERA5 patch + HRRR patch per sample
  - Fills missing HRRR `strd` from ERA5 (bilinear interpolation 8×8 → 64×64)
  - Dry-run mode: synthetic data for both ERA5 and HRRR
  - Checkpointing every N samples (configurable)
  - Resume from checkpoint on restart

### `src/pipeline/nc_writer.py`
- **Role**: Stage 2 — writes output NetCDF in HRRR-Mini schema
- **Groups**: `input` (ERA5 LR), `output` (HRRR HR), `invariant` (static)
- **Input channels** (26 total): 12 surface + 20 pressure-level (u/v/z/t/q × 4 levels) - some overlap
- **Output channels** (9): 2t, 10u, 10v, tp, ssrd, strd, sp, q, sf

### `src/pipeline/compute_stats.py`
- **Role**: Stage 3 — computes per-channel mean/std → `stats.json`
- **Used by**: CorrDiff training for input/output normalization

### `src/utils/config.py`
- **Role**: Loads `pipeline_config.yaml` into a nested namespace object
- **Access pattern**: `cfg.sampling.n_samples`, `cfg.domain.lat_min`, etc.

### `src/utils/logger.py`
- **Role**: Shared logger with consistent formatting
- **Format**: `2026-04-13 15:00:00 [INFO] module_name — message`

### `src/utils/time_sampler.py`
- **Role**: Draws random (day, hour) pairs without replacement
- **Resume**: Accepts pre-used timestamps from checkpoint; fast-forwards RNG

---

## 5. Environment Setup

### On any machine (local, HPC, cloud):

```bash
# 1. Clone the repo
git clone https://github.com/sranjbar94/DataPipeline_for_regional_hrrr_train_dataset.git
cd DataPipeline_for_regional_hrrr_train_dataset

# 2. Create conda environment
conda env create -f environment.yml
conda activate corrdiff_pipeline

# 3. Install the package in editable mode (enables `from src.xxx` imports)
pip install -e .
```

### Key dependencies (from environment.yml):
- Python 3.11
- xarray, netCDF4, numpy, scipy
- cdsapi (ERA5 download)
- cfgrib, eccodes (HRRR GRIB2 parsing)
- s3fs (AWS S3 anonymous access)
- pyyaml (config loading)
- pytest (testing)

### If conda is not available (e.g., Docker, bare HPC):
```bash
pip install xarray netCDF4 numpy scipy cdsapi cfgrib eccodes s3fs pyyaml pytest
pip install -e .
```

---

## 6. CDS API Credentials

ERA5 downloads require a free Copernicus CDS account.

### Step 1: Register
Go to https://cds.climate.copernicus.eu and create a free account.

### Step 2: Get your API key
After login, go to https://cds.climate.copernicus.eu/how-to-api

### Step 3: Create credentials file
```bash
cat > ~/.cdsapirc << 'EOF'
url: https://cds.climate.copernicus.eu/api
key: YOUR-UID:YOUR-API-KEY
EOF
chmod 600 ~/.cdsapirc
```

### Step 4: Accept the license
You must accept the ERA5 data license on the CDS website before downloading.
Visit: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
Click "Download" and accept the terms.

### On HPC (no browser):
Create `~/.cdsapirc` on the HPC with the same key from your local machine.

---

## 7. Quick Start — Smoke Test

This validates the full pipeline with synthetic data. No downloads needed.

```bash
conda activate corrdiff_pipeline
bash scripts/smoke_test.sh
```

**What it does:**
1. Builds 50 samples using synthetic ERA5 + HRRR data
2. Inspects the output NetCDF (dimensions, groups, variables)
3. Prints projected size for the full 200k dataset

**Expected output:**
```
Build complete: 50 / 50 samples
File   : data/output/regional_hrrr_train_dataset.nc
Size   : 6.9 MB  for 50 samples
Groups : ['input', 'output', 'invariant']
[input]  33 variables
[output]  9 variables
Projected full dataset (200k samples): ~28 GB
Smoke test PASSED.
```

### Run unit tests:
```bash
pytest tests/test_nc_schema.py tests/test_time_sampler.py -v
```
All 9 tests should pass.

---

## 8. Stage 1 — ERA5 Download

### Test download (10 days):
```bash
python run_pipeline.py download --test_mode --workers 2
```

This downloads 2015-01-01 to 2015-01-10 (20 files: 10 single-level + 10 pressure-level).
Takes ~5–10 minutes.

### Full download (2015–2025):
```bash
python run_pipeline.py download --workers 2
```

- ~3,652 days × 2 request types = ~7,304 CDS API calls
- Takes 3–7 days depending on CDS queue
- Keep workers ≤ 4 to respect CDS fair-use limits

### Skip pressure levels (faster, smaller):
```bash
python run_pipeline.py download --workers 2 --skip_pressure
```

### Custom date range:
```bash
python run_pipeline.py download --date_start 2020-01-01 --date_end 2020-12-31 --workers 2
```

### Resume after interruption:
Just re-run the same command. Existing files are automatically skipped.

### How the download works internally:
1. For each day, makes **two** CDS requests:
   - Instantaneous variables: t2m, d2m, sp, msl, u10, v10, lsm, tcwv
   - Accumulated variables: ssrd, strd, tp, sf
2. Merges both into a single NetCDF: `era5_sl_YYYYMMDD.nc`
3. This split prevents CDS from returning ZIP archives

### Output:
```
data/era5_raw/
├── single_level/
│   ├── era5_sl_20150101.nc   (~40 MB each)
│   ├── era5_sl_20150102.nc
│   └── ...
├── pressure_level/
│   ├── era5_pl_20150101.nc   (~45 MB each)
│   ├── era5_pl_20150102.nc
│   └── ...
└── download_results.json
```

---

## 9. Stage 2 — Dataset Build

### Test build with real data (10 samples):
```bash
rm -f data/output/regional_hrrr_train_dataset.nc
rm -f data/checkpoints/build_checkpoint.json
python run_pipeline.py build --test_mode --n_samples 10 --samples_per_ts 5
```
Takes ~3–5 minutes (HRRR S3 downloads are slow, ~30–60s per timestamp).

### Test build with synthetic HRRR (fast):
```bash
rm -f data/output/regional_hrrr_train_dataset.nc
rm -f data/checkpoints/build_checkpoint.json
python run_pipeline.py build --test_mode --dry_run
```
Takes <1 second for 200 samples.

### Full production build:
```bash
python run_pipeline.py build
```
- 200,000 samples × ~30s per HRRR fetch = many hours
- Run overnight or on HPC with `nohup` or `tmux`

### How the build works internally:
1. Load ERA5 land-sea mask → find valid ocean patch centres
2. For each iteration:
   a. Draw random timestamp from 2015–2025
   b. Pick N random ocean centres
   c. Extract ERA5 8×8 patch from local disk
   d. Fetch HRRR 64×64 patch from AWS S3
   e. If HRRR missing `strd`, fill from ERA5 (bilinear interpolation)
   f. Write sample to output NetCDF
3. Checkpoint every 1000 samples

### Resume after interruption:
Just re-run `python run_pipeline.py build`. It reads the checkpoint and
continues from where it left off.

### Important: Clean start
If you want to start fresh (different parameters, etc.):
```bash
rm -f data/output/regional_hrrr_train_dataset.nc
rm -f data/checkpoints/build_checkpoint.json
```

---

## 10. Stage 3 — Compute Stats

```bash
python run_pipeline.py stats
```

Reads the output NetCDF and computes per-channel mean and standard deviation.
Writes `data/output/stats.json`. This file is required by CorrDiff for
input/output normalization during training.

---

## 11. Full Production Run

```bash
# Stage 1 — Download ERA5 (3–7 days)
python run_pipeline.py download --workers 2

# Stage 2 — Build dataset (run overnight / on HPC)
python run_pipeline.py build

# Stage 3 — Compute stats
python run_pipeline.py stats

# Or run all three sequentially:
python run_pipeline.py all
```

### Running in background on HPC:
```bash
# Using nohup
nohup python run_pipeline.py download --workers 2 > logs/download.log 2>&1 &

# Using tmux (recommended)
tmux new -s pipeline
python run_pipeline.py download --workers 2
# Ctrl+B, then D to detach
# tmux attach -t pipeline to reconnect
```

---

## 12. Deploying on Yale Bouchet HPC

### Step 1: SSH into Bouchet
```bash
ssh sr2723@bouchet.hpc.yale.edu
```

### Step 2: Clone and setup
```bash
cd ~/scratch_pi_ey239/sr2723
git clone https://github.com/sranjbar94/DataPipeline_for_regional_hrrr_train_dataset.git
cd DataPipeline_for_regional_hrrr_train_dataset

module load miniconda
conda env create -f environment.yml
conda activate corrdiff_pipeline
pip install -e .
```

### Step 3: CDS API key
```bash
cat > ~/.cdsapirc << 'EOF'
url: https://cds.climate.copernicus.eu/api
key: YOUR-UID:YOUR-API-KEY
EOF
chmod 600 ~/.cdsapirc
```

### Step 4: Run smoke test
```bash
bash scripts/smoke_test.sh
pytest tests/ -v
```

### Step 5: Submit download job (SLURM)
Create `scripts/slurm_download.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=era5_download
#SBATCH --partition=day
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/download_%j.log

module load miniconda
conda activate corrdiff_pipeline
cd ~/scratch_pi_ey239/sr2723/DataPipeline_for_regional_hrrr_train_dataset

python run_pipeline.py download --workers 2
```

```bash
mkdir -p logs
sbatch scripts/slurm_download.sh
```

### Step 6: Submit build job
Create `scripts/slurm_build.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=corrdiff_build
#SBATCH --partition=week
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/build_%j.log

module load miniconda
conda activate corrdiff_pipeline
cd ~/scratch_pi_ey239/sr2723/DataPipeline_for_regional_hrrr_train_dataset

python run_pipeline.py build
python run_pipeline.py stats
```

```bash
sbatch scripts/slurm_build.sh
```

### Step 7: Link output to CorrDiff
```bash
cd ~/physicsnemo/examples/weather/corrdiff
mkdir -p data/regional_hrrr
ln -s ~/scratch_pi_ey239/sr2723/DataPipeline_for_regional_hrrr_train_dataset/data/output/regional_hrrr_train_dataset.nc \
      data/regional_hrrr/regional_hrrr_train_dataset.nc
ln -s ~/scratch_pi_ey239/sr2723/DataPipeline_for_regional_hrrr_train_dataset/data/output/stats.json \
      data/regional_hrrr/stats.json
```

### Alternative: Transfer from local machine
If you built the dataset locally:
```bash
# On your local machine:
bash scripts/transfer_to_hpc.sh
```

---

## 13. Deploying on Other HPC / Cloud

### Generic Linux / Cloud VM:

```bash
# Install miniconda (if not available)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
export PATH="$HOME/miniconda3/bin:$PATH"

# Clone and setup
git clone https://github.com/sranjbar94/DataPipeline_for_regional_hrrr_train_dataset.git
cd DataPipeline_for_regional_hrrr_train_dataset
conda env create -f environment.yml
conda activate corrdiff_pipeline
pip install -e .

# CDS API key
cat > ~/.cdsapirc << 'EOF'
url: https://cds.climate.copernicus.eu/api
key: YOUR-UID:YOUR-API-KEY
EOF

# Verify
bash scripts/smoke_test.sh
pytest tests/ -v
```

### AWS EC2:
- Instance type: `m5.xlarge` or larger (4 vCPU, 16 GB RAM)
- Storage: 200 GB EBS for ERA5, 500 GB for full dataset
- S3 access to HRRR is fast from us-east-1

### Google Cloud:
- Same as above, `n2-standard-4` or larger
- HRRR S3 access works globally (anonymous)

### Docker:
```dockerfile
FROM continuumio/miniconda3
WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml
COPY . .
RUN conda run -n corrdiff_pipeline pip install -e .
ENTRYPOINT ["conda", "run", "-n", "corrdiff_pipeline"]
CMD ["bash", "scripts/smoke_test.sh"]
```

---

## 14. Output NetCDF Schema

The output file matches the HRRR-Mini schema exactly:

```
regional_hrrr_train_dataset.nc
├── Dimensions
│   sample   = 200000
│   y_lr     = 8        (ERA5 patch height)
│   x_lr     = 8        (ERA5 patch width)
│   y_hr     = 64       (HRRR patch height)
│   x_hr     = 64       (HRRR patch width)
│   coord    = 2        (lat_idx, lon_idx)
│
├── Root variables
│   time(sample)          float64  — hours since 2015-01-01
│   coord(sample, coord)  int32    — ERA5 grid indices of patch centre
│
├── Group [input]  — ERA5 low-resolution channels
│   Shape: (sample, y_lr=8, x_lr=8)
│   Surface (13): u10, v10, t2m, tcwv, sp, msl, d2m, q, ssrd, strd, tp, sf, lsm
│   Pressure-level (20): {u,v,z,t,q} × {1000,850,500,250} hPa
│   Total: 33 variables
│
├── Group [output]  — HRRR high-resolution targets
│   Shape: (sample, y_hr=64, x_hr=64)
│   Channels (9): 2t, 10u, 10v, tp, ssrd, strd, sp, q, sf
│
└── Group [invariant]  — static fields (written once)
    Shape: (y_lr=8, x_lr=8) or scalar
    Fields: latitude, longitude, elev_mean, lsm_mean
```

---

## 15. Variable Mapping Reference

### Input (ERA5 → NetCDF)

| ERA5 CDS Name | Short Name | Description |
|---------------|-----------|-------------|
| 2m_temperature | t2m | 2m air temperature (K) |
| 2m_dewpoint_temperature | d2m | 2m dewpoint (K) |
| surface_pressure | sp | Surface pressure (Pa) |
| mean_sea_level_pressure | msl | Sea level pressure (Pa) |
| 10m_u_component_of_wind | u10 | 10m zonal wind (m/s) |
| 10m_v_component_of_wind | v10 | 10m meridional wind (m/s) |
| surface_solar_radiation_downwards | ssrd | Downward SW (J/m²) |
| surface_thermal_radiation_downwards | strd | Downward LW (J/m²) |
| total_precipitation | tp | Total precip (m) |
| snowfall | sf | Snowfall (m water equiv) |
| land_sea_mask | lsm | 0=ocean, 1=land |
| total_column_water_vapour | tcwv | Column water vapour (kg/m²) |
| Derived: specific_humidity | q | From d2m + sp |

### Output (HRRR → NetCDF)

| Output Channel | HRRR cfgrib Name | Source | Description |
|---------------|-----------------|--------|-------------|
| 2t | t2m | HRRR | 2m temperature (K) |
| 10u | u10 | HRRR | 10m zonal wind (m/s) |
| 10v | v10 | HRRR | 10m meridional wind (m/s) |
| tp | tp | HRRR fxx=1 | Total precip (kg/m²) |
| ssrd | sdswrf | HRRR | Downward shortwave (W/m²) |
| strd | N/A | ERA5 interpolated | Downward longwave (J/m²) |
| sp | sp | HRRR | Surface pressure (Pa) |
| q | sh2 | HRRR | 2m specific humidity (kg/kg) |
| sf | sdwe | HRRR | Snow water equivalent (kg/m²) |

### Pressure-Level Input Variables

| Variable | Levels (hPa) | Keys in NetCDF |
|----------|-------------|----------------|
| u (zonal wind) | 1000, 850, 500, 250 | u1000, u850, u500, u250 |
| v (meridional wind) | 1000, 850, 500, 250 | v1000, v850, v500, v250 |
| z (geopotential) | 1000, 850, 500, 250 | z1000, z850, z500, z250 |
| t (temperature) | 1000, 850, 500, 250 | t1000, t850, t500, t250 |
| q (specific humidity) | 1000, 850, 500, 250 | q1000, q850, q500, q250 |

---

## 16. Known Limitations & Design Decisions

### HRRR does not provide downward longwave radiation (DLWRF)
- **Impact**: `strd` output channel cannot come from HRRR
- **Solution**: Filled from ERA5 `strd`, bilinearly interpolated from 8×8 to 64×64
- **Justification**: Longwave radiation varies smoothly at ~200 km scale
- **Code location**: `src/pipeline/dataset_builder.py`, around the `_unzip_if_needed` section

### HRRR fxx=0 has zero precipitation and no radiation fluxes
- **Solution**: Use `fxx=1` (1-hour forecast) instead of `fxx=0` (analysis)
- **Config**: `configs/pipeline_config.yaml` → `hrrr.forecast_hour: 1`

### CDS API returns ZIP files when mixing variable types
- **Solution**: Split single-level downloads into two requests (instantaneous + accumulated), then merge
- **Code location**: `src/pipeline/era5_downloader.py`

### CDS API daily request strategy
- One request per day (all 24 hours) to stay under the ~2 GB per-request cap
- Monthly requests over CONUS domain exceed the limit

### HRRR coverage
- HRRR only covers CONUS. Ocean patches near domain edges may fail
- These failures are logged as warnings and skipped; the build continues

### HRRR S3 download speed
- Each HRRR GRIB2 file is ~100+ MB
- ~30–60 seconds per timestamp to download + parse
- Full 200k build takes many hours; run on HPC

---

## 17. Troubleshooting

### `FileNotFoundError: ERA5 single-level file not found`
→ Run `python run_pipeline.py download` first

### `OSError: Unknown file format` (when reading ERA5)
→ The file is a ZIP, not NetCDF. The reader auto-unzips, but if it persists:
```bash
rm data/era5_raw/single_level/era5_sl_YYYYMMDD.nc
python run_pipeline.py download --date_start YYYY-MM-DD --date_end YYYY-MM-DD
```

### `KeyError: land_sea_mask / lsm not found`
→ The ERA5 file only has accumulated variables (ZIP issue). Re-download:
```bash
rm data/era5_raw/single_level/era5_sl_*.nc
python run_pipeline.py download --test_mode --workers 2
```

### `NetCDF: Index exceeds dimension bound`
→ Old checkpoint from a smaller build. Clean up:
```bash
rm -f data/output/regional_hrrr_train_dataset.nc
rm -f data/checkpoints/build_checkpoint.json
```

### `ModuleNotFoundError: No module named 'src'`
→ Install the package:
```bash
pip install -e .
```

### `Request too large` from CDS
→ The downloader already handles this by splitting by day. If it still happens, reduce the domain in `pipeline_config.yaml`.

### HRRR fetch failures (warnings during build)
→ Normal. Some HRRR files don't exist (maintenance, early years). The build skips failed samples and continues.

### `strd` shows ERA5-scale values in HRRR output
→ Expected. HRRR doesn't have DLWRF; we fill from ERA5. The values are in J/m² (accumulated), which is the ERA5 convention.

---

## 18. Making Edits — LLM Context Guide

This section helps an LLM understand the codebase for making changes.

### To change the number of samples:
Edit `configs/pipeline_config.yaml`:
```yaml
sampling:
  n_samples: 200000  # change this
```
Or use CLI: `python run_pipeline.py build --n_samples 50000`

### To change the date range:
Edit `configs/pipeline_config.yaml`:
```yaml
time:
  date_start: "2015-01-01"
  date_end: "2025-12-31"
```
Or use CLI: `--date_start 2020-01-01 --date_end 2020-12-31`

### To add a new ERA5 input variable:
1. Add the CDS variable name to `configs/pipeline_config.yaml` under `era5.single_level`
2. Add the short name mapping to `src/pipeline/era5_reader.py` in `_SL_RENAME`
3. Add the channel name to `src/pipeline/nc_writer.py` in `_all_input_channels()`

### To add a new HRRR output variable:
1. Add mapping to `src/pipeline/hrrr_fetcher.py` in `_HRRR_VAR_MAP`
2. Add channel name to `src/pipeline/nc_writer.py` in `_OUTPUT_CHANNELS`
3. Add synthetic range to `src/pipeline/dataset_builder.py` in `_synthetic_era5_patch()`

### To change the patch sizes:
Edit `configs/pipeline_config.yaml`:
```yaml
patches:
  era5_patch_size: 8      # input LR patch
  hrrr_patch_size: 64     # output HR patch
  era5_resolution: 0.25   # degrees per ERA5 pixel
```

### To change the ocean mask threshold:
Edit `configs/pipeline_config.yaml`:
```yaml
ocean:
  lsm_threshold: 0.2      # ERA5 lsm < this = ocean
  min_ocean_frac: 0.6      # fraction of patch that must be ocean
```

### To change the HRRR forecast hour:
Edit `configs/pipeline_config.yaml`:
```yaml
hrrr:
  forecast_hour: 1   # 0=analysis, 1=1hr forecast (recommended)
```

### Import structure:
All files use `from src.pipeline.xxx import ...` or `from src.utils.xxx import ...`.
The `pyproject.toml` with `pip install -e .` makes this work.
The `[tool.pytest.ini_options] pythonpath = ["."]` in `pyproject.toml` makes tests work.

### Config access pattern:
```python
from src.utils.config import load_config
cfg = load_config("configs/pipeline_config.yaml")
cfg.sampling.n_samples    # 200000
cfg.domain.lat_min        # 18.0
cfg.era5.single_level     # list of variable names
```

### Data flow summary:
```
run_pipeline.py
  → cmd_download() → era5_downloader.run_downloader()
  → cmd_build()    → DatasetBuilder.run()
                       → ERA5Reader.get_ocean_patch_centers()
                       → ERA5Reader.extract_patch()
                       → fetch_hrrr_patch()
                       → NCWriter.write_sample()
  → cmd_stats()    → compute_channel_stats()
```

---

## 19. Testing

### Run all tests:
```bash
pytest tests/ -v
```

### Test descriptions:

| Test | What it validates |
|------|-------------------|
| `test_dimensions` | NetCDF has correct dimensions (sample, y_lr, x_lr, y_hr, x_hr) |
| `test_groups_exist` | NetCDF has input, output, invariant groups |
| `test_output_channels` | All 9 output channels present |
| `test_time_encoding` | Time variable uses correct epoch (2015-01-01) |
| `test_no_nan_in_written_data` | Written data has no NaN values |
| `test_invariant_written_once` | Invariant group written correctly |
| `test_no_repeats` | Timestamp sampler doesn't repeat |
| `test_within_range` | All timestamps within configured date range |
| `test_resume_from_used` | Sampler correctly resumes from checkpoint |

### Adding new tests:
Put them in `tests/test_*.py`. Pytest discovers them automatically.

---

## 20. Storage Estimates

| Item | Approx. Size |
|------|-------------|
| ERA5 single-level (2015–2025, CONUS) | 35–55 GB |
| ERA5 pressure-level (2015–2025, CONUS) | 70–110 GB |
| Output NetCDF (200k samples) | ~28 GB (projected from smoke test) |
| stats.json | < 1 MB |
| **Total disk needed** | **~200 GB** |

### Check projected size:
```bash
bash scripts/smoke_test.sh
# Last line prints: "Projected full dataset (200k samples): ~XX GB"
```

---

## Appendix: Quick Command Reference

```bash
# Setup
git clone https://github.com/sranjbar94/DataPipeline_for_regional_hrrr_train_dataset.git
cd DataPipeline_for_regional_hrrr_train_dataset
conda env create -f environment.yml
conda activate corrdiff_pipeline
pip install -e .

# Smoke test
bash scripts/smoke_test.sh

# Unit tests
pytest tests/ -v

# Test download (10 days)
python run_pipeline.py download --test_mode --workers 2

# Test build (10 samples, real data)
python run_pipeline.py build --test_mode --n_samples 10 --samples_per_ts 5

# Test build (fast, synthetic HRRR)
python run_pipeline.py build --test_mode --dry_run

# Full production run
python run_pipeline.py download --workers 2
python run_pipeline.py build
python run_pipeline.py stats

# Or all at once
python run_pipeline.py all

# Clean start
rm -f data/output/regional_hrrr_train_dataset.nc
rm -f data/checkpoints/build_checkpoint.json

# Transfer to HPC
bash scripts/transfer_to_hpc.sh
```

---

*Last updated: April 2026*
*Pipeline version: 0.1.0*
