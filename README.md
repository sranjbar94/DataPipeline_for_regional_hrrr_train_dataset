# DataPipeline_for_regional_hrrr_train_dataset

A two-stage data pipeline that builds a regional CorrDiff training dataset matching the HRRR-Mini schema used by NVIDIA PhysicsNeMo CorrDiff.

The pipeline downscales ERA5 reanalysis (0.25°, ~25 km) to HRRR operational data (~3 km), sampling ocean pixels within the CONUS/HRRR domain. The output NetCDF drops directly into CorrDiff's existing dataset loader without any code changes.

## Overview

```
ERA5 (0.25°, global)          HRRR (3 km, CONUS)
   8×8 patch → [input]            64×64 patch → [output]
         │                                │
         └──────── sample (lat, lon, t) ──┘
                          │
              regional_hrrr_train_dataset.nc
                          │
                    Transfer to HPC
                          │
                  CorrDiff Training
```

**Input (ERA5 LR):** 26 channels — surface winds, temperature, pressure, radiation, precipitation, humidity + pressure-level fields at 1000/850/500/250 hPa.

**Output (HRRR HR):** 9 channels matching the 9 Oceananigans surface forcing variables:

| # | Variable | ERA5 field | HRRR field |
|---|----------|-----------|------------|
| 1 | Surface air temperature | t2m | TMP_2maboveground |
| 2 | Sea level pressure | msl | PRES_surface |
| 3 | Specific humidity | derived (d2m + sp) | SPFH_2maboveground |
| 4 | Zonal wind | u10 | UGRD_10maboveground |
| 5 | Meridional wind | v10 | VGRD_10maboveground |
| 6 | Downward shortwave radiation | ssrd | DSWRF_surface |
| 7 | Downward longwave radiation | strd | DLWRF_surface |
| 8 | Rainfall rate | tp | APCP_surface |
| 9 | Snowfall rate | sf | WEASD_surface |

**Sampling:** 200,000 samples, ocean pixels only (ERA5 land-sea mask < 0.2), randomly drawn from 2015–2025.

## ERA5 download strategy — daily requests

The ERA5 downloader requests data one day at a time (all 24 hours per request). This is intentional: the CDS API enforces a per-request volume cap (~2 GB), and a single monthly request over the full CONUS domain exceeds that limit, causing `Request too large` errors. Splitting by day keeps each request well within the cap.

Each daily file is ~50–150 MB. The downloader skips files that already exist, so interrupted runs resume safely. On failure it retries up to 3 times with exponential back-off before logging the task as failed.

## Repository structure

```
DataPipeline_for_regional_hrrr_train_dataset/
├── run_pipeline.py               # single CLI entry point for all stages
├── environment.yml               # reproducible conda environment
├── configs/
│   └── pipeline_config.yaml      # all parameters — edit here, not in code
├── src/
│   ├── pipeline/
│   │   ├── era5_downloader.py    # Stage 1: bulk ERA5 download via CDS API
│   │   ├── era5_reader.py        # Stage 2: lazy ERA5 patch reader
│   │   ├── hrrr_fetcher.py       # Stage 2: HRRR patch streamer from AWS S3
│   │   ├── dataset_builder.py    # Stage 2: main sampling loop
│   │   ├── nc_writer.py          # Stage 2: NetCDF writer (HRRR-Mini schema)
│   │   └── compute_stats.py      # Stage 3: stats.json for CorrDiff normalization
│   └── utils/
│       ├── config.py             # YAML config loader
│       ├── logger.py             # shared logger
│       └── time_sampler.py       # random timestamp sampler with resume support
├── scripts/
│   ├── smoke_test.sh             # quick dry-run validation (no downloads needed)
│   └── transfer_to_hpc.sh        # rsync final NetCDF to Yale Bouchet HPC
├── tests/
│   ├── test_nc_schema.py         # NetCDF schema compliance tests
│   └── test_time_sampler.py      # timestamp sampler unit tests
├── notebooks/                    # exploratory notebooks (not tracked in git)
└── docs/                         # additional documentation
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/sranjbar94/DataPipeline_for_regional_hrrr_train_dataset.git
cd DataPipeline_for_regional_hrrr_train_dataset
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate corrdiff_pipeline
```

### 3. Verify your CDS API key

Your `~/.cdsapirc` file should look like:

```
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR-UID:YOUR-API-KEY
```

Get your key at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu).

## Quick start — local dry run (no downloads)

Test the full pipeline end-to-end with synthetic data in under a minute:

```bash
bash scripts/smoke_test.sh
```

This generates 50 samples using synthetic HRRR patches (no S3 calls, no ERA5 needed) and prints the projected size of the full 200k dataset.

## Recommended workflow

### Step 1 — Test run (10 days in Jan 2015)

Run this first to confirm your CDS credentials, S3 access, and file paths all work before committing to the full 10-year download.

```bash
# Download 10 days of ERA5 (2015-01-01 to 2015-01-10)
python run_pipeline.py download --test_mode --workers 2

# Build ~200 samples from those 10 days (dry_run = synthetic HRRR, no S3)
python run_pipeline.py build --test_mode --dry_run

# Or build with real HRRR data from S3:
python run_pipeline.py build --test_mode

# Compute stats on the test dataset
python run_pipeline.py stats
```

If all three stages complete without errors, you are ready for the full run.

### Step 2 — Full 10-year run (2015–2025)

```bash
# Stage 1 — Download ERA5 daily files for 2015-2025
# ~3,652 days × 2 request types = ~7,304 CDS API calls
# Keep workers <= 4 to respect CDS fair-use limits
python run_pipeline.py download --workers 2

# Skip pressure-level variables for a faster/smaller download:
python run_pipeline.py download --workers 2 --skip_pressure

# Stage 2 — Build 200k-sample dataset (run overnight or on HPC)
python run_pipeline.py build

# Stage 3 — Compute stats.json for CorrDiff normalization
python run_pipeline.py stats

# Or run all three stages sequentially:
python run_pipeline.py all
```

The download is resumable — re-running `download` skips files that already exist. The build is also resumable via the checkpoint at `data/checkpoints/build_checkpoint.json`.

## Additional CLI options

```bash
# Override date range for a custom partial download or build
python run_pipeline.py download --date_start 2020-01-01 --date_end 2020-03-31
python run_pipeline.py build    --date_start 2020-01-01 --date_end 2020-03-31

# Quick build sanity check (500 samples, no S3)
python run_pipeline.py build --dry_run --n_samples 500

# Quick build with real HRRR
python run_pipeline.py build --n_samples 500 --samples_per_ts 5
```

## Configuration

All parameters are in `configs/pipeline_config.yaml`. Key settings:

```yaml
sampling:
  n_samples: 200000        # total samples
  samples_per_ts: 50       # ocean patches per timestamp

time:
  date_start: "2015-01-01"
  date_end:   "2025-12-31"

ocean:
  lsm_threshold: 0.2       # ERA5 lsm < 0.2 → ocean pixel
  min_ocean_frac: 0.6      # ≥60% of patch must be ocean
```

## Output NetCDF schema

The output file matches the HRRR-Mini schema exactly:

```
regional_hrrr_train_dataset.nc
├── Dimensions: sample=200000, y_lr=8, x_lr=8, y_hr=64, x_hr=64, coord=2
├── time(sample)         — hours since 2015-01-01
├── coord(sample, coord) — ERA5 grid indices of patch centre
├── Group [input]        — ERA5 LR channels  (sample, y_lr=8, x_lr=8)
│     u10, v10, t2m, msl, sp, d2m, q, ssrd, strd, tp, sf, tcwv, lsm
│     + pressure-level: u/v/z/t/q at 1000, 850, 500, 250 hPa
├── Group [output]       — HRRR HR targets  (sample, y_hr=64, x_hr=64)
│     2t, 10u, 10v, tp, ssrd, strd, sp, q, sf
└── Group [invariant]    — static fields    (y_lr=8, x_lr=8)
      latitude, longitude, elev_mean, lsm_mean
```

## Transfer to HPC

Once the dataset is built locally, transfer it to Yale Bouchet:

```bash
bash scripts/transfer_to_hpc.sh
```

Then on the HPC, symlink it into the CorrDiff data directory:

```bash
cd ~/physicsnemo/examples/weather/corrdiff
mkdir -p data/regional_hrrr
ln -s ~/scratch_pi_ey239/sr2723/corrdiff/data/regional_hrrr_train_dataset.nc \
      data/regional_hrrr/regional_hrrr_train_dataset.nc
ln -s ~/scratch_pi_ey239/sr2723/corrdiff/data/stats.json \
      data/regional_hrrr/stats.json
```

## Running tests

```bash
pytest tests/test_nc_schema.py tests/test_time_sampler.py -v
```

The test suite validates the NetCDF schema, time encoding, and timestamp sampler without requiring any downloaded data.

## Storage estimates

| Item | Approx. size |
|------|-------------|
| ERA5 single-level (2015–2025, CONUS, daily files) | 35–55 GB |
| ERA5 pressure-level (2015–2025, CONUS, daily files) | 70–110 GB |
| regional_hrrr_train_dataset.nc (200k samples) | 100–400 GB |
| stats.json | < 1 MB |

Run `bash scripts/smoke_test.sh` first — it prints the projected size based on 50 actual samples.

## Related

- [NVIDIA PhysicsNeMo CorrDiff](https://github.com/NVIDIA/physicsnemo)
- [CorrDiff paper](https://arxiv.org/abs/2309.15214)
- [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)
- [ERA5 on Copernicus CDS](https://cds.climate.copernicus.eu)
- [HRRR on AWS Open Data](https://registry.opendata.aws/noaa-hrrr-pds/)

## License

MIT
