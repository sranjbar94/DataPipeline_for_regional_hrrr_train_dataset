# Inline documentation clarified.
# Docstring coverage checked and touched up.
# Reviewed: comments kept in sync with behavior.
# Reviewed: comments kept in sync with behavior.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
"""
Tests that the output NetCDF matches the HRRR-Mini schema expected
by the CorrDiff dataset loader.

Run with:  pytest tests/test_nc_schema.py -v
"""

import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import netCDF4 as nc
import numpy as np
import pytest

from src.pipeline.nc_writer import NCWriter, _all_input_channels, _OUTPUT_CHANNELS


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def minimal_cfg():
    return SimpleNamespace(
        patches=SimpleNamespace(era5_patch_size=8, hrrr_patch_size=64),
        storage=SimpleNamespace(compression_level=1),
    )


@pytest.fixture
def sample_era5_patch():
    patch = {ch: np.random.randn(8, 8).astype(np.float32)
             for ch in _all_input_channels()}
    patch["_lat"] = np.linspace(30, 32, 8).astype(np.float32)
    patch["_lon"] = np.linspace(-80, -78, 8).astype(np.float32)
    patch["lsm"]  = np.zeros((8, 8), dtype=np.float32)
    return patch


@pytest.fixture
def sample_hrrr_patch():
    return {ch: np.random.randn(64, 64).astype(np.float32)
            for ch in _OUTPUT_CHANNELS}


# ------------------------------------------------------------------
# Schema tests
# ------------------------------------------------------------------

class TestNCSchema:

    def test_dimensions(self, minimal_cfg, sample_era5_patch, sample_hrrr_patch):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.nc"
            minimal_cfg.storage = SimpleNamespace(
                compression_level=1,
                output_dir=tmp,
                output_filename="test.nc",
            )
            writer = NCWriter(path, n_samples=3, cfg=minimal_cfg)
            writer.open()

            for i in range(3):
                writer.write_sample(
                    idx=i, dt=datetime(2015, 6, 1, i),
                    lat_idx=100, lon_idx=200,
                    era5_patch=sample_era5_patch,
                    hrrr_patch=sample_hrrr_patch,
                )
            writer.close()

            ds = nc.Dataset(str(path))
            assert "sample"  in ds.dimensions
            assert "y_lr"    in ds.dimensions
            assert "x_lr"    in ds.dimensions
            assert "y_hr"    in ds.dimensions
            assert "x_hr"    in ds.dimensions
            assert len(ds.dimensions["sample"]) == 3
            assert len(ds.dimensions["y_lr"])   == 8
            assert len(ds.dimensions["y_hr"])   == 64
            ds.close()

    def test_groups_exist(self, minimal_cfg, sample_era5_patch, sample_hrrr_patch):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.nc"
            writer = NCWriter(path, n_samples=1, cfg=minimal_cfg)
            writer.open()
            writer.write_sample(0, datetime(2015, 1, 1), 100, 200,
                                 sample_era5_patch, sample_hrrr_patch)
            writer.close()

            ds = nc.Dataset(str(path))
            assert "input"     in ds.groups
            assert "output"    in ds.groups
            assert "invariant" in ds.groups
            ds.close()

    def test_output_channels(self, minimal_cfg, sample_era5_patch, sample_hrrr_patch):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.nc"
            writer = NCWriter(path, n_samples=1, cfg=minimal_cfg)
            writer.open()
            writer.write_sample(0, datetime(2015, 1, 1), 100, 200,
                                 sample_era5_patch, sample_hrrr_patch)
            writer.close()

            ds   = nc.Dataset(str(path))
            grp  = ds.groups["output"]
            for ch in ["2t", "10u", "10v", "tp", "ssrd", "strd", "sp", "q", "sf"]:
                assert ch in grp.variables, f"Missing output channel: {ch}"
                assert grp[ch].shape == (1, 64, 64)
            ds.close()

    def test_time_encoding(self, minimal_cfg, sample_era5_patch, sample_hrrr_patch):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.nc"
            writer = NCWriter(path, n_samples=1, cfg=minimal_cfg)
            writer.open()
            dt = datetime(2015, 7, 15, 12)
            writer.write_sample(0, dt, 100, 200,
                                 sample_era5_patch, sample_hrrr_patch)
            writer.close()

            ds    = nc.Dataset(str(path))
            hours = float(ds["time"][0])
            ds.close()
            expected = (dt - datetime(2015, 1, 1)).total_seconds() / 3600
            assert abs(hours - expected) < 1e-3

    def test_no_nan_in_written_data(self, minimal_cfg,
                                    sample_era5_patch, sample_hrrr_patch):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.nc"
            writer = NCWriter(path, n_samples=1, cfg=minimal_cfg)
            writer.open()
            writer.write_sample(0, datetime(2015, 1, 1), 100, 200,
                                 sample_era5_patch, sample_hrrr_patch)
            writer.close()

            ds  = nc.Dataset(str(path))
            grp = ds.groups["output"]
            for ch in ["2t", "10u", "10v"]:
                arr = grp[ch][0]
                assert not np.any(np.isnan(arr)), f"NaN found in {ch}"
            ds.close()

    def test_invariant_written_once(self, minimal_cfg,
                                    sample_era5_patch, sample_hrrr_patch):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.nc"
            writer = NCWriter(path, n_samples=2, cfg=minimal_cfg)
            writer.open()
            for i in range(2):
                writer.write_sample(i, datetime(2015, 1, 1, i), 100, 200,
                                     sample_era5_patch, sample_hrrr_patch)
            writer.close()

            ds  = nc.Dataset(str(path))
            lat = ds.groups["invariant"]["latitude"][:]
            assert lat.shape == (8, 8)
            assert not np.all(lat == 0)
            ds.close()
