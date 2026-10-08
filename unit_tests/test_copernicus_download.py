"""Offline SDK tests using real xarray data and NetCDF archives."""

import datetime
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Geometry and geopy are legacy Base dependencies, unrelated to SDK I/O.
# Keep these narrow import-only substitutes scoped to a private test package.
_stubs = {}
if importlib.util.find_spec("shapely") is None:
    _geometry = types.ModuleType("shapely.geometry")
    _geometry.box = lambda *bounds: bounds
    _geometry.MultiPolygon = list
    _shapely = types.ModuleType("shapely")
    _shapely.__path__ = []
    _shapely.geometry = _geometry
    _stubs.update({"shapely": _shapely, "shapely.geometry": _geometry})
if importlib.util.find_spec("geopy") is None:
    _geopy = types.ModuleType("geopy")
    _geopy.Point = object
    _stubs["geopy"] = _geopy
_package = types.ModuleType("_test_copernicus_pkg")
_package.__path__ = [str((Path(__file__).resolve().parents[1] / "Utilities" / "Data"))]
sys.modules[_package.__name__] = _package
with patch.dict(sys.modules, _stubs):
    copernicus = importlib.import_module(f"{_package.__name__}.CopernicusGlobal")
sys.modules[copernicus.__name__] = copernicus


def source_dataset(
    start="2024-01-01", end="2024-01-04", *, units="m s-1", vertical=False
):
    times = pd.date_range(start, end, freq="6h").tz_localize(None)
    shape = (len(times), 3, 2, 3)
    variables = {
        "uo": (
            ("time", "depth", "latitude", "longitude"),
            np.ones(shape),
            {"units": units},
        ),
        "vo": (
            ("time", "depth", "latitude", "longitude"),
            np.ones(shape) * 2,
            {"units": units},
        ),
    }
    if vertical:
        variables["wo"] = (
            ("time", "depth", "latitude", "longitude"),
            np.ones(shape) * 3,
            {"units": units, "standard_name": "upward_sea_water_velocity"},
        )
    return xr.Dataset(
        variables,
        coords={
            "time": times,
            "depth": ("depth", [0.49, 10.0, 50.0], {"units": "m", "positive": "down"}),
            "latitude": ("latitude", [38.8, 43.1], {"units": "degrees_north"}),
            "longitude": (
                "longitude",
                [-127.1, -125.0, -123.8],
                {"units": "degrees_east"},
            ),
        },
    )


@pytest.fixture
def sdk(monkeypatch):
    calls = []

    def open_dataset(**kwargs):
        calls.append(kwargs)
        return source_dataset(kwargs["start_datetime"], kwargs["end_datetime"])

    module = types.SimpleNamespace(
        open_dataset=open_dataset,
        login=lambda **kwargs: pytest.fail("Automatic login is forbidden"),
    )
    monkeypatch.setitem(sys.modules, "copernicusmarine", module)
    return calls


def test_import_is_offline_and_does_not_load_optional_resources(sdk):
    before = set(sys.modules)
    with patch.dict(sys.modules, _stubs):
        importlib.reload(copernicus)
    assert sdk == []
    assert not hasattr(copernicus.HumboldtCopernicus, "dataset")
    for name in (
        "GeneralUtilities.Plot.Cartopy.regional_plot",
        "GeneralUtilities.Compute.Depth.depth_utilities",
    ):
        assert name in before or name not in sys.modules


def test_subset_forwards_depth_dates_and_current_sdk_options(sdk):
    data = copernicus.HumboldtCopernicus.get_dataset(
        43,
        38.9,
        -123.9,
        -127,
        -123,
        "a-historical-product",
        "2001-01-01",
        "2001-02-01",
        dataset_version="v1",
        dataset_part="part1",
        credentials_file="local.auth",
        service="geoseries",
    )
    data.close()
    assert sdk[0] == {
        "dataset_id": "a-historical-product",
        "minimum_longitude": -127,
        "maximum_longitude": -123.9,
        "minimum_latitude": 38.9,
        "maximum_latitude": 43,
        "start_datetime": "2001-01-01",
        "end_datetime": "2001-02-01",
        "variables": ["uo", "vo"],
        "minimum_depth": 0,
        "maximum_depth": 123,
        "vertical_axis": "depth",
        "coordinates_selection_method": "outside",
        "dataset_version": "v1",
        "dataset_part": "part1",
        "credentials_file": "local.auth",
        "service": "geoseries",
    }


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_invalid_depth_fails_before_remote_open(sdk, bad):
    with pytest.raises(ValueError, match="finite"):
        copernicus.HumboldtCopernicus.get_dataset(43, 38.9, -123.9, -127, bad, "id")
    assert not sdk


def test_legacy_dimensions_keep_real_depths_and_coordinate_types():
    data = source_dataset()
    dimensions = copernicus.HumboldtCopernicus.get_dimensions(
        -123.9, -127, 43, 38.9, 50, data
    )
    times, lats, lons, depths = dimensions[:4]
    assert depths == [-0.49, -10.0, -50.0]
    assert lats == [38.8, 43.1]
    assert lons == [-127.1, -125.0, -123.8]
    assert isinstance(times[0], datetime.datetime)
    assert dimensions[-2] == "m s-1"


def test_lazy_initialize_is_per_region_and_force_closes_old_dataset(monkeypatch):
    calls = []
    closed = []

    def get_dataset(cls, *args):
        calls.append(cls)
        data = source_dataset()
        data.set_close(lambda: closed.append(cls))
        return data

    monkeypatch.setattr(
        copernicus.CopernicusGlobal, "get_dataset", classmethod(get_dataset)
    )
    for cls in (copernicus.HumboldtCopernicus, copernicus.SoCalCopernicus):
        for attr in ("dataset", "units", "dataset_time", "lats", "lons", "depths"):
            if attr in cls.__dict__:
                monkeypatch.delattr(cls, attr)
    try:
        humboldt = copernicus.HumboldtCopernicus.initialize()
        assert copernicus.HumboldtCopernicus.initialize() is humboldt
        copernicus.SoCalCopernicus.initialize()
        copernicus.HumboldtCopernicus.initialize(force=True)
        assert calls == [
            copernicus.HumboldtCopernicus,
            copernicus.SoCalCopernicus,
            copernicus.HumboldtCopernicus,
        ]
        assert closed == [copernicus.HumboldtCopernicus]
    finally:
        for cls in (copernicus.HumboldtCopernicus, copernicus.SoCalCopernicus):
            for attr in ("dataset", "units", "dataset_time", "lats", "lons", "depths"):
                if attr in cls.__dict__:
                    delattr(cls, attr)


def test_time_chunk_indices_do_not_duplicate_final_index():
    assert copernicus.CopUVTimeList([]).return_time_list() == []
    dates = [
        datetime.datetime(2024, 1, 1) + datetime.timedelta(hours=i) for i in range(82)
    ]
    assert copernicus.CopUVTimeList(dates[:1]).return_time_list() == [0]
    assert copernicus.CopUVTimeList(dates[:81]).return_time_list() == [0, 40, 80]
    assert copernicus.CopUVTimeList(dates).return_time_list() == [0, 40, 80, 81]


def test_nanosecond_time_round_trip():
    values = np.array(
        ["2024-01-01", "2024-01-01T06:00"], dtype="datetime64[ns]"
    ).tolist()
    assert copernicus.nanosecond_convert(values, np.datetime64("1970-01-01")) == [
        datetime.datetime(2024, 1, 1),
        datetime.datetime(2024, 1, 1, 6),
    ]


def capture_chunk(monkeypatch, source):
    archive = importlib.import_module(f"{copernicus.__package__}.current_archive")
    captures = {}
    closed = []
    source.set_close(lambda: closed.append(True))
    monkeypatch.setattr(
        copernicus.HumboldtCopernicus, "get_dataset", lambda *args, **kwargs: source
    )

    def fake_download(fetch_chunk, **kwargs):
        captures.update(kwargs)
        captures["dataset"] = fetch_chunk(
            pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC")
        )
        return "manifest"

    monkeypatch.setattr(archive, "download_record", fake_download)
    return captures, closed


def test_canonical_chunk_has_si_velocities_real_depths_and_no_fabricated_w(
    monkeypatch, tmp_path
):
    source = source_dataset(units="cm/s")
    captures, closed = capture_chunk(monkeypatch, source)
    result = copernicus.HumboldtCopernicus.download_record(
        "2024-01-01", "2024-01-02", tmp_path, workers=3, worker_index=1
    )
    assert result == "manifest"
    data = captures["dataset"]
    assert set(data.data_vars) == {"U", "V"}
    assert data.U.dims == ("time", "depth", "lat", "lon")
    np.testing.assert_allclose(data.U, 0.01)
    np.testing.assert_allclose(data.V, 0.02)
    np.testing.assert_allclose(data.depth, [0.49, 10.0, 50.0])
    assert data.depth.attrs["positive"] == "down"
    assert data.U.attrs["units"] == "m s-1"
    assert data.time.max().values < np.datetime64("2024-01-02")
    assert data.lat.min() < 38.9 and data.lat.max() > 43
    assert data.lon.min() < -127 and data.lon.max() > -123.9
    assert captures["workers"] == 3 and captures["worker_index"] == 1
    assert captures["settings"]["bounds"] == [-127.0, -123.9, 38.9, 43.0]
    assert "credentials_file" not in captures["settings"]
    assert closed == [True]
    assert source.uo.attrs["units"] == "cm/s"


def test_true_vertical_velocity_uses_positive_down_convention(monkeypatch, tmp_path):
    source = source_dataset(units="cm/s", vertical=True)
    source.wo.attrs["positive"] = "up"
    captures, _ = capture_chunk(monkeypatch, source)
    copernicus.HumboldtCopernicus.download_record(
        "2024-01-01",
        "2024-01-02",
        tmp_path,
        variables={"U": "uo", "V": "vo", "W": "wo"},
    )
    data = captures["dataset"]
    np.testing.assert_allclose(data.W, -0.03)
    assert data.W.attrs["standard_name"] == "downward_sea_water_velocity"
    assert data.W.attrs["positive"] == "down"
    assert source.wo.attrs["positive"] == "up"
    archive = importlib.import_module(f"{copernicus.__package__}.current_archive")
    assert (
        archive._validate_currents(
            data, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
        )["records"]
        == 4
    )


def test_failed_normalization_closes_source(monkeypatch, tmp_path):
    _, closed = capture_chunk(monkeypatch, source_dataset(units="unknown"))
    with pytest.raises(ValueError, match="unsupported velocity units"):
        copernicus.HumboldtCopernicus.download_record(
            "2024-01-01", "2024-01-02", tmp_path
        )
    assert closed == [True]


@pytest.mark.parametrize(
    "mapping",
    [
        {"U": "uo"},
        {"U": "uo", "V": "vo", "Q": "q"},
        {"U": "uo", "V": ""},
        {},
        {"U": "uo", "V": "uo"},
    ],
)
def test_invalid_variable_mapping_fails_before_fetch(monkeypatch, tmp_path, mapping):
    monkeypatch.setattr(
        copernicus.HumboldtCopernicus,
        "get_dataset",
        lambda *args, **kwargs: pytest.fail("Invalid config must not fetch"),
    )
    with pytest.raises(ValueError):
        copernicus.HumboldtCopernicus.download_record(
            "2024-01-01", "2024-01-02", tmp_path, variables=mapping
        )


def test_real_archive_resumes_and_preserves_exclusive_chunks(sdk, tmp_path):
    manifest = copernicus.HumboldtCopernicus.download_record(
        "2024-01-01",
        "2024-01-04",
        tmp_path,
        workers=2,
        chunk_days=1,
        dataset_id="custom-source",
        time_step="6h",
        max_depth=50,
    )
    assert len(manifest) == 3
    requests = sorted(sdk, key=lambda row: row["start_datetime"])
    assert requests[0]["dataset_id"] == "custom-source"
    assert requests[0]["maximum_depth"] == 50
    for request in requests:
        start = pd.Timestamp(request["start_datetime"])
        stop = pd.Timestamp(request["end_datetime"])
        assert stop - start == pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    files = sorted(tmp_path.glob("*.nc"))
    assert len(files) == 3
    times = []
    for filename in files:
        with xr.open_dataset(filename) as data:
            assert data.U.dims == ("time", "depth", "lat", "lon")
            assert set(data.data_vars) == {"U", "V"}
            times.extend(data.time.values.tolist())
    assert len(times) == len(set(times)) == 12
    call_count = len(sdk)
    resumed = copernicus.HumboldtCopernicus.download_record(
        "2024-01-01",
        "2024-01-04",
        tmp_path,
        workers=2,
        chunk_days=1,
        dataset_id="custom-source",
        time_step="6h",
        max_depth=50,
    )
    assert len(resumed) == 3
    assert len(sdk) == call_count


@pytest.mark.parametrize("keep", [[0], [0, 1, 2], [1, 2, 3], [0, 2, 3]])
def test_partial_chunks_fail_instead_of_saving_incomplete_records(
    monkeypatch, tmp_path, keep
):
    _, closed = capture_chunk(monkeypatch, source_dataset().isel(time=keep))
    with pytest.raises(ValueError, match="Incomplete Copernicus time coverage"):
        copernicus.HumboldtCopernicus.download_record(
            "2024-01-01", "2024-01-02", tmp_path
        )
    assert closed == [True]


def test_daily_reanalysis_uses_native_noon_phase(monkeypatch, tmp_path):
    source = source_dataset().sel(
        time=["2024-01-01T12:00", "2024-01-02T12:00", "2024-01-03T12:00"]
    )
    captures, _ = capture_chunk(monkeypatch, source)
    copernicus.HumboldtCopernicus.download_record(
        "2024-01-01",
        "2024-01-02",
        tmp_path,
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    )
    assert captures["settings"]["time_step_seconds"] == 86400
    assert (
        captures["dataset"].time.values.tolist()
        == source.time.isel(time=slice(0, 1)).values.tolist()
    )
    assert captures["dataset"].attrs["native_time_phase_seconds"] == 43200


def test_custom_dataset_cadence_is_explicit_and_saved(monkeypatch, tmp_path):
    source = source_dataset()
    captures, _ = capture_chunk(monkeypatch, source)
    copernicus.HumboldtCopernicus.download_record(
        "2024-01-01", "2024-01-02", tmp_path, dataset_id="custom-source", time_step="6h"
    )
    assert captures["settings"]["time_step_seconds"] == 21600
    with pytest.raises(ValueError, match="explicit fixed time_step"):
        copernicus.HumboldtCopernicus.download_record(
            "2024-01-01", "2024-01-02", tmp_path, dataset_id="custom-source"
        )


@pytest.mark.parametrize("step", [0, -1, np.nan, True, "0h"])
def test_invalid_cadence_fails_before_remote_open(sdk, tmp_path, step):
    with pytest.raises(ValueError):
        copernicus.HumboldtCopernicus.download_record(
            "2024-01-01", "2024-01-02", tmp_path, time_step=step
        )
    assert sdk == []


def test_wrapped_longitudes_and_descending_axes_are_sorted_with_values(
    monkeypatch, tmp_path
):
    source = source_dataset()
    source.uo[:] = np.broadcast_to([0.1, 0.2, 0.3], source.uo.shape)
    source = source.isel(
        depth=slice(None, None, -1),
        latitude=slice(None, None, -1),
        longitude=slice(None, None, -1),
    )
    source = source.assign_coords(
        longitude=(
            source.longitude.dims,
            source.longitude.values % 360,
            dict(source.longitude.attrs),
        )
    )
    captures, _ = capture_chunk(monkeypatch, source)
    copernicus.HumboldtCopernicus.download_record("2024-01-01", "2024-01-02", tmp_path)
    data = captures["dataset"]
    np.testing.assert_allclose(data.lon, [-127.1, -125, -123.8])
    np.testing.assert_allclose(data.lat, [38.8, 43.1])
    np.testing.assert_allclose(data.depth, [0.49, 10, 50])
    np.testing.assert_allclose(data.U.isel(time=0, depth=0, lat=0), [0.1, 0.2, 0.3])
    assert (source.longitude > 180).all()
    assert np.all(np.diff(source.depth) < 0)
    archive = importlib.import_module(f"{copernicus.__package__}.current_archive")
    assert (
        archive._validate_currents(
            data, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
        )["records"]
        == 4
    )
