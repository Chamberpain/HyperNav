"""Offline WCOFS archive, source clock and physical-grid regressions."""

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture(scope="module")
def wcofs():
    # UVBase is unrelated to the archive API; isolate its legacy plotting stack.
    package_name = "_wcofs_archive_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str((Path(__file__).resolve().parents[1] / "Utilities" / "Data"))]
    sys.modules[package_name] = package
    legacy = types.ModuleType("HyperNav.Utilities.Data.UVBase")
    legacy.Base = type("Base", (), {"scale_factor": 1})
    legacy.UVTimeList = list
    old = sys.modules.get(legacy.__name__)
    sys.modules[legacy.__name__] = legacy
    try:
        name = f"{package_name}.WCOFS"
        spec = importlib.util.spec_from_file_location(
            name, (Path(__file__).resolve().parents[1] / "Utilities" / "Data") / "WCOFS.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        if old is None:
            sys.modules.pop(legacy.__name__, None)
        else:
            sys.modules[legacy.__name__] = old
    return module


def regular_grid(valid_time="2020-01-01T00:00", *, surface=False):
    lats = np.asarray([38.0, 39.0, 41.0, 43.0, 44.0])
    lons = np.asarray([232.0, 233.0, 235.0, 236.1, 237.0])
    dims = ("time", "ny", "nx") if surface else ("time", "Depth", "ny", "nx")
    shape = (1, 5, 5) if surface else (1, 3, 5, 5)
    u = np.arange(np.prod(shape), dtype=float).reshape(shape) / 100
    source = xr.Dataset(
        {
            "u_eastward": (dims, u, {"units": "meters second-1"}),
            "v_northward": (dims, -u, {"units": "m/s"}),
            "Latitude": (("ny", "nx"), np.broadcast_to(lats[:, None], (5, 5)).copy()),
            "Longitude": (("ny", "nx"), np.broadcast_to(lons[None, :], (5, 5)).copy()),
        },
        coords={"time": np.asarray([valid_time], dtype="datetime64[ns]")},
    )
    if not surface:
        source = source.assign_coords(
            Depth=("Depth", [0.0, 50.0, 100.0], {"units": "m", "positive": "down"})
        )
    return source


@pytest.mark.parametrize(
    "date,expected",
    [
        ("2020-01-01T00:00", "2020/01/01/wcofs.t03z.20200101.regulargrid.n021.nc"),
        ("2020-01-01T03:00", "2020/01/01/wcofs.t03z.20200101.regulargrid.n024.nc"),
        ("2020-01-01T06:00", "2020/01/02/wcofs.t03z.20200102.regulargrid.n003.nc"),
        ("2020-12-31T21:00", "2021/01/01/wcofs.t03z.20210101.regulargrid.n018.nc"),
        ("2020-01-01T00:00-08:00", None),
    ],
)
def test_nowcast_urls_use_requested_cycle_date(wcofs, date, expected):
    if expected is None:
        with pytest.raises(ValueError, match="3-hour"):
            wcofs.WCOFSBase.source_url(date)
    else:
        assert wcofs.WCOFSBase.source_url(date).endswith(expected)


def test_timezone_and_forecast_clock(wcofs):
    assert wcofs.WCOFSBase.source_url("2019-12-31T16:00-08:00").endswith(
        "20200101.regulargrid.n021.nc"
    )
    assert wcofs.WCOFSBase.source_url(
        "2020-01-01T09:00", cycle="2020-01-01T03:00"
    ).endswith("20200101.regulargrid.f006.nc")
    assert wcofs.WCOFSBase.source_url(
        "2020-01-01T03:00", cycle="2020-01-01T03:00"
    ).endswith("20200101.regulargrid.n024.nc")
    with pytest.raises(ValueError, match="3 to 72"):
        wcofs.WCOFSBase.source_url("2020-01-04T06:00", cycle="2020-01-01T03:00")
    with pytest.raises(ValueError, match="03Z"):
        wcofs.WCOFSBase.source_url("2020-01-01T09:00", cycle="2020-01-01T00:00")


def test_normalization_preserves_values_and_brackets_humboldt(wcofs):
    source = regular_grid()
    original = source.copy(deep=True)
    result = wcofs.WCOFSBase._normalise(
        source, bounds=(-127, -123.9, 38.9, 43), max_depth=70
    )
    assert result.U.dims == ("time", "depth", "lat", "lon")
    np.testing.assert_allclose(result.depth, [0, 50])
    np.testing.assert_allclose(result.lat, [38, 39, 41, 43])
    np.testing.assert_allclose(result.lon, [-127, -125, -123.9])
    np.testing.assert_allclose(result.U, source.u_eastward.values[:, :2, :4, 1:4])
    assert result.U.attrs["units"] == "m s-1"
    assert result.depth.attrs["positive"] == "down"
    xr.testing.assert_identical(source, original)


def test_reversed_axes_and_positive_up_physical_depth(wcofs):
    source = regular_grid().isel(
        ny=slice(None, None, -1), nx=slice(None, None, -1), Depth=slice(None, None, -1)
    )
    source = source.assign_coords(
        Depth=("Depth", -source.Depth.values, {"units": "m", "positive": "up"})
    )
    result = wcofs.WCOFSBase._normalise(
        source, bounds=(-127, -123.9, 38.9, 43), max_depth=100
    )
    np.testing.assert_allclose(result.depth, [0, 50, 100])
    np.testing.assert_allclose(
        result.U, regular_grid().u_eastward.values[:, :, :4, 1:4]
    )


def test_surface_current_is_not_replicated_vertically(wcofs):
    result = wcofs.WCOFSBase._normalise(
        regular_grid(surface=True), bounds=(-127, -123.9, 38.9, 43), max_depth=700
    )
    assert result.sizes["depth"] == 1
    assert result.depth.item() == 0


@pytest.mark.parametrize(
    "problem,match",
    [
        ("curved", "Curvilinear"),
        ("sigma", "physical metre"),
        ("missing_units", "declare metre-per-second"),
        ("staggered", "staggered"),
        ("negative_depth", "positive down"),
        ("duplicate_axis", "unique finite"),
    ],
)
def test_invalid_source_rejected(wcofs, problem, match):
    source = regular_grid()
    if problem == "curved":
        source["Latitude"].values[0, 1] += 1
    elif problem == "sigma":
        source.Depth.attrs["units"] = "1"
    elif problem == "missing_units":
        source.u_eastward.attrs = {}
    elif problem == "staggered":
        source["u_eastward"] = xr.DataArray(
            source.u_eastward.values,
            dims=("time", "Depth", "eta_u", "xi_u"),
            attrs={"units": "m/s"},
        )
    elif problem == "negative_depth":
        source = source.assign_coords(Depth=("Depth", [0, -50, -100], {"units": "m"}))
    elif problem == "duplicate_axis":
        source.Longitude.values[:, 1] = source.Longitude.values[:, 0]
    with pytest.raises(ValueError, match=match):
        wcofs.WCOFSBase._normalise(
            source, bounds=(-127, -123.9, 38.9, 43), max_depth=100
        )


def test_bounds_do_not_extrapolate(wcofs):
    with pytest.raises(ValueError, match="beyond WCOFS coverage"):
        wcofs.WCOFSBase._normalise(
            regular_grid(), bounds=(-140, -123.9, 38.9, 43), max_depth=100
        )


def test_get_dataset_reports_retention_and_passes_requested_url(wcofs, monkeypatch):
    paths = []

    def unavailable(path, **kwargs):
        paths.append(path)
        raise OSError("missing")

    monkeypatch.setattr(wcofs.xr, "open_dataset", unavailable)
    with pytest.raises(OSError, match="approximately three days"):
        wcofs.WCOFSBase.get_dataset("2020-01-01T00:00")
    assert paths[0].endswith("20200101.regulargrid.n021.nc")


def local_files(wcofs, folder, start, end, *, wrong_time=False):
    template = str(folder / "wcofs.t03z.{date}.regulargrid.{kind}{hour:03d}.nc")
    for valid_time in pd.date_range(start, end, freq="3h", inclusive="left"):
        path = wcofs.WCOFSBase.source_url(valid_time, url_template=template)
        source = regular_grid(
            (
                valid_time + pd.Timedelta(hours=3) if wrong_time else valid_time
            ).isoformat()
        )
        source.to_netcdf(path, engine="scipy")
    return template


def test_parallel_download_and_resume_real_netcdf(wcofs, tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    start, end = "2020-01-01", "2020-01-03"
    template = local_files(wcofs, folder, start, end)
    destination = tmp_path / "record"
    manifest = wcofs.WCOFSHumboldt.download_record(
        start,
        end,
        destination,
        workers=2,
        retries=1,
        url_template=template,
        source="regulargrid",
        engine="scipy",
        max_depth=70,
    )
    assert len(manifest) == 2
    assert set(manifest.status) == {"downloaded"}
    chunks = sorted(destination.glob("*.nc"))
    assert len(chunks) == 2
    for path in chunks:
        with xr.open_dataset(path) as dataset:
            assert dataset.U.dims == ("time", "depth", "lat", "lon")
            assert dataset.sizes["time"] == 8
            assert dataset.sizes["depth"] == 2
            assert dataset.depth.attrs["positive"] == "down"
    before = {path: path.stat().st_mtime_ns for path in chunks}
    resumed = wcofs.WCOFSHumboldt.download_record(
        start,
        end,
        destination,
        workers=2,
        retries=1,
        url_template=template,
        source="regulargrid",
        engine="scipy",
        max_depth=70,
    )
    assert set(resumed.status) <= {"downloaded"}
    assert before == {path: path.stat().st_mtime_ns for path in chunks}


def test_wrong_source_valid_time_fails_without_chunk(wcofs, tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    template = local_files(wcofs, folder, "2020-01-01", "2020-01-02", wrong_time=True)
    output = tmp_path / "bad"
    manifest = wcofs.WCOFSHumboldt.download_record(
        "2020-01-01",
        "2020-01-02",
        output,
        workers=1,
        retries=1,
        url_template=template,
        source="regulargrid",
        engine="scipy",
    )
    assert set(manifest.status) == {"failed"}
    assert "different valid time" in manifest.error.iloc[0]
    assert not list(output.glob("*.nc"))


def test_raw_archive_template_filename_clock(wcofs):
    template = "https://example.org/{year:04d}/{month:02d}/{day:02d}/{filename}"
    assert wcofs.WCOFSBase.source_url(
        "2021-01-01T06:00", url_template=template, source="roms"
    ).endswith("/2021/01/02/wcofs.t03z.20210102.fields.n003.nc")
    assert wcofs.WCOFSBase.source_url(
        "2021-01-01T06:00", url_template="/local/{legacy_filename}", source="roms"
    ).endswith("/nos.wcofs.fields.n003.20210102.t03z.nc")
    with pytest.raises(ValueError, match="source must"):
        wcofs.WCOFSBase.source_url("2021-01-01", source="unknown")


def test_raw_archive_preprocessor_receives_geometry_and_validated_time(wcofs, tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    template = str(folder / "{filename}")
    valid_time = "2020-01-01T00:00"
    xr.Dataset(
        {"ocean_time": ("ocean_time", np.asarray([valid_time], dtype="datetime64[ns]"))}
    ).to_netcdf(
        wcofs.WCOFSBase.source_url(valid_time, url_template=template, source="roms"),
        engine="scipy",
    )
    calls = []

    def processor(dataset, **kwargs):
        calls.append(kwargs)
        return wcofs.WCOFSBase._normalise(
            regular_grid(pd.Timestamp(dataset.ocean_time.values[0]).isoformat()),
            bounds=kwargs["bounds"],
            max_depth=kwargs["max_depth"],
        )

    manifest = wcofs.WCOFSHumboldt.download_record(
        valid_time,
        "2020-01-01T03:00",
        tmp_path / "record",
        workers=1,
        url_template=template,
        source="roms",
        preprocess=processor,
        max_depth=100,
        depth_levels=[0, 50, 100],
        grid_spacing=0.1,
        engine="scipy",
    )
    assert set(manifest.status) == {"downloaded"}
    assert calls == [
        {
            "bounds": (-127, -123.9, 38.9, 43),
            "max_depth": 100.0,
            "depth_levels": [0, 50, 100],
            "grid_spacing": 0.1,
        }
    ]


@pytest.mark.parametrize(
    "valid_time,filename",
    [
        ("2023-05-26T06:00", "2023/05/nos.wcofs.fields.n003.20230527.t03z.nc"),
        ("2024-09-08T06:00", "2024/09/nos.wcofs.fields.n003.20240909.t03z.nc"),
        ("2024-09-09T06:00", "2024/09/wcofs.t03z.20240910.fields.n003.nc"),
        ("2026-09-26T06:00", "2026/09/wcofs.t03z.20260927.fields.n003.nc"),
    ],
)
def test_default_ncei_historical_url_and_noaa_rename_boundary(
    wcofs, valid_time, filename
):
    assert wcofs.WCOFSBase.source_url(valid_time, source="roms") == (
        "https://www.ncei.noaa.gov/thredds/dodsC/model-wcofs-files/" + filename
    )


def test_native_roms_file_to_parallel_current_archive(wcofs, tmp_path):
    # The analytic fixture declares all real ROMS geometry/transform metadata.
    spec = importlib.util.spec_from_file_location(
        "_adapter_roms_fixture", Path(__file__).with_name("test_wcofs_roms.py")
    )
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    folder = tmp_path / "native"
    folder.mkdir()
    template = str(folder / "{archive_filename}")
    for valid_time in pd.date_range(
        "2020-01-01", "2020-01-03", freq="3h", inclusive="left"
    ):
        source = fixtures.native_dataset(angle=np.pi / 2).assign_coords(
            ocean_time=[valid_time.to_datetime64()]
        )
        source.to_netcdf(
            wcofs.WCOFSBase.source_url(
                valid_time, source="roms", url_template=template
            ),
            engine="scipy",
        )
    output = tmp_path / "processed"
    manifest = wcofs.WCOFSHumboldt.download_record(
        "2020-01-01",
        "2020-01-03",
        output,
        workers=2,
        bounds=[-126.9, -126.5, 40.1, 40.5],
        max_depth=100,
        depth_levels=[10, 25, 50, 80],
        grid_spacing=0.1,
        url_template=template,
        engine="scipy",
    )
    assert set(manifest.status) == {"downloaded"}
    for path in sorted(output.glob("*.nc")):
        with xr.open_dataset(path) as currents:
            assert currents.U.dims == ("time", "depth", "lat", "lon")
            assert currents.sizes["time"] == 8
            np.testing.assert_allclose(currents.U, -2, atol=1e-6)
            np.testing.assert_allclose(currents.V, 1, atol=1e-6)
            assert currents.attrs["vertical_reference"]
            assert "processing" in currents.attrs
    import json

    settings = json.loads((output / "request.json").read_text())["settings"]
    assert settings["source"] == "roms"
    assert len(settings["preprocessor_sha256"]) == 64
