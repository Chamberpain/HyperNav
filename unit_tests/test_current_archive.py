"""Offline regression checks for reusable regional current archives."""

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from threading import Lock
import time

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def archive():
    return load_module(
        "_test_current_archive", (Path(__file__).resolve().parents[1] / "Utilities" / "Data") / "current_archive.py"
    )


def currents(times=("2020-01-01T00:00",)):
    shape = (len(times), 2, 2, 2)
    return xr.Dataset(
        {
            "U": (
                ("time", "depth", "lat", "lon"),
                np.full(shape, 0.15),
                {"units": "m s-1", "standard_name": "eastward_sea_water_velocity"},
            ),
            "V": (
                ("time", "depth", "lat", "lon"),
                np.full(shape, -0.05),
                {"units": "m s-1", "standard_name": "northward_sea_water_velocity"},
            ),
        },
        coords={
            "time": np.asarray(times, dtype="datetime64[ns]"),
            "depth": (
                "depth",
                [0.0, 100.0],
                {"units": "m", "positive": "down", "axis": "Z"},
            ),
            "lat": ("lat", [39.0, 43.0], {"units": "degrees_north", "axis": "Y"}),
            "lon": ("lon", [-127.0, -124.0], {"units": "degrees_east", "axis": "X"}),
        },
    )


def download(archive, tmp_path, fetch_chunk=None, **kwargs):
    options = {
        "start": "2020-01-01",
        "end": "2020-01-04",
        "output_dir": tmp_path,
        "model": "test_currents",
        "workers": 2,
        "engine": "scipy",
    }
    options.update(kwargs)
    if fetch_chunk is None:

        def fetch_chunk(start, end):
            return currents((start,))

    return archive.download_record(fetch_chunk, **options)


def test_netcdf_canonical_roundtrip_and_real_cec_loader(archive, tmp_path):
    frame = download(archive, tmp_path)
    assert len(frame) == 3
    assert set(frame.status) == {"downloaded"}
    assert frame.records.tolist() == [1, 1, 1]
    assert frame.chunk_id.is_unique
    paths = frame.output_path.map(Path).tolist()
    for path in paths:
        with xr.open_dataset(path) as dataset:
            assert set(dataset.data_vars) == {"U", "V"}
            assert dataset.U.dims == ("time", "depth", "lat", "lon")
            assert dataset.depth.attrs["positive"] == "down"
            assert dataset.U.attrs["units"] == "m s-1"
            assert dataset.lat.attrs["units"] == "degrees_north"
            assert dataset.lon.attrs["units"] == "degrees_east"
            assert dataset.attrs["archive_settings_sha256"]
            assert dataset.attrs["requested_start"]
            assert dataset.attrs["requested_end"]
            np.testing.assert_allclose(dataset.U, 0.15)
    loader = pytest.importorskip("CEC.current_load_year")
    result = loader.ModelCurrent({2020: paths}).load_year(2020, as_fieldset=False)
    np.testing.assert_array_equal(
        result.time.values,
        np.asarray(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]"),
    )
    np.testing.assert_allclose(result.U, 0.15)
    np.testing.assert_allclose(result.V, -0.05)


def test_real_vertical_velocity_is_preserved(archive, tmp_path):
    def fetch(start, end):
        result = currents((start,))
        result["W"] = xr.full_like(result.U, 0.003)
        result.W.attrs.update(units="m s-1", positive="down")
        return result

    frame = download(archive, tmp_path, fetch)
    with xr.open_dataset(frame.output_path.iloc[0]) as dataset:
        np.testing.assert_allclose(dataset.W, 0.003)
        assert dataset.W.attrs["positive"] == "down"


@pytest.mark.parametrize(
    "attributes",
    [
        {"standard_name": "upward_sea_water_velocity"},
        {"standard_name": "upward_sea_water_velocity", "positive": "down"},
        {"standard_name": "downward_sea_water_velocity", "positive": "up"},
    ],
)
def test_non_downward_or_conflicting_w_direction_is_rejected(
    archive, tmp_path, attributes
):
    def fetch(start, end):
        result = currents((start,))
        result["W"] = xr.full_like(result.U, 0.003)
        result.W.attrs = {"units": "m s-1"} | attributes
        return result

    frame = download(archive, tmp_path, fetch, retries=1, end="2020-01-02")
    assert frame.status.tolist() == ["failed"]
    assert "positive downward" in frame.error.iloc[0]
    assert not Path(frame.output_path.iloc[0]).exists()


@pytest.mark.parametrize("invalid", [False, True])
def test_source_dataset_is_closed_after_copy_or_validation_failure(
    archive, tmp_path, invalid
):
    fetched, closed = [], []

    def fetch(start, end):
        result = currents((start,)).load()
        token = len(fetched)
        fetched.append(token)
        result.set_close(lambda: closed.append(token))
        if invalid:
            result.U.attrs["units"] = "unphysical_units"
        return result

    frame = download(archive, tmp_path, fetch, retries=2, end="2020-01-02")
    assert frame.status.tolist() == ["failed" if invalid else "downloaded"]
    assert sorted(closed) == fetched
    assert len(closed) == (2 if invalid else 1)


def test_land_mask_remains_missing_in_netcdf(archive, tmp_path):
    def fetch(start, end):
        result = currents((start,))
        result["U"][:, :, 0, 0] = np.nan
        result["V"][:, :, 0, 0] = np.nan
        return result

    result = download(archive, tmp_path, fetch, end="2020-01-02")
    assert result.status.tolist() == ["downloaded"]
    with xr.open_dataset(result.output_path.iloc[0]) as dataset:
        assert np.isnan(dataset.U[:, :, 0, 0]).all()
        assert np.isnan(dataset.V[:, :, 0, 0]).all()
        np.testing.assert_allclose(dataset.U[:, :, 1, 1], 0.15)


def test_thread_pool_obeys_worker_limit_and_overlaps_requests(archive, tmp_path):
    lock = Lock()
    active = maximum = 0

    def fetch(start, end):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.04)
            return currents((start,))
        finally:
            with lock:
                active -= 1

    frame = download(archive, tmp_path, fetch, workers=3, end="2020-01-08")
    assert len(frame) == 7
    assert set(frame.status) == {"downloaded"}
    assert 1 < maximum <= 3


def test_partitioned_workers_cover_every_chunk_once(archive, tmp_path):
    observed = []

    def fetch(start, end):
        observed.append(start)
        return currents((start,))

    for index in range(3):
        frame = download(
            archive,
            tmp_path,
            fetch,
            workers=3,
            worker_index=index,
            end="2020-01-08",
        )
        expected = pd.date_range("2020-01-01", "2020-01-07")[index::3]
        assert set(observed) >= set(expected)
    assert sorted(observed) == list(pd.date_range("2020-01-01", "2020-01-07"))
    assert len(frame) == 7
    assert set(frame.status) == {"downloaded"}


def test_simultaneous_partition_workers_do_not_lose_manifest_updates(archive, tmp_path):
    observed, lock = [], Lock()

    def fetch(start, end):
        with lock:
            observed.append(start)
        time.sleep(0.015)
        return currents((start,))

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                download,
                archive,
                tmp_path,
                fetch,
                workers=3,
                worker_index=index,
                end="2020-01-08",
            )
            for index in range(3)
        ]
        for future in futures:
            future.result(timeout=30)
    final = download(archive, tmp_path, fetch, workers=3, end="2020-01-08")
    assert set(final.status) == {"downloaded"}
    assert len(final) == 7
    assert sorted(observed) == list(pd.date_range("2020-01-01", "2020-01-07"))


def test_duplicate_terminal_worker_does_not_download_same_target_twice(
    archive, tmp_path
):
    calls, lock = [], Lock()

    def fetch(start, end):
        with lock:
            calls.append(start)
        time.sleep(0.02)
        return currents((start,))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                download, archive, tmp_path, fetch, workers=2, worker_index=0
            )
            for _ in range(2)
        ]
        for future in futures:
            result = future.result(timeout=30)
    assert sorted(calls) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03")]
    assert result.status.tolist() == ["downloaded", "pending", "downloaded"]


def test_resume_valid_data_preserves_bytes_without_remote_calls(archive, tmp_path):
    initial = download(archive, tmp_path)
    original = {path: Path(path).read_bytes() for path in initial.output_path}

    def unexpected_request(start, end):
        pytest.fail("A complete, validated chunk must be resumed without downloading.")

    resumed = download(archive, tmp_path, unexpected_request, workers=1)
    assert set(resumed.status) == {"downloaded"}
    assert original == {path: Path(path).read_bytes() for path in resumed.output_path}


def test_tuple_settings_resume_as_equivalent_json_lists(archive, tmp_path):
    settings = {"bounds": (-127, -124, 39, 43), "depth_levels": (5, 10)}
    original = download(archive, tmp_path, settings=settings, end="2020-01-02")
    preserved = Path(original.output_path.iloc[0]).read_bytes()

    def unexpected_request(start, end):
        pytest.fail(
            "Tuple settings and equivalent saved JSON lists identify the same archive."
        )

    for matching_settings in (
        settings,
        {name: list(values) for name, values in settings.items()},
    ):
        result = download(
            archive,
            tmp_path,
            unexpected_request,
            settings=matching_settings,
            end="2020-01-02",
        )
        assert result.status.tolist() == ["downloaded"]
        assert Path(result.output_path.iloc[0]).read_bytes() == preserved
    saved = json.loads((tmp_path / "request.json").read_text())
    assert saved["settings"] == {
        name: list(values) for name, values in settings.items()
    }


def test_repair_only_truncated_target_and_preserve_other_chunks(archive, tmp_path):
    initial = download(archive, tmp_path)
    target = Path(initial.output_path.iloc[1])
    target.write_bytes(b"truncated netcdf")
    intact = {
        path: Path(path).read_bytes()
        for path in initial.output_path
        if path != str(target)
    }
    requests = []

    def fetch(start, end):
        requests.append(start)
        return currents((start,))

    repaired = download(archive, tmp_path, fetch)
    assert requests == [pd.Timestamp("2020-01-02")]
    assert set(repaired.status) == {"downloaded"}
    assert intact == {path: Path(path).read_bytes() for path in intact}
    with xr.open_dataset(target) as dataset:
        np.testing.assert_allclose(dataset.U, 0.15)


def test_failed_repair_does_not_replace_existing_target(archive, tmp_path):
    initial = download(archive, tmp_path)
    target = Path(initial.output_path.iloc[0])
    existing = b"interrupted old download"
    target.write_bytes(existing)
    requests = []

    def fetch(start, end):
        requests.append(start)
        raise OSError("service unavailable")

    result = download(archive, tmp_path, fetch, retries=2)
    failed = result.loc[result.status == "failed"]
    assert len(failed) == 1
    assert "service unavailable" in failed.error.iloc[0]
    assert len(requests) == 2
    assert target.read_bytes() == existing
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_netcdf_write_does_not_replace_existing_target(
    archive, tmp_path, monkeypatch
):
    initial = download(archive, tmp_path, end="2020-01-02")
    target = Path(initial.output_path.iloc[0])
    existing = b"previous interrupted download"
    target.write_bytes(existing)

    def interrupted_write(dataset, path, **kwargs):
        Path(path).write_bytes(b"interrupted replacement")
        raise OSError("disk write failed")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", interrupted_write)
    result = download(archive, tmp_path, retries=1, end="2020-01-02")
    assert result.status.tolist() == ["failed"]
    assert "disk write failed" in result.error.iloc[0]
    assert target.read_bytes() == existing
    assert list(tmp_path.glob("*.nc")) == [target]


def test_resume_rejects_valid_netcdf_with_wrong_chunk_header(archive, tmp_path):
    initial = download(archive, tmp_path)
    target = Path(initial.output_path.iloc[1])
    with xr.open_dataset(target) as dataset:
        changed = dataset.load()
    changed.attrs["requested_start"] = "2021-01-01"
    changed.to_netcdf(target, engine="scipy")
    fetched = []

    def fetch(start, end):
        fetched.append(start)
        return currents((start,))

    result = download(archive, tmp_path, fetch)
    assert fetched == [pd.Timestamp("2020-01-02")]
    assert set(result.status) == {"downloaded"}


def test_transient_failure_retries_before_marking_success(archive, tmp_path):
    calls = 0

    def fetch(start, end):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OSError("temporary upstream failure")
        return currents((start,))

    result = download(archive, tmp_path, fetch, retries=3, end="2020-01-02")
    assert calls == 3
    assert result.status.tolist() == ["downloaded"]
    assert result.attempts.tolist() == [3]
    assert result.error.fillna("").tolist() == [""]


def test_failed_chunk_manifest_survives_restart_and_can_be_retried(archive, tmp_path):
    def broken(start, end):
        raise TimeoutError("bounded failure")

    result = download(archive, tmp_path, broken, retries=1)
    assert set(result.status) == {"failed"}
    assert all("bounded failure" in error for error in result.error)
    recovered = download(archive, tmp_path)
    assert set(recovered.status) == {"downloaded"}
    assert len(recovered) == 3


@pytest.mark.parametrize(
    "change",
    [
        {"model": "other_model"},
        {"end": "2020-01-05"},
        {"chunk_days": 2},
        {"settings": {"max_depth": 50}},
        {"engine": None},
    ],
)
def test_existing_archive_rejects_different_job_settings(archive, tmp_path, change):
    original = download(archive, tmp_path, settings={"max_depth": 100})
    preserved = {path: Path(path).read_bytes() for path in original.output_path}
    with pytest.raises(
        (ValueError, FileExistsError),
        match="(?i)(settings|config|archive|job|different)",
    ):
        download(archive, tmp_path, **({"settings": {"max_depth": 100}} | change))
    assert preserved == {path: Path(path).read_bytes() for path in preserved}


@pytest.mark.parametrize(
    "options",
    [
        {"workers": 0},
        {"workers": -1},
        {"workers": 1.5},
        {"workers": True},
        {"worker_index": -1},
        {"worker_index": 2},
        {"retries": 0},
        {"chunk_days": 0},
        {"start": "2020-01-04", "end": "2020-01-04"},
        {"start": "not-a-date"},
    ],
)
def test_invalid_job_rejected_before_fetch(archive, tmp_path, options):
    calls = []

    def fetch(start, end):
        calls.append(start)
        return currents((start,))

    with pytest.raises((ValueError, TypeError)):
        download(archive, tmp_path, fetch, **options)
    assert calls == []


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "all_missing",
        "no_joint_samples",
        "missing_component",
        "wrong_dimensions",
        "duplicate_time",
        "before_start",
        "at_end",
        "unordered_depth",
        "negative_depth",
        "wrong_velocity_units",
        "upward_depth",
        "infinite_coordinate",
        "infinite_velocity",
        "not_a_dataset",
    ],
)
def test_invalid_payload_never_becomes_completed_chunk(archive, tmp_path, kind):
    def fetch(start, end):
        result = currents((start,))
        if kind == "empty":
            result = result.isel(time=slice(0, 0))
        elif kind == "all_missing":
            result["U"][:] = np.nan
            result["V"][:] = np.nan
        elif kind == "no_joint_samples":
            result["U"][:, :, 0, :] = np.nan
            result["V"][:, :, 1, :] = np.nan
        elif kind == "missing_component":
            result = result.drop_vars("V")
        elif kind == "wrong_dimensions":
            result["U"] = result.U.isel(depth=0, drop=True)
        elif kind == "duplicate_time":
            result = xr.concat([result, result], dim="time")
        elif kind == "before_start":
            result = currents((start - pd.Timedelta(hours=1),))
        elif kind == "at_end":
            result = currents((end,))
        elif kind == "unordered_depth":
            result = result.isel(depth=[1, 0])
        elif kind == "negative_depth":
            result = result.assign_coords(depth=[-100, 0])
        elif kind == "wrong_velocity_units":
            result.U.attrs["units"] = "cm s-1"
        elif kind == "upward_depth":
            result.depth.attrs["positive"] = "up"
        elif kind == "infinite_coordinate":
            result = result.assign_coords(lon=[-127, np.inf])
        elif kind == "infinite_velocity":
            result["U"][0, 0, 0, 0] = np.inf
        elif kind == "not_a_dataset":
            return None
        return result

    result = download(archive, tmp_path, fetch, retries=1, end="2020-01-02")
    assert result.status.tolist() == ["failed"]
    assert result.error.iloc[0]
    assert not Path(result.output_path.iloc[0]).exists()


def test_chunk_boundaries_are_half_open_and_include_partial_final_chunk(
    archive, tmp_path
):
    observed = []

    def fetch(start, end):
        observed.append((start, end))
        return currents((start, end - pd.Timedelta(minutes=1)))

    result = download(
        archive, tmp_path, fetch, chunk_days=2, end="2020-01-06T12:00", workers=1
    )
    assert observed == [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03")),
        (pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-05")),
        (pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-06T12:00")),
    ]
    assert result.records.tolist() == [2, 2, 2]
