"""Offline command-line checks without importing model SDKs or remote sources."""

import builtins
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pandas as pd
import pytest


@pytest.fixture
def cli(monkeypatch):
    package_name = "_current_archive_cli_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str((Path(__file__).resolve().parents[1] / "Utilities" / "Data"))]
    monkeypatch.setitem(sys.modules, package_name, package)
    name = f"{package_name}.current_archive"
    spec = importlib.util.spec_from_file_location(
        name, (Path(__file__).resolve().parents[1] / "Utilities" / "Data") / "current_archive.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    models = {}
    for filename, class_name in (
        ("CopernicusGlobal", "HumboldtCopernicus"),
        ("WCOFS", "WCOFSHumboldt"),
    ):
        adapter = types.ModuleType(f"{package_name}.{filename}")
        download = Mock(return_value=pd.DataFrame({"status": ["downloaded"]}))
        setattr(adapter, class_name, types.SimpleNamespace(download_record=download))
        monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
        models[filename] = download
    return module, models


def command(tmp_path, model="copernicus", *extra):
    return [
        model,
        "--start",
        "2023-12-16",
        "--end",
        "2024-05-15",
        "--output-dir",
        str(tmp_path / "archive"),
        *extra,
    ]


def block_adapter_imports(monkeypatch):
    real_import = builtins.__import__

    def checked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level and name in {"CopernicusGlobal", "WCOFS"}:
            raise AssertionError(
                "Argument parsing should finish before importing an adapter."
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", checked_import)


def test_help_exits_successfully_without_adapter_imports(cli, monkeypatch, capsys):
    archive, models = cli
    block_adapter_imports(monkeypatch)
    with pytest.raises(SystemExit) as exit_info:
        archive.main(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--worker-index" in help_text
    assert "--source-engine" in help_text
    assert "--time-step" in help_text
    assert all(not download.called for download in models.values())


def test_copernicus_routes_shared_and_dataset_options(cli, tmp_path):
    archive, models = cli
    result = archive.main(
        command(
            tmp_path,
            "copernicus",
            "--workers",
            "6",
            "--worker-index",
            "2",
            "--chunk-days",
            "10",
            "--retries",
            "4",
            "--max-depth",
            "500",
            "--dataset-id",
            "custom_product",
            "--time-step",
            "6h",
        )
    )
    assert result == 0
    models["CopernicusGlobal"].assert_called_once_with(
        "2023-12-16",
        "2024-05-15",
        tmp_path / "archive",
        workers=6,
        worker_index=2,
        chunk_days=10,
        retries=4,
        max_depth=500.0,
        dataset_id="custom_product",
        time_step="6h",
    )
    assert not models["WCOFS"].called


def test_copernicus_defaults_leave_dataset_and_cadence_to_adapter(cli, tmp_path):
    archive, models = cli
    assert archive.main(command(tmp_path)) == 0
    models["CopernicusGlobal"].assert_called_once_with(
        "2023-12-16",
        "2024-05-15",
        tmp_path / "archive",
        workers=4,
        worker_index=None,
        retries=3,
        max_depth=800.0,
    )


def test_wcofs_defaults_to_historical_roms(cli, tmp_path):
    archive, models = cli
    assert archive.main(command(tmp_path, "wcofs")) == 0
    models["WCOFS"].assert_called_once_with(
        "2023-12-16",
        "2024-05-15",
        tmp_path / "archive",
        workers=4,
        worker_index=None,
        retries=3,
        max_depth=800.0,
        source="roms",
    )
    assert not models["CopernicusGlobal"].called


def test_wcofs_routes_source_regridding_and_local_reader_options(cli, tmp_path):
    archive, models = cli
    assert (
        archive.main(
            command(
                tmp_path,
                "wcofs",
                "--workers",
                "3",
                "--source",
                "roms",
                "--url-template",
                "/data/{filename}",
                "--source-engine",
                "scipy",
                "--depth-levels",
                "5",
                "10",
                "20",
                "--grid-spacing",
                "0.1",
            )
        )
        == 0
    )
    models["WCOFS"].assert_called_once_with(
        "2023-12-16",
        "2024-05-15",
        tmp_path / "archive",
        workers=3,
        worker_index=None,
        retries=3,
        max_depth=800.0,
        source="roms",
        url_template="/data/{filename}",
        engine="scipy",
        depth_levels=[5.0, 10.0, 20.0],
        grid_spacing=0.1,
    )


def test_wcofs_can_select_recent_regulargrid_source(cli, tmp_path):
    archive, models = cli
    assert archive.main(command(tmp_path, "wcofs", "--source", "regulargrid")) == 0
    assert models["WCOFS"].call_args.kwargs["source"] == "regulargrid"


@pytest.mark.parametrize(
    "extra",
    [
        ["--source", "roms"],
        ["--url-template", "/data/{filename}"],
        ["--depth-levels", "5", "10"],
        ["--grid-spacing", "0.1"],
        ["--source-engine", "scipy"],
    ],
)
def test_copernicus_rejects_wcofs_options_before_adapter_import(
    cli, tmp_path, monkeypatch, capsys, extra
):
    archive, models = cli
    block_adapter_imports(monkeypatch)
    with pytest.raises(SystemExit) as exit_info:
        archive.main(command(tmp_path, "copernicus", *extra))
    assert exit_info.value.code == 2
    assert "apply only to wcofs" in capsys.readouterr().err
    assert all(not download.called for download in models.values())


@pytest.mark.parametrize(
    "extra", [["--dataset-id", "custom_product"], ["--time-step", "6h"]]
)
def test_wcofs_rejects_copernicus_options_before_adapter_import(
    cli, tmp_path, monkeypatch, capsys, extra
):
    archive, models = cli
    block_adapter_imports(monkeypatch)
    with pytest.raises(SystemExit) as exit_info:
        archive.main(command(tmp_path, "wcofs", *extra))
    assert exit_info.value.code == 2
    assert "applies only to Copernicus" in capsys.readouterr().err
    assert all(not download.called for download in models.values())


@pytest.mark.parametrize(
    "model,adapter", [("copernicus", "CopernicusGlobal"), ("wcofs", "WCOFS")]
)
@pytest.mark.parametrize("worker_index,expected", [(0, 0), (1, 1)])
def test_terminal_worker_exit_status_only_considers_assigned_rows(
    cli, tmp_path, model, adapter, worker_index, expected
):
    archive, models = cli
    models[adapter].return_value = pd.DataFrame(
        {"status": ["downloaded", "failed", "downloaded", "pending"]}
    )
    assert (
        archive.main(
            command(
                tmp_path, model, "--workers", "2", "--worker-index", str(worker_index)
            )
        )
        == expected
    )


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (["downloaded", "downloaded"], 0),
        (["downloaded", "failed"], 1),
        (["downloaded", "pending"], 1),
    ],
)
def test_local_thread_pool_exit_status_includes_every_chunk(
    cli, tmp_path, statuses, expected
):
    archive, models = cli
    models["CopernicusGlobal"].return_value = pd.DataFrame({"status": statuses})
    assert archive.main(command(tmp_path)) == expected
