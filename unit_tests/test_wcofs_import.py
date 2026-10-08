"""Import the real WCOFS adapter with its real ABC base in fresh interpreters."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def run_in_real_package(body, tmp_path):
    """Use this directory's adapter and the installed HyperNav package helpers."""
    package = importlib.util.find_spec("HyperNav")
    assert package is not None and package.origin is not None
    projects = Path(package.origin).resolve().parent.parent
    adapter_dir = (Path(__file__).resolve().parents[1] / "Utilities" / "Data")
    setup = f"""
import abc
import importlib
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, {str(projects)!r})
import pandas as pd
import xarray as xr
import HyperNav.Utilities.Data as data_package
from HyperNav.Utilities.Data.UVBase import Base
import GeneralUtilities.Data.Filepath.instance as path_module

assert isinstance(Base, abc.ABCMeta)
data_package.__path__.insert(0, {str(adapter_dir)!r})
factory = Mock(side_effect=AssertionError("Cache creation during import"))
class FakeFilePathHandler:
    def __init__(self, root, name):
        self.__dict__.update(vars(factory(root, name)))
path_module.FilePathHandler = FakeFilePathHandler
blocked = Mock(side_effect=AssertionError("Network or dataset access during import"))
socket.create_connection = blocked
socket.socket.connect = blocked
xr.open_dataset = blocked
xr.open_mfdataset = blocked

def load_adapter():
    module = importlib.import_module("HyperNav.Utilities.Data.WCOFS")
    assert Path(module.__file__).resolve() == Path({str(adapter_dir / "WCOFS.py")!r})
    assert issubclass(module.WCOFSBase, Base)
    factory.assert_not_called()
    blocked.assert_not_called()
    return module
"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(setup) + textwrap.dedent(body)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_abc_import_creates_no_cache_or_remote_dataset(tmp_path):
    run_in_real_package(
        """
wcofs = load_adapter()
assert issubclass(wcofs.WCOFSHumboldt, Base)
assert wcofs.WCOFSBase.file_handler is wcofs.file_handler
assert wcofs.WCOFSHumboldt.file_handler is wcofs.file_handler
assert not wcofs.WCOFSBase.__abstractmethods__
assert wcofs.WCOFSHumboldt.dataset is None
""",
        tmp_path,
    )


def test_introspection_never_initializes_lazy_handler(tmp_path):
    run_in_real_package(
        """
wcofs = load_adapter()
for name in ("__isabstractmethod__", "__wrapped__", "__signature__"):
    assert not hasattr(wcofs.file_handler, name)
    factory.assert_not_called()
blocked.assert_not_called()
""",
        tmp_path,
    )


def test_handler_initializes_once_without_replacing_shared_proxy(tmp_path):
    run_in_real_package(
        """
wcofs = load_adapter()
proxy = wcofs.file_handler
handler = SimpleNamespace(tmp_file=Mock(return_value="cached-file"), marker="cached")
factory.side_effect = None
factory.return_value = handler
assert proxy.tmp_file("first") == "cached-file"
assert wcofs.WCOFSHumboldt.file_handler.marker == "cached"
assert proxy.tmp_file("second") == "cached-file"
try:
    proxy.missing_attribute
except AttributeError:
    pass
else:
    raise AssertionError("Missing ordinary attributes must raise AttributeError")
assert not hasattr(proxy, "__isabstractmethod__")
factory.assert_called_once_with(wcofs.ROOT_DIR, "WGOFS")
assert handler.tmp_file.call_args_list[0].args == ("first",)
assert handler.tmp_file.call_args_list[1].args == ("second",)
assert wcofs.file_handler is proxy
assert wcofs.WCOFSBase.file_handler is proxy
assert wcofs.WCOFSHumboldt.file_handler is proxy
blocked.assert_not_called()
""",
        tmp_path,
    )


def test_real_wcofs_cli_routes_user_command_to_download_record(tmp_path):
    run_in_real_package(
        """
wcofs = load_adapter()
archive = importlib.import_module("HyperNav.Utilities.Data.current_archive")
download = Mock(return_value=pd.DataFrame({"status": ["downloaded"]}))
wcofs.WCOFSHumboldt.download_record = download
output_dir = Path("wcofs_humboldt")
assert archive.main([
    "wcofs", "--start", "2023-12-16", "--end", "2024-05-15",
    "--workers", "4", "--output-dir", str(output_dir)
]) == 0
download.assert_called_once_with(
    "2023-12-16", "2024-05-15", output_dir,
    workers=4, worker_index=None, retries=3, max_depth=800.0, source="roms"
)
factory.assert_not_called()
blocked.assert_not_called()
""",
        tmp_path,
    )
