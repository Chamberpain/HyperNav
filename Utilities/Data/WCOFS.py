"""WCOFS currents on physical depth/latitude/longitude grids.

``WCOFSHumboldt.download_record(start, end, output_dir, workers=4)`` saves
restartable, daily NetCDF chunks with U/V(time, depth, lat, lon), velocities
in m/s and positive-down depths in metres. Dates are UTC and end is exclusive.
Record downloads default to NCEI's historical raw ROMS nowcasts, destaggered,
rotated and interpolated to physical depths and a geographic grid. The optional
``source='regulargrid'`` uses NOAA's ready-interpolated files, retained for
approximately three days. Retained local archives accept ``url_template``.
See https://tidesandcurrents.noaa.gov/ofs/ofs_faq.html for archive retention.
Importing this module opens no remote datasets and creates no cache folders.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shapely.geometry
import xarray as xr

from HyperNav.Utilities.Data.UVBase import Base, UVTimeList
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Compute.list import DepthList, LatList, LonList, TimeList

try:
    from GeneralUtilities.Plot.Cartopy.regional_plot import CCSCartopy
except ImportError:
    CCSCartopy = None
try:
    from GeneralUtilities.Compute.Depth.depth_utilities import ETopo1Depth
except ImportError:
    ETopo1Depth = None

from GeneralUtilities.Data.Download.wcofs_download import WCOFSDownloader, _utc
from .current_archive import NETCDF_LOCK, download_record as download_archive


class _LazyFileHandler:
    _handler = None

    def __getattr__(self, name):
        # ABCMeta and other introspection tools probe special attributes while
        # the owning class is still being defined. Those probes must stay lazy.
        if name.startswith("__"):
            raise AttributeError(name)

        if self._handler is None:
            from GeneralUtilities.Data.Filepath.instance import FilePathHandler

            from GeneralUtilities.Data.Download.download_paths import make_download_paths

            self._handler = make_download_paths(ROOT_DIR, "WGOFS")
        return getattr(self._handler, name)


file_handler = _LazyFileHandler()


def _box(west, east, south, north):
    return shapely.geometry.MultiPolygon(
        [shapely.geometry.box(west, south, east, north)]
    )


class WCOFSBase(WCOFSDownloader, Base):
    # Keep WGOFS to preserve existing HyperNav pickle cache paths.
    file_handler = file_handler
    dataset = None
    dataset_time = None
    lats = None
    lons = None
    depths = None
    units = "m/s"
    ref_date = None
    _initialized = False

    def __init__(self, *args, **kwargs):
        if self.__class__.lats is None:
            raise ValueError(
                "Initialize regional axes or load a local record before constructing WCOFS"
            )
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize(cls, valid_time=None, *, force=False):
        """Load regional axes only when explicitly needed by legacy pickle APIs."""
        if cls.__dict__.get("_initialized", False) and not force:
            return cls
        valid_time = _utc(
            valid_time
            if valid_time is not None
            else pd.Timestamp.now(tz="UTC").normalize()
        )
        with cls.get_dataset(valid_time) as source:
            cls.dataset = cls._normalise(
                source,
                bounds=(cls.lllon, cls.urlon, cls.lllat, cls.urlat),
                max_depth=cls.max_depth,
            )
        cls.lats = LatList(cls.dataset.lat.values.tolist())
        cls.lons = LonList(cls.dataset.lon.values.tolist())
        cls.depths = DepthList((-cls.dataset.depth.values).tolist())
        cls.ref_date = valid_time.to_pydatetime()
        cls.dataset_time = UVTimeList(
            [
                cls.ref_date + cls.time_step * index
                for index in range(len(cls.hours_list))
            ]
        )
        cls._initialized = True
        return cls

    @classmethod
    def get_dimensions(cls, urlon, lllon, urlat, lllat, max_depth, dataset):
        result = cls._normalise(
            dataset, bounds=(lllon, urlon, lllat, urlat), max_depth=max_depth
        )
        time = pd.DatetimeIndex(result.time.values).to_pydatetime().tolist()
        return (
            UVTimeList(time),
            LatList(result.lat.values.tolist()),
            LonList(result.lon.values.tolist()),
            DepthList((-result.depth.values).tolist()),
            0,
            result.sizes["lon"],
            0,
            result.sizes["lat"],
            "m/s",
            time[0],
        )

    @classmethod
    def _load_pickles(cls, times, filename):
        records = []
        for time in times:
            path = Path(filename(time))
            if not path.exists():
                continue
            with path.open("rb") as handle:
                records.append(pickle.load(handle))
        if not records:
            raise FileNotFoundError("No WCOFS cached current records were found")
        if cls.lats is None:
            raise ValueError(
                "Call initialize(date) to establish axes for legacy pickle caches"
            )
        return cls(
            u=np.stack([record["u"] for record in records]) * cls.scale_factor,
            v=np.stack([record["v"] for record in records]) * cls.scale_factor,
            time=[record["time"] for record in records],
        )

    @classmethod
    def load(cls):
        if cls.dataset_time is None:
            cls.initialize()
        return cls._load_pickles(
            cls.dataset_time,
            lambda time: cls.file_handler.tmp_file(cls.make_k_filename(time)),
        )

    @classmethod
    def load_date(cls, date):
        times = [date + cls.time_step * index for index in range(25)]
        return cls._load_pickles(
            times,
            lambda time: cls.file_handler.tmp_file(
                f"{cls.dataset_description}_{cls.location}_data_{date.strftime('%b-')}{date.day}/{time}"
            ),
        )

    @classmethod
    def get_dataset_shape(cls):
        if cls.dataset is None:
            cls.initialize()
        return _box(
            float(cls.dataset.lon.min()),
            float(cls.dataset.lon.max()),
            float(cls.dataset.lat.min()),
            float(cls.dataset.lat.max()),
        )


class WCOFSSouthernCalifornia(WCOFSBase):
    location = "SoCal"
    facecolor = "Pink"
    urlat, lllat, lllon, urlon, max_depth = 35, 30, -122, -116.5, -700
    PlotClass = CCSCartopy
    DepthClass = ETopo1Depth
    ocean_shape = _box(lllon, urlon, lllat, urlat)


class WCOFSSouthernCaliforniaHistorical(WCOFSSouthernCalifornia):
    urlat, lllat, lllon, urlon, max_depth = 33.7, 32.5, -118, -117, -500
    ocean_shape = _box(lllon, urlon, lllat, urlat)

    @classmethod
    def load(cls):
        # XROM_Utilities imports optional xroms; defer its historical metadata.
        from HyperNav.Utilities.Data.XROM_Utilities import return_dims, dataset_time

        lats, lons, depths = return_dims(
            cls.lllon, cls.urlon, cls.lllat, cls.urlat, cls.max_depth
        )
        cls.lats, cls.lons, cls.depths = (
            LatList(lats.tolist()),
            LonList(lons.tolist()),
            DepthList(depths.tolist()),
        )
        folder = Path(
            cls.file_handler.tmp_file(
                f"{cls.dataset_description}_{cls.location}_historical_data"
            )
        )
        return cls(
            u=np.load(folder / "u.npy") * cls.scale_factor,
            v=np.load(folder / "v.npy") * cls.scale_factor,
            time=TimeList(dataset_time),
        )


class WCOFSMonterey(WCOFSBase):
    location = "Monterey"
    facecolor = "Brown"
    urlat, lllat, lllon, urlon, max_depth = 39, 34, -126, -121.5, -700
    PlotClass = CCSCartopy
    DepthClass = ETopo1Depth
    ocean_shape = _box(lllon, urlon, lllat, urlat)
    ID = "HYCOM_reg7_latest3d"


class WCOFSMontereyHistorical(WCOFSMonterey):
    load = classmethod(WCOFSSouthernCaliforniaHistorical.load.__func__)


class WCOFSHumboldt(WCOFSBase):
    location = "Humboldt"
    facecolor = "Brown"
    lllon, urlon, lllat, urlat, max_depth = -127, -123.9, 38.9, 43, -700
    PlotClass = CCSCartopy
    DepthClass = ETopo1Depth
    ocean_shape = _box(lllon, urlon, lllat, urlat)
