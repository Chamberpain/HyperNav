"""Copernicus currents, with explicit initialization and parallel NetCDF archives.

Authenticate once with ``copernicusmarine login`` or the SDK's documented
COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD variables.
Importing this module never authenticates or opens a remote dataset.
"""

import datetime
import importlib
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely.geometry

from GeneralUtilities.Compute.list import DepthList, LatList, LonList

from .UVBase import Base, UVTimeList
from GeneralUtilities.Data.Download.copernicus_global_download import CopernicusDownloader


class _LazyResource:
    """Load optional plotting/depth dependencies or legacy paths on first use."""

    def __init__(self, module, name, *, file_handler=False):
        self.module = module
        self.name = name
        self.file_handler = file_handler
        self.value = None

    def __get__(self, instance, owner):
        if self.value is None:
            value = getattr(importlib.import_module(self.module), self.name)
            if self.file_handler:
                from GeneralUtilities.Data.Download.download_paths import make_download_paths
                value = make_download_paths(str(Path(__file__).resolve().parent), "Copernicus")
            self.value = value
        return self.value


class CopUVTimeList(UVTimeList):
    def return_time_list(self):
        if not self:
            return []
        indices = list(range(0, len(self), 40))
        if indices[-1] != len(self) - 1:
            indices.append(len(self) - 1)
        return indices


def nanosecond_convert(time_list, ref_date):
    return [
        (ref_date + np.timedelta64(int(value), "ns"))
        .astype("datetime64[us]")
        .astype(datetime.datetime)
        for value in time_list
    ]


class CopernicusGlobal(CopernicusDownloader, Base):
    @staticmethod
    def _archive_downloader():
        from .current_archive import download_record
        return download_record

    facecolor = "green"
    dataset_description = "GOPAF"
    DepthClass = _LazyResource(
        "GeneralUtilities.Compute.Depth.depth_utilities", "ETopo1Depth"
    )
    file_handler = _LazyResource(
        "GeneralUtilities.Data.Filepath.instance", "FilePathHandler", file_handler=True
    )
    time_method = staticmethod(nanosecond_convert)

    def __init__(self, *args, **kwargs):
        self.__class__.initialize()
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize(cls, *, start_date="2022-06-01", end_date=None, force=False):
        """Initialize metadata for the legacy Base interface only when requested."""
        if not force and "dataset" in cls.__dict__:
            return cls.dataset
        dataset = cls.get_dataset(
            cls.urlat,
            cls.lllat,
            cls.urlon,
            cls.lllon,
            cls.max_depth,
            cls.ID,
            start_date,
            end_date,
        )
        dimensions = cls.get_dimensions(
            cls.urlon, cls.lllon, cls.urlat, cls.lllat, cls.max_depth, dataset
        )
        if force and "dataset" in cls.__dict__:
            cls.dataset.close()
        cls.dataset = dataset
        (
            cls.dataset_time,
            cls.lats,
            cls.lons,
            cls.depths,
            cls.lllon_idx,
            cls.urlon_idx,
            cls.lllat_idx,
            cls.urlat_idx,
            cls.units,
            cls.ref_date,
        ) = dimensions
        return dataset

    @classmethod
    def get_dataset_shape(cls):
        dataset = cls.initialize()
        west, east = float(dataset.longitude.min()), float(dataset.longitude.max())
        south, north = float(dataset.latitude.min()), float(dataset.latitude.max())
        return shapely.geometry.MultiPolygon(
            [shapely.geometry.box(west, south, east, north)]
        )

    @classmethod
    def get_dimensions(cls, urlon, lllon, urlat, lllat, max_depth, dataset):
        time_values = np.asarray(dataset.time.values).astype("datetime64[us]")
        times = CopUVTimeList(time_values.astype(datetime.datetime).tolist())
        lats = LatList(dataset.latitude.values.tolist())
        lons = LonList(dataset.longitude.values.tolist())
        depths = DepthList((-np.asarray(dataset.depth.values)).tolist())
        units = dataset.uo.attrs.get("units", "")
        return (
            times,
            lats,
            lons,
            depths,
            0,
            -1,
            0,
            -1,
            units,
            np.datetime64("1970-01-01T00:00:00"),
        )

    @classmethod
    def load(cls, *args, **kwargs):
        cls.initialize()
        return super().load(*args, **kwargs)


_PLOT_MODULE = "GeneralUtilities.Plot.Cartopy.regional_plot"


class SoCalCopernicus(CopernicusGlobal):
    location = "SouthernCalifornia"
    facecolor = "Pink"
    urlat, lllat = 35, 30
    lllon, urlon = -122, -116.5
    PlotClass = _LazyResource(_PLOT_MODULE, "CCSCartopy")
    ocean_shape = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(lllon, lllat, urlon, urlat)]
    )


class MontereyCopernicus(CopernicusGlobal):
    location = "Monterey"
    facecolor = "Pink"
    urlat, lllat = 39, 34
    lllon, urlon = -126, -121.5
    max_depth = 700
    PlotClass = _LazyResource(_PLOT_MODULE, "CCSCartopy")
    ocean_shape = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(lllon, lllat, urlon, urlat)]
    )


class HumboldtCopernicus(CopernicusGlobal):
    location = "Humboldt"
    urlat, lllat = 43.0, 38.9
    lllon, urlon = -127.0, -123.9
    PlotClass = _LazyResource(_PLOT_MODULE, "CCSCartopy")
    ocean_shape = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(lllon, lllat, urlon, urlat)]
    )


class PuertoRicoCopernicus(CopernicusGlobal):
    location = "PuertoRico"
    facecolor = "yellow"
    urlat, lllat = 22.5, 16
    lllon, urlon = -68.5, -65
    max_depth = 700
    PlotClass = _LazyResource(_PLOT_MODULE, "PuertoRicoCartopy")
    ocean_shape = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(lllon, lllat, urlon, urlat)]
    )


class TahitiCopernicus(CopernicusGlobal):
    location = "Tahiti"
    urlat, lllat = -15, -21
    lllon, urlon = -152.5, -147.0
    PlotClass = _LazyResource(_PLOT_MODULE, "TahitiCartopy")
    ocean_shape = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(lllon, lllat, urlon, urlat)]
    )


class HawaiiCopernicus(CopernicusGlobal):
    location = "Hawaii"
    urlat, lllat = 22, 16
    lllon, urlon = -158, -154
    PlotClass = _LazyResource(_PLOT_MODULE, "KonaCartopy")
    ocean_shape = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(lllon, lllat, urlon, urlat)]
    )


class HawaiiOffshoreCopernicus(CopernicusGlobal):
    location = "HawaiiOffshore"
    urlat, lllat = 17.5, 14.5
    lllon, urlon = -158, -154
    PlotClass = _LazyResource(_PLOT_MODULE, "KonaCartopy")
    ocean_shape = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(lllon, lllat, urlon, urlat)]
    )


class BermudaCopernicus(CopernicusGlobal):
    location = "Bermuda"
    urlat, lllat = 34.5, 29.5
    lllon, urlon = -67, -62
    PlotClass = _LazyResource(_PLOT_MODULE, "BermudaCartopy")
    ocean_shape = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(lllon, lllat, urlon, urlat)]
    )


class CanaryCopernicus(CopernicusGlobal):
    location = "Canary"
    urlat, lllat = 30, 25
    lllon, urlon = -19, -14
    PlotClass = _LazyResource(_PLOT_MODULE, "CanaryCartopy")
    ocean_shape = shapely.geometry.MultiPolygon(
        [shapely.geometry.box(lllon, lllat, urlon, urlat)]
    )
