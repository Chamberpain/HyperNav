"""HYCOM/ESPC-D-V02 ocean-field access for HyperNav.

This module replaces the retired NCEI ERDDAP ``HYCOM_reg*_latest3d``
datasets with the current HYCOM.org ESPC-D-V02 THREDDS/OPeNDAP feeds.

The operational feed is a rolling Forecast Model Run Collection (FMRC):

* velocity: ``water_u`` and ``water_v`` in the combined ``uv3z`` dataset;
* tracers: ``water_temp`` and ``salinity`` in the combined ``ts3z`` dataset.

Set ``HYPERNAV_HYCOM_LAZY=1`` to avoid opening the remote dataset while this
module is imported.  All public download/load/profile methods initialize the
connection on first use, so lazy mode does not otherwise change their API.
"""

from __future__ import annotations

import datetime
import json
import os
import pickle
import re
import warnings
from socket import timeout as SocketTimeout
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
from urllib.error import HTTPError

import gsw
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import shapely.geometry
from GeneralUtilities.Data.Download.hycom_download import HYCOMDownloader, open_url
from requests.exceptions import RequestException

try:
    from pydap.exceptions import DapError
except ImportError:  # pragma: no cover - compatibility with older pydap
    DapError = RuntimeError

from HyperNav.Utilities.Data.UVBase import Base, UVTimeList
from HyperNav.Utilities.Data.__init__ import ROOT_DIR
from GeneralUtilities.Compute.Depth.depth_utilities import PACIOOS, ETopo1Depth
from GeneralUtilities.Compute.list import DepthList, LatList, LonList
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from GeneralUtilities.Plot.Cartopy.regional_plot import (
    CCSCartopy,
    GOMCartopy,
    KonaCartopy,
    PuertoRicoCartopy,
    USVICartopy,
    HumboldtCartopy
)


from GeneralUtilities.Data.Download.download_paths import LazyDownloadPaths

file_handler = LazyDownloadPaths(ROOT_DIR, "HYCOMBase")


def _box(lllon: float, urlon: float, lllat: float, urlat: float):
    """Return the rectangular region geometry used by the original classes."""

    polygon = shapely.geometry.Polygon(
        [
            [lllon, urlat],
            [urlon, urlat],
            [urlon, lllat],
            [lllon, lllat],
            [lllon, urlat],
        ]
    )
    return shapely.geometry.MultiPolygon([polygon])


class HYCOMBase(HYCOMDownloader, Base):
    """Regional currents/plots using the shared HYCOM downloader."""
    file_handler = file_handler
    time_list_class = UVTimeList

    def __init__(self, *args, **kwargs):
        self.__class__.initialize()
        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        cls.initialize()
        return super().load(*args, **kwargs)

    @classmethod
    def plot_sound_speed_waterfall(
        cls,
        when: Optional[datetime.datetime] = None,
        lllon: Optional[float] = None,
        urlon: Optional[float] = None,
        lllat: Optional[float] = None,
        urlat: Optional[float] = None,
        max_depth: Optional[float] = None,
        spatial_stride: Any = 1,
        temperature_kind: str = "in_situ",
        profile_order: str = "latitude",
        projection: str = "3d",
        profile_spacing: float = 1.0,
        profile_offset: float = 2.0,
        color_by: Optional[str] = "latitude",
        cmap: str = "viridis",
        linewidth: float = 0.7,
        alpha: float = 0.8,
        ax: Any = None,
        refresh_source: bool = False,
    ):
        """Create a waterfall plot of all vertical sound-speed profiles.

        The default 3-D plot keeps sound speed, profile sequence, and depth on
        separate axes.  Set ``projection='offset'`` for a conventional 2-D
        waterfall in which each successive profile is shifted horizontally by
        ``profile_offset`` m/s.  Profiles are ordered by latitude then
        longitude by default; use ``profile_order='longitude'`` to reverse
        that priority.

        Returns ``(figure, axes, profile_data)``.  ``profile_data`` is the
        dictionary returned by :meth:`get_sound_speed_profiles` after applying
        the requested profile ordering.
        """

        data = cls.get_sound_speed_profiles(
            when=when,
            lllon=lllon,
            urlon=urlon,
            lllat=lllat,
            urlat=urlat,
            max_depth=max_depth,
            spatial_stride=spatial_stride,
            temperature_kind=temperature_kind,
            refresh_source=refresh_source,
        )
        profiles = np.asarray(data["sound_speed"], dtype=float)
        depths = np.asarray(data["depths"], dtype=float)
        latitudes = np.asarray(data["latitudes"], dtype=float)
        longitudes = np.asarray(data["longitudes"], dtype=float)

        order_name = str(profile_order).strip().lower()
        if order_name in {"latitude", "lat"}:
            order = np.lexsort((longitudes, latitudes))
        elif order_name in {"longitude", "lon"}:
            order = np.lexsort((latitudes, longitudes))
        elif order_name in {"source", "none", "native"}:
            order = np.arange(profiles.shape[0])
        else:
            raise ValueError(
                "profile_order must be 'latitude', 'longitude', or 'source'"
            )
        profiles = profiles[order]
        latitudes = latitudes[order]
        longitudes = longitudes[order]
        data["sound_speed"] = profiles
        data["latitudes"] = latitudes
        data["longitudes"] = longitudes
        data["profile_order"] = order_name

        n_profiles = profiles.shape[0]
        if n_profiles > 2000:
            warnings.warn(
                "Rendering %d profiles may be slow; spatial_stride can reduce "
                "the plotted density without changing the regional bounds."
                % n_profiles,
                RuntimeWarning,
            )

        color_name = None if color_by is None else str(color_by).strip().lower()
        if color_name in {None, "none"}:
            color_values = None
            color_label = None
        elif color_name in {"latitude", "lat"}:
            color_values = latitudes
            color_label = "Latitude (degrees north)"
        elif color_name in {"longitude", "lon"}:
            color_values = longitudes
            color_label = "Longitude (degrees east)"
        elif color_name in {"profile", "index", "sequence"}:
            color_values = np.arange(n_profiles, dtype=float)
            color_label = "Profile sequence"
        else:
            raise ValueError(
                "color_by must be 'latitude', 'longitude', 'profile', or None"
            )

        projection_name = str(projection).strip().lower()
        if projection_name in {"3d", "three_dimensional", "waterfall"}:
            if float(profile_spacing) <= 0:
                raise ValueError("profile_spacing must be greater than zero")
            if ax is None:
                figure = plt.figure()
                ax = figure.add_subplot(111, projection="3d")
            else:
                figure = ax.figure
                if not hasattr(ax, "add_collection3d"):
                    raise ValueError("A 3-D Matplotlib axes is required")

            profile_positions = np.arange(n_profiles, dtype=float) * float(
                profile_spacing
            )
            segments = []
            segment_colors = []
            for profile_idx, profile in enumerate(profiles):
                finite = np.isfinite(profile) & np.isfinite(depths)
                if np.count_nonzero(finite) < 2:
                    continue
                segments.append(
                    np.column_stack(
                        (
                            profile[finite],
                            np.full(
                                np.count_nonzero(finite), profile_positions[profile_idx]
                            ),
                            depths[finite],
                        )
                    )
                )
                if color_values is not None:
                    segment_colors.append(color_values[profile_idx])

            collection = Line3DCollection(
                segments,
                cmap=cmap,
                linewidths=float(linewidth),
                alpha=float(alpha),
            )
            if color_values is not None:
                collection.set_array(np.asarray(segment_colors, dtype=float))
            ax.add_collection3d(collection)

            finite_speed = profiles[np.isfinite(profiles)]
            if finite_speed.size:
                ax.set_xlim(
                    float(np.nanmin(finite_speed)), float(np.nanmax(finite_speed))
                )
            if n_profiles == 1:
                ax.set_ylim(-0.5, 0.5)
            else:
                ax.set_ylim(profile_positions[0], profile_positions[-1])
            ax.set_zlim(float(np.nanmin(depths)), 0.0)
            ax.set_xlabel(r"Sound speed ($m\ s^{-1}$)")
            ax.set_ylabel("Profile number")
            ax.set_zlabel("Depth (m)")

            tick_count = min(7, n_profiles)
            tick_indices = np.unique(
                np.linspace(0, n_profiles - 1, tick_count, dtype=int)
            )
            ax.set_yticks(profile_positions[tick_indices])
            ax.set_yticklabels([str(index + 1) for index in tick_indices])
            data["profile_positions"] = profile_positions
            data["profile_tick_indices"] = tick_indices
        elif projection_name in {"offset", "2d", "two_dimensional"}:
            if float(profile_offset) <= 0:
                raise ValueError("profile_offset must be greater than zero")
            if ax is None:
                figure, ax = plt.subplots()
            else:
                figure = ax.figure

            offsets = np.arange(n_profiles, dtype=float) * float(profile_offset)
            segments = []
            segment_colors = []
            for profile_idx, profile in enumerate(profiles):
                finite = np.isfinite(profile) & np.isfinite(depths)
                if np.count_nonzero(finite) < 2:
                    continue
                segments.append(
                    np.column_stack(
                        (profile[finite] + offsets[profile_idx], depths[finite])
                    )
                )
                if color_values is not None:
                    segment_colors.append(color_values[profile_idx])

            collection = LineCollection(
                segments,
                cmap=cmap,
                linewidths=float(linewidth),
                alpha=float(alpha),
            )
            if color_values is not None:
                collection.set_array(np.asarray(segment_colors, dtype=float))
            ax.add_collection(collection)
            ax.autoscale_view()
            ax.set_ylim(float(np.nanmin(depths)), 0.0)
            ax.set_xlabel(
                r"Sound speed ($m\ s^{-1}$) + %.3g per profile" % float(profile_offset)
            )
            ax.set_ylabel("Depth (m)")
            data["profile_offsets"] = offsets
        else:
            raise ValueError("projection must be '3d' or 'offset'")

        west, east, south, north = data["bounds"]
        ax.set_title(
            "%s sound-speed profiles at %s\n"
            "%.2f° to %.2f° lon, %.2f° to %.2f° lat"
            % (
                cls.location,
                data["time"].strftime("%Y-%m-%d %H:%M UTC"),
                west,
                east,
                south,
                north,
            )
        )
        if color_values is not None and len(segments):
            colorbar = figure.colorbar(collection, ax=ax, pad=0.1)
            colorbar.set_label(color_label)
        if projection_name in {"offset", "2d", "two_dimensional"}:
            figure.tight_layout()
        return figure, ax, data

    @classmethod
    def get_sal_temp_profiles(
        cls, lat, lon, start_date, end_date, refresh_source: bool = False
    ):
        """Plot mean salinity, temperature, and sigma0 from the tracer feed."""

        start_date = cls._normalise_datetime(start_date)
        end_date = cls._normalise_datetime(end_date)
        if start_date >= end_date:
            raise ValueError("start_date must be earlier than end_date")

        tracer_dataset = cls.get_tracer_dataset(force=refresh_source)
        metadata = cls._source_metadata(
            tracer_dataset,
            cls.tracer_url,
            unit_variable="water_temp",
            force=refresh_source,
        )
        tracer_times = list(metadata["time"])
        available_start, available_end = min(tracer_times), max(tracer_times)
        if start_date < available_start or end_date > available_end:
            raise ValueError(
                "Requested profile interval %s to %s is outside the rolling "
                "ESPC-D-V02 tracer window %s to %s.  Use archive_url() for "
                "older records."
                % (start_date, end_date, available_start, available_end)
            )

        time_start_idx = min(
            range(len(tracer_times)),
            key=lambda index: abs(tracer_times[index] - start_date),
        )
        time_end_idx = min(
            range(len(tracer_times)),
            key=lambda index: abs(tracer_times[index] - end_date),
        )
        if time_end_idx < time_start_idx:
            time_start_idx, time_end_idx = time_end_idx, time_start_idx
        time_stop = time_end_idx + 1

        lon_idx = cls._nearest_index(metadata["lons"], lon)
        lat_idx = cls._nearest_index(metadata["lats"], lat)
        depth_idx = cls._nearest_index(metadata["depths"], -700.0)
        depth_stop = depth_idx + 1

        index = (
            slice(time_start_idx, time_stop),
            slice(0, depth_stop),
            slice(lat_idx, lat_idx + 1),
            slice(lon_idx, lon_idx + 1),
        )
        salinity = cls._clean_numeric_array(
            cls._read_grid(tracer_dataset, "salinity", index)
        )
        temperature = cls._clean_numeric_array(
            cls._read_grid(tracer_dataset, "water_temp", index)
        )

        # Preserve time/depth even when pydap squeezes singleton lat/lon axes.
        salinity = np.asarray(salinity).reshape(
            time_stop - time_start_idx, depth_stop, -1
        )
        temperature = np.asarray(temperature).reshape(
            time_stop - time_start_idx, depth_stop, -1
        )
        sal_data = np.nanmean(salinity, axis=(0, 2))
        temp_data = np.nanmean(temperature, axis=(0, 2))
        profile_depths = metadata["depths"][:depth_stop]

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Salinity (psu)", color="tab:red")
        ax1.set_ylabel("Depth (m)")
        ax1.plot(sal_data, profile_depths, color="tab:red")
        ax1.tick_params(axis="x", labelcolor="tab:red")

        ax2 = ax1.twiny()
        ax2.set_xlabel(r"Temperature ($^\circ$C)", color="tab:blue")
        ax2.plot(temp_data, profile_depths, color="tab:blue")
        ax2.tick_params(axis="x", labelcolor="tab:blue")
        fig.tight_layout()

        _, absolute_salinity, conservative_temperature, _ = cls._teos10_sound_speed(
            sal_data[:, np.newaxis, np.newaxis],
            temp_data[:, np.newaxis, np.newaxis],
            profile_depths,
            [float(metadata["lats"][lat_idx])],
            [float(metadata["lons"][lon_idx])],
            temperature_kind="in_situ",
        )
        density = gsw.sigma0(
            absolute_salinity[:, 0, 0],
            conservative_temperature[:, 0, 0],
        )
        fig1, density_axis = plt.subplots()
        density_axis.plot(density, profile_depths)
        density_axis.set_xlabel(r"$\sigma_0\ (kg\ m^{-3})$")
        density_axis.set_ylabel("Depth (m)")
        return fig, fig1

    @classmethod
    def get_dataset_shape(cls):
        cls.initialize()
        source_url = cls._resolve_source(cls.ID)
        metadata = cls._source_metadata(
            cls.dataset, source_url, unit_variable="water_u"
        )
        lllat = float(np.nanmin(metadata["lats"]))
        urlat = float(np.nanmax(metadata["lats"]))
        # The source is global on a 0..360 grid; expose the conventional
        # -180..180 footprint used elsewhere in HyperNav.
        return _box(-180.0, 180.0, lllat, urlat)


class HYCOMAlaska(HYCOMBase):
    location = "Alaska"
    facecolor = "brown"
    urlat = 75
    lllat = 70
    lllon = -155
    urlon = -128
    max_depth = -1500
    ocean_shape = _box(lllon, urlon, lllat, urlat)
    ID = HYCOMBase.ID
    DepthClass = ETopo1Depth


class HYCOMSouthernCalifornia(HYCOMBase):
    location = "SoCal"
    facecolor = "Pink"
    urlat = 35
    lllat = 30
    lllon = -122
    urlon = -116
    max_depth = -700
    PlotClass = CCSCartopy
    ocean_shape = _box(lllon, urlon, lllat, urlat)
    ID = HYCOMBase.ID
    DepthClass = ETopo1Depth


class HYCOMGOM(HYCOMBase):
    location = "GOM"
    urlat = 28
    lllat = 26
    lllon = -93
    urlon = -90.5
    max_depth = -2500
    ocean_shape = _box(lllon, urlon, lllat, urlat)
    ID = HYCOMBase.ID
    PlotClass = GOMCartopy
    DepthClass = ETopo1Depth


class HYCOMMonterey(HYCOMBase):
    location = "Monterey"
    facecolor = "Pink"
    urlat = 39
    lllat = 34
    lllon = -126
    urlon = -121.5
    max_depth = -700
    PlotClass = CCSCartopy
    ocean_shape = _box(lllon, urlon, lllat, urlat)
    ID = HYCOMBase.ID
    DepthClass = ETopo1Depth


class HYCOMHawaii(HYCOMBase):
    location = "Hawaii"
    facecolor = "blue"
    urlat = 22
    lllat = 16
    lllon = -159
    urlon = -154
    max_depth = -2500
    ocean_shape = _box(lllon, urlon, lllat, urlat)
    ID = HYCOMBase.ID
    PlotClass = KonaCartopy
    DepthClass = PACIOOS


class HYCOMPuertoRico(HYCOMBase):
    location = "PuertoRico"
    facecolor = "yellow"
    urlon = -65
    lllon = -68.5
    urlat = 22.5
    lllat = 16
    max_depth = -700
    ocean_shape = _box(lllon, urlon, lllat, urlat)
    ID = HYCOMBase.ID
    PlotClass = PuertoRicoCartopy
    DepthClass = ETopo1Depth

class HYCOMUSVI(HYCOMBase):
    location = "USVI"
    facecolor = "yellow"
    urlon = -64.0
    lllon = -65.5
    urlat = 18.5
    lllat = 17.5
    max_depth = -700
    ocean_shape = _box(lllon, urlon, lllat, urlat)
    ID = HYCOMBase.ID
    PlotClass = USVICartopy
    DepthClass = ETopo1Depth

class HYCOMHumboldt(HYCOMBase):
    location = "Humboldt"
    facecolor = "yellow"
    urlon = -123.9
    lllon = -127.0
    urlat = 43.0
    lllat = 38.9
    max_depth = -5000
    ocean_shape = _box(lllon, urlon, lllat, urlat)
    ID = HYCOMBase.ID
    PlotClass = HumboldtCartopy
    DepthClass = ETopo1Depth


HYCOM_REGIONS = (
    HYCOMAlaska,
    HYCOMSouthernCalifornia,
    HYCOMGOM,
    HYCOMMonterey,
    HYCOMHawaii,
    HYCOMPuertoRico,
    HYCOMUSVI,
)


def _eager_initialize_regions():
    lazy = os.environ.get("HYPERNAV_HYCOM_LAZY", "0").strip().lower()
    if lazy in {"1", "true", "yes", "on"}:
        return
    try:
        for region_class in HYCOM_REGIONS:
            region_class.initialize()
    except Exception as exc:
        # A transient remote outage should not make the whole HyperNav package
        # unimportable.  The first method call will retry initialization.
        warnings.warn(
            "HYCOM ESPC-D-V02 was not reachable during import; initialization "
            "will be retried on first use: %s" % exc,
            RuntimeWarning,
        )


_eager_initialize_regions()
