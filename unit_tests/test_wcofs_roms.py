"""Analytic offline checks of native ROMS rotation and physical regridding."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import xarray as xr


@pytest.fixture(scope="module")
def roms():
    name = "_test_wcofs_roms"
    spec = importlib.util.spec_from_file_location(
        name, (Path(__file__).resolve().parents[1] / "Utilities" / "Data") / "wcofs_roms.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def native_dataset(*, transform=2, angle=0.0, zeta=(0.0,)):
    ny = nx = 7
    lon, lat = np.meshgrid(-127 + np.arange(nx) * 0.1, 40 + np.arange(ny) * 0.1)
    shape_u, shape_v = (len(zeta), 3, ny, nx - 1), (len(zeta), 3, ny - 1, nx)
    return xr.Dataset(
        {
            "u": (
                ("ocean_time", "s_rho", "eta_u", "xi_u"),
                np.ones(shape_u),
                {"units": "m s-1"},
            ),
            "v": (
                ("ocean_time", "s_rho", "eta_v", "xi_v"),
                np.full(shape_v, 2.0),
                {"units": "m s-1"},
            ),
            "h": (("eta_rho", "xi_rho"), np.full((ny, nx), 100.0), {"units": "m"}),
            "angle": (
                ("eta_rho", "xi_rho"),
                np.full((ny, nx), angle),
                {"units": "radians"},
            ),
            "lon_rho": (("eta_rho", "xi_rho"), lon, {"units": "degrees_east"}),
            "lat_rho": (("eta_rho", "xi_rho"), lat, {"units": "degrees_north"}),
            "zeta": (
                ("ocean_time", "eta_rho", "xi_rho"),
                np.broadcast_to(
                    np.asarray(zeta)[:, None, None], (len(zeta), ny, nx)
                ).copy(),
                {"units": "m"},
            ),
            "mask_rho": (("eta_rho", "xi_rho"), np.ones((ny, nx))),
            "mask_u": (("eta_u", "xi_u"), np.ones((ny, nx - 1))),
            "mask_v": (("eta_v", "xi_v"), np.ones((ny - 1, nx))),
            "Cs_r": ("s_rho", [-0.9, -0.5, -0.1]),
            "hc": xr.DataArray(10.0, attrs={"units": "m"}),
            "Vtransform": transform,
        },
        coords={
            "ocean_time": np.datetime64("2020-01-01")
            + np.arange(len(zeta)) * np.timedelta64(3, "h"),
            "s_rho": [-0.9, -0.5, -0.1],
        },
    )


def regrid(roms, dataset, **options):
    kwargs = {
        "bounds": [-126.9, -126.5, 40.1, 40.5],
        "max_depth": 100,
        "depth_levels": [10, 25, 50, 80],
        "grid_spacing": 0.1,
    }
    kwargs.update(options)
    return roms.regrid_roms(dataset, **kwargs)


@pytest.mark.parametrize("transform", [1, 2])
def test_constant_velocities_cf_and_input_preserved(roms, transform):
    source = native_dataset(transform=transform)
    original = source.copy(deep=True)
    result = regrid(roms, source)
    assert result.U.dims == ("time", "depth", "lat", "lon")
    assert set(result.data_vars) == {"U", "V"}
    assert result.sizes == {"time": 1, "depth": 4, "lat": 5, "lon": 5}
    assert result.depth.attrs["units"] == "m"
    assert result.depth.attrs["positive"] == "down"
    assert result.U.attrs["standard_name"] == "eastward_sea_water_velocity"
    assert result.V.attrs["standard_name"] == "northward_sea_water_velocity"
    np.testing.assert_allclose(result.U, 1)
    np.testing.assert_allclose(result.V, 2)
    xr.testing.assert_identical(source, original)


def test_grid_angle_rotates_vectors_to_true_east_and_north(roms):
    result = regrid(roms, native_dataset(angle=np.pi / 2))
    np.testing.assert_allclose(result.U, -2, atol=1e-6)
    np.testing.assert_allclose(result.V, 1, atol=1e-6)


def test_actual_c_grid_faces_are_averaged_at_interior_rho_locations(roms):
    source = native_dataset()
    source["u"][:] = np.arange(6)[None, None, None, :] + 0.5
    source["v"][:] = np.arange(6)[None, None, :, None] + 0.5
    result = regrid(roms, source)
    expected_x = (result.lon.values + 127) / 0.1
    expected_y = (result.lat.values - 40) / 0.1
    np.testing.assert_allclose(
        result.U, np.broadcast_to(expected_x, result.U.shape), atol=1e-6
    )
    np.testing.assert_allclose(
        result.V,
        np.broadcast_to(expected_y[None, None, :, None], result.V.shape),
        atol=1e-6,
    )


@pytest.mark.parametrize("transform", [1, 2])
def test_time_varying_free_surface_uses_physical_depth_and_no_vertical_extrapolation(
    roms, transform
):
    source = native_dataset(transform=transform, zeta=(0, 10))
    # With C(s)=s the two transforms give d=-[zeta+(100+zeta)*s].
    source_depth = -(
        np.asarray([0, 10])[:, None]
        + np.asarray([100, 110])[:, None] * np.asarray([-0.9, -0.5, -0.1])[None, :]
    )
    source["u"][:] = source_depth[:, :, None, None] * 0.01
    source["v"][:] = source_depth[:, :, None, None] * -0.02
    result = regrid(roms, source, depth_levels=[0, 5, 10, 25, 80, 95])
    assert np.isnan(result.U.sel(depth=0)).all()
    assert np.isnan(result.U.sel(depth=5).isel(time=0)).all()
    np.testing.assert_allclose(result.U.sel(depth=5).isel(time=1), 0.05, atol=1e-6)
    assert np.isnan(result.U.sel(depth=95)).all()
    for depth in (10, 25, 80):
        np.testing.assert_allclose(result.U.sel(depth=depth), depth * 0.01, atol=1e-6)
        np.testing.assert_allclose(result.V.sel(depth=depth), depth * -0.02, atol=1e-6)


@pytest.mark.parametrize(
    "transform,expected",
    [
        (1, 4 - 2 * (10 - 1.9) / (27.5 - 1.9)),
        (2, 4 - 2 * (10 - 20 / 11) / (300 / 11 - 20 / 11)),
    ],
)
def test_vtransform_equations_with_nonlinear_supplied_stretching(
    roms, transform, expected
):
    source = native_dataset(transform=transform)
    source = source.assign_coords(s_rho=[-0.8, -0.5, -0.1])
    source["Cs_r"] = ("s_rho", [-0.64, -0.25, -0.01])
    source["u"][:] = np.asarray([1, 2, 4])[None, :, None, None]
    result = regrid(roms, source, depth_levels=[10])
    np.testing.assert_allclose(result.U, expected, atol=1e-6)


def test_curvilinear_horizontal_grid_interpolates_geographic_linear_velocity(roms):
    source = native_dataset()
    shift = np.arange(7)[:, None] * 0.02
    source["lon_rho"][:] += shift
    # A geographic linear field remains linear after face averaging and regridding.
    face_lon = (source.lon_rho.values[:, :-1] + source.lon_rho.values[:, 1:]) / 2
    source["u"][:] = (face_lon[None, None, :, :] + 127) * 10
    source["v"][:] = 0
    result = regrid(roms, source, bounds=[-126.7, -126.5, 40.2, 40.4])
    valid = np.isfinite(result.U.values)
    expected = np.broadcast_to((result.lon.values + 127) * 10, result.U.shape)
    assert valid.any()
    np.testing.assert_allclose(result.U.values[valid], expected[valid], atol=1e-6)


def test_land_hole_and_adjacent_face_masks_remain_missing(roms):
    source = native_dataset()
    source["mask_rho"][3, 3] = 0
    result = regrid(roms, source)
    assert np.isnan(result.U.sel(lon=-126.7, lat=40.3, method="nearest")).all()
    assert np.isnan(result.V.sel(lon=-126.7, lat=40.3, method="nearest")).all()
    np.testing.assert_allclose(result.U.sel(lon=-126.9, lat=40.1, method="nearest"), 1)


def test_explicit_face_mask_blocks_its_contributing_rho_velocities(roms):
    source = native_dataset()
    source["mask_u"][3, 2] = 0
    result = regrid(roms, source)
    assert np.isnan(result.U.sel(lon=-126.7, lat=40.3, method="nearest")).all()
    assert np.isnan(result.V.sel(lon=-126.7, lat=40.3, method="nearest")).all()
    np.testing.assert_allclose(result.U.sel(lon=-126.9, lat=40.1, method="nearest"), 1)


def test_missing_middle_vertical_level_is_not_interpolated_across(roms):
    source = native_dataset()
    source["u"][:, 1] = np.nan
    result = regrid(roms, source, depth_levels=[10, 25, 50, 80])
    np.testing.assert_allclose(result.U.sel(depth=10), 1)
    assert np.isnan(result.U.sel(depth=[25, 50, 80])).all()
    assert np.isnan(result.V.sel(depth=[25, 50, 80])).all()


def test_time_varying_dry_mask_is_preserved(roms):
    source = native_dataset(zeta=(0, 0))
    wet = np.ones((2, 7, 7))
    wet[1, 3, 3] = 0
    source["wetdry_mask_rho"] = (("ocean_time", "eta_rho", "xi_rho"), wet)
    result = regrid(roms, source)
    location = result.U.sel(lon=-126.7, lat=40.3, method="nearest")
    assert np.isfinite(location.isel(time=0)).all()
    assert np.isnan(location.isel(time=1)).all()


def test_time_varying_face_mask_is_preserved(roms):
    source = native_dataset(zeta=(0, 0))
    wet = np.ones((2, 7, 6))
    wet[1, 3, 2] = 0
    source["wetdry_mask_u"] = (("ocean_time", "eta_u", "xi_u"), wet)
    result = regrid(roms, source)
    location = result.U.sel(lon=-126.7, lat=40.3, method="nearest")
    assert np.isfinite(location.isel(time=0)).all()
    assert np.isnan(location.isel(time=1)).all()


def test_native_outer_boundary_is_not_extrapolated(roms):
    result = regrid(roms, native_dataset(), bounds=[-127, -126.4, 40, 40.6])
    assert np.isnan(result.U.isel(lon=0)).all()
    assert np.isnan(result.U.isel(lon=-1)).all()
    assert np.isnan(result.U.isel(lat=0)).all()
    assert np.isnan(result.U.isel(lat=-1)).all()
    assert np.isfinite(result.U.isel(lon=2, lat=2)).all()


def test_default_first_depth_is_positive_and_fixed_grid_can_load_in_cec(roms, tmp_path):
    result = regrid(roms, native_dataset(), depth_levels=None, max_depth=60)
    assert result.depth.values[0] == 5
    assert result.depth.values[-1] == 60
    assert np.isfinite(result.U.sel(depth=10)).all()
    output = tmp_path / "wcofs_physical.nc"
    result.to_netcdf(output, engine="scipy")
    loader = pytest.importorskip("CEC.current_load_year")
    native = loader.ModelCurrent({2020: output}).load_year(2020, as_fieldset=False)
    np.testing.assert_allclose(native.U, result.U, equal_nan=True)
    np.testing.assert_array_equal(native.depth, result.depth)


@pytest.mark.parametrize(
    "kind",
    [
        "missing_transform",
        "missing_stretching",
        "unknown_transform",
        "large_hc_v1",
        "negative_hc",
        "no_zeta",
        "h_in_km",
        "hc_in_km",
        "angle_degrees",
        "sigma_is_depth",
        "nonmonotone_stretching",
        "wrong_face_dimensions",
        "mask_values",
        "all_land",
        "infinite_velocity",
        "duplicate_time",
    ],
)
def test_rejects_missing_or_incompatible_physical_metadata(roms, kind):
    source = native_dataset(zeta=(0, 0))
    if kind == "missing_transform":
        source = source.drop_vars("Vtransform")
    elif kind == "missing_stretching":
        source = source.drop_vars("Cs_r")
    elif kind == "unknown_transform":
        source["Vtransform"] = 3
    elif kind == "large_hc_v1":
        source["Vtransform"] = 1
        source["hc"] = 150
    elif kind == "negative_hc":
        source["hc"] = -1
    elif kind == "no_zeta":
        source = source.drop_vars("zeta")
    elif kind == "h_in_km":
        source.h.attrs["units"] = "km"
    elif kind == "hc_in_km":
        source.hc.attrs["units"] = "km"
    elif kind == "angle_degrees":
        source.angle.attrs["units"] = "degrees"
    elif kind == "sigma_is_depth":
        source = source.assign_coords(s_rho=[10, 50, 90])
    elif kind == "nonmonotone_stretching":
        source["Cs_r"] = ("s_rho", [-0.9, -0.1, -0.5])
    elif kind == "wrong_face_dimensions":
        source["u"] = source.u.isel(xi_u=0, drop=True)
    elif kind == "mask_values":
        source.mask_rho[3, 3] = 2
    elif kind == "all_land":
        source.mask_rho[:] = 0
    elif kind == "infinite_velocity":
        source.u[0, 0, 3, 3] = np.inf
    elif kind == "duplicate_time":
        source = source.assign_coords(ocean_time=[np.datetime64("2020-01-01")] * 2)
    with pytest.raises(ValueError):
        regrid(roms, source)


@pytest.mark.parametrize(
    "options",
    [
        {"grid_spacing": 0},
        {"max_depth": -1},
        {"depth_levels": [20, 10]},
        {"depth_levels": [-1, 10]},
        {"depth_levels": [10, 200]},
        {"bounds": [-126, -127, 40, 41]},
        {"bounds": [0, 1, 0, 1]},
    ],
)
def test_invalid_requested_physical_grid_rejected(roms, options):
    with pytest.raises(ValueError):
        regrid(roms, native_dataset(), **options)
