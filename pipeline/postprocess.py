"""Denormalize raw decoder output and save it as NetCDF in WindBorne-API-like
variable/unit conventions.

The repo does not ship a denormalize function (see utils.load_normalization docstring)
so we implement the z-score inverse ourselves: actual = normalized * std + mean.
"""
from datetime import datetime, timezone

import numpy as np
import xarray as xr

from utils import load_normalization

# WM-3's own var codes -> WindBorne-style output names + units.
SFC_VAR_MAP = {
    "167_2t": ("temperature_2m", "K"),
    "168_2d": ("dewpoint_2m", "K"),
    "151_msl": ("pressure_msl", "Pa"),
    "165_10u": ("wind_u_10m", "m/s"),
    "166_10v": ("wind_v_10m", "m/s"),
    "246_100u": ("wind_u_100m", "m/s"),
    "247_100v": ("wind_v_100m", "m/s"),
    "45_tcc": ("total_cloud_cover", "0-1"),
    "142_lsp": ("large_scale_precipitation", "m"),
    "143_cp": ("convective_precipitation", "m"),
    "201_mx2t": ("max_temperature_2m", "K"),
    "202_mn2t": ("min_temperature_2m", "K"),
    "15_msnswrf": ("net_solar_radiation", "W/m^2"),
}
# WM-3 outputs precip/min/max on the model's own step interval (6h here), not the
# brief's assumed 3h — named accordingly rather than mislabeled "_3h".
SFC_VAR_MAP.update({
    "142_lsp-6h": ("large_scale_precipitation_6h", "m"),
    "143_cp-6h": ("convective_precipitation_6h", "m"),
    "201_mx2t-6h": ("max_temperature_2m_6h", "K"),
    "202_mn2t-6h": ("min_temperature_2m_6h", "K"),
})

PR_VAR_MAP = {
    "129_z": ("geopotential", "m^2/s^2"),
    "130_t": ("temperature", "K"),
    "131_u": ("wind_u", "m/s"),
    "132_v": ("wind_v", "m/s"),
    "133_q": ("specific_humidity", "kg/kg"),
}


def denormalize(y_normalized, mesh):
    """y_normalized: (1, H, W, n_vars) tensor (or ndarray) on the mesh's own channel layout."""
    y = np.asarray(y_normalized.detach().cpu().float() if hasattr(y_normalized, "detach") else y_normalized)
    y = y[0]  # drop batch dim -> (H, W, n_vars)

    _, std, mean = load_normalization(mesh, with_means=True)
    return y * std + mean


def split_channels(y_actual, mesh):
    """Split the flat (H, W, n_vars) physical array into named pressure/surface arrays."""
    H, W, _ = y_actual.shape
    n_pr = mesh.n_pr
    n_levels = mesh.n_levels

    pr = y_actual[:, :, :n_pr].reshape(H, W, mesh.n_pr_vars, n_levels)
    sfc = y_actual[:, :, n_pr:]

    pr_out = {var: pr[:, :, i, :] for i, var in enumerate(mesh.pressure_vars)}
    sfc_out = {var: sfc[:, :, i] for i, var in enumerate(mesh.sfc_vars) if var != "zeropad"}
    return pr_out, sfc_out


def to_dataset(y_normalized, mesh, init_time, forecast_hour, model_name="WeatherMesh-3"):
    """Build an xarray Dataset matching the brief's WindBorne-style naming/unit convention."""
    y_actual = denormalize(y_normalized, mesh)
    pr_raw, sfc_raw = split_channels(y_actual, mesh)

    lats = mesh.lats
    lons = mesh.lons
    levels = mesh.levels

    data_vars = {}
    for code, arr in pr_raw.items():
        name, unit = PR_VAR_MAP.get(code, (code, "unknown"))
        data_vars[name] = (("level", "lat", "lon"), np.transpose(arr, (2, 0, 1)), {"units": unit})

    for code, arr in sfc_raw.items():
        name, unit = SFC_VAR_MAP.get(code, (code, "unknown"))
        data_vars[name] = (("lat", "lon"), arr, {"units": unit})

    # Derived wind speeds, matching the brief's requested convention.
    if "wind_u_10m" in data_vars and "wind_v_10m" in data_vars:
        u, v = data_vars["wind_u_10m"][1], data_vars["wind_v_10m"][1]
        data_vars["wind_speed_10m"] = (("lat", "lon"), np.sqrt(u ** 2 + v ** 2), {"units": "m/s"})
    if "wind_u_100m" in data_vars and "wind_v_100m" in data_vars:
        u, v = data_vars["wind_u_100m"][1], data_vars["wind_v_100m"][1]
        data_vars["wind_speed_100m"] = (("lat", "lon"), np.sqrt(u ** 2 + v ** 2), {"units": "m/s"})

    init_dt = datetime.fromtimestamp(init_time, tz=timezone.utc)
    valid_dt = datetime.fromtimestamp(init_time + forecast_hour * 3600, tz=timezone.utc)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"lat": lats, "lon": lons, "level": levels},
        attrs={
            "model": model_name,
            "init_time": init_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "forecast_hour": forecast_hour,
            "valid_time": valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": (
                "GFS-derived initial conditions fed into both the GFS-native and "
                "HRES-native encoder paths (no ECMWF HRES data available); precip/min/max "
                "fields are on this checkpoint's native 6-hour step, not the assumed 3h."
            ),
        },
    )
    return ds


def save_netcdf(ds, path):
    """int16 scale/offset packing + zlib. Uncompressed float32 global fields run
    ~650MB/file (plain zlib only gets ~1.6x on smooth fields), which doesn't fit a
    161-file/cycle * multiple-cycles/day budget on a small disk. Packing to int16 with a
    per-variable linear scale is the standard trick for gridded weather data (same idea
    grib2 uses) and gets ~4x with negligible precision loss for these variables."""
    encoding = {}
    for var in ds.data_vars:
        data = ds[var].values
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
        if vmax == vmin:
            vmax = vmin + 1.0
        # Center the offset so packed codes span [-32767, 32767] (int16), not [0, 65534]
        # which would overflow the signed range and wrap for the upper half of values.
        scale_factor = (vmax - vmin) / 65534
        add_offset = (vmax + vmin) / 2
        encoding[var] = {
            "zlib": True,
            "complevel": 4,
            "dtype": "int16",
            "scale_factor": scale_factor,
            "add_offset": add_offset,
            "_FillValue": -32768,
        }
    ds.to_netcdf(path, encoding=encoding)
