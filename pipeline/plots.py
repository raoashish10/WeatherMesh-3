import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_temperature_map(ds, path, title=None):
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    t2m = ds["temperature_2m"]
    im = ax.pcolormesh(ds.lon, ds.lat, t2m, transform=ccrs.PlateCarree(), cmap="RdBu_r", shading="auto")
    ax.coastlines(linewidth=0.5)
    ax.set_title(title or f"WM-3 2m Temperature (K) — init {ds.attrs.get('init_time')} +{ds.attrs.get('forecast_hour')}h")
    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, label="K")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_pressure_wind_map(ds, path, title=None, stride=12):
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    msl = ds["pressure_msl"] / 100.0  # Pa -> hPa
    im = ax.pcolormesh(ds.lon, ds.lat, msl, transform=ccrs.PlateCarree(), cmap="viridis", shading="auto")
    u = ds["wind_u_10m"].values[::stride, ::stride]
    v = ds["wind_v_10m"].values[::stride, ::stride]
    lon2d, lat2d = np.meshgrid(ds.lon.values[::stride], ds.lat.values[::stride])
    ax.quiver(lon2d, lat2d, u, v, transform=ccrs.PlateCarree(), scale=800, width=0.001, color="white", alpha=0.7)
    ax.coastlines(linewidth=0.5)
    ax.set_title(title or f"WM-3 MSL Pressure (hPa) + 10m Wind — init {ds.attrs.get('init_time')} +{ds.attrs.get('forecast_hour')}h")
    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7, label="hPa")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_hour0_vs_input(model_t2m, model_lat, model_lon, gfs_t2m, gfs_lat, gfs_lon, path):
    """Side-by-side WM-3 hour-0 (encode+decode, no processor step) vs the actual raw
    GFS input for the same field/time -- the model's own reconstruction of what it was
    just given, which should look nearly identical to the real input if inference works."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    vmin = min(np.nanmin(model_t2m), np.nanmin(gfs_t2m))
    vmax = max(np.nanmax(model_t2m), np.nanmax(gfs_t2m))

    im0 = axes[0].pcolormesh(model_lon, model_lat, model_t2m, transform=ccrs.PlateCarree(),
                              cmap="RdBu_r", shading="auto", vmin=vmin, vmax=vmax)
    axes[0].coastlines(linewidth=0.5)
    axes[0].set_title("WM-3 hour-0 reconstruction (2m temp, K)")

    im1 = axes[1].pcolormesh(gfs_lon, gfs_lat, gfs_t2m, transform=ccrs.PlateCarree(),
                              cmap="RdBu_r", shading="auto", vmin=vmin, vmax=vmax)
    axes[1].coastlines(linewidth=0.5)
    axes[1].set_title("Actual GFS input (2m temp, K)")

    fig.colorbar(im1, ax=axes, orientation="horizontal", pad=0.08, shrink=0.5, label="K")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
