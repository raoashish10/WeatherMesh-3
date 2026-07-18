"""Builds the predicted-vs-actual track plot from the NetCDF files
scripts/validate_cyclone.py already saved locally -- no GPU rerun needed."""
import json
from datetime import datetime, timedelta, timezone

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import pipeline  # noqa: F401
from pipeline.cyclone_track import load_best_track, interpolate_track_position, find_pressure_min
from pipeline.storage import local_path, cycle_label
from scripts.validate_cyclone import INIT_DATE, INIT_HOUR, LEAD_HOURS, BEST_TRACK_PATH, haversine_km
from utils import to_unix

init_time = to_unix(datetime.strptime(f"{INIT_DATE}{INIT_HOUR}", "%Y%m%d%H").replace(tzinfo=timezone.utc))


def main():
    storm_name, source, track = load_best_track(BEST_TRACK_PATH)

    predicted, errors = [], []
    for dt in LEAD_HOURS:
        path = local_path(init_time, dt)
        try:
            ds = xr.open_dataset(path)
        except FileNotFoundError:
            continue
        valid_time = datetime.fromtimestamp(init_time, tz=timezone.utc) + timedelta(hours=dt)
        true_pos = interpolate_track_position(track, valid_time)
        if true_pos is None:
            continue
        true_lat, true_lon = true_pos
        found = find_pressure_min(ds, true_lat, true_lon, radius_deg=4.0)
        if found is None:
            continue
        pred_lat, pred_lon, pred_pressure = found
        err_km = haversine_km(true_lat, true_lon, pred_lat, pred_lon)
        predicted.append({"dt": dt, "time": valid_time.isoformat(), "lat": pred_lat, "lon": pred_lon,
                           "pressure_hpa": pred_pressure, "true_lat": true_lat, "true_lon": true_lon,
                           "track_error_km": err_km})
        errors.append(err_km)

    with open("docs/bavi_wm3_vs_besttrack.json", "w") as f:
        json.dump({"storm": storm_name, "init_time": "2026-07-03T00:00:00Z",
                   "mean_track_error_km": float(np.mean(errors)),
                   "median_track_error_km": float(np.median(errors)),
                   "max_track_error_km": float(np.max(errors)),
                   "n_matched": len(errors), "predictions": predicted}, f, indent=2)
    print(f"mean={np.mean(errors):.0f}km median={np.median(errors):.0f}km "
          f"max={np.max(errors):.0f}km n={len(errors)}")

    fig, ax = plt.subplots(figsize=(11, 9), subplot_kw={
        "projection": ccrs.PlateCarree(central_longitude=150)})
    ax.set_extent([110, 175, 5, 40], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.7)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    true_lons = [p["true_lon"] for p in predicted]
    true_lats = [p["true_lat"] for p in predicted]
    pred_lons = [p["lon"] for p in predicted]
    pred_lats = [p["lat"] for p in predicted]

    ax.plot(true_lons, true_lats, "o-", color="black", markersize=4, linewidth=1.5,
            label="Actual best track (CIRA/CSU)", transform=ccrs.PlateCarree(), zorder=3)
    ax.plot(pred_lons, pred_lats, "s-", color="crimson", markersize=4, linewidth=1.5,
            label="WM-3 predicted (init 2026-07-03 00Z)", transform=ccrs.PlateCarree(), zorder=3)
    ax.scatter([true_lons[0]], [true_lats[0]], marker="*", s=250, color="gold",
               edgecolor="black", zorder=4, transform=ccrs.PlateCarree(), label="Forecast init")

    for p in predicted[::4]:  # sparse lead-time labels to avoid clutter
        ax.annotate(f"+{p['dt']}h", (p["lon"], p["lat"]), transform=ccrs.PlateCarree(),
                    fontsize=7, color="crimson", xytext=(3, 3), textcoords="offset points")

    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(
        f"Typhoon Bavi (2026): WM-3 forecast track vs. actual\n"
        f"Init 2026-07-03 00Z, +6h..+{LEAD_HOURS[-1]}h  |  "
        f"mean track error {np.mean(errors):.0f}km, n={len(errors)} lead times"
    )
    out_path = "docs/plots/04_cyclone_bavi_track.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    main()
