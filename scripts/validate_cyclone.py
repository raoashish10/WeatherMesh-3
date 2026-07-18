"""Retrospective validation: run WM-3 forward from a real historical GFS init time and
check its predicted track for Typhoon Bavi (2026) against real best-track observations.

This is a one-off retrospective run (needs archived GFS from NOAA's AWS Open Data bucket,
since NOMADS' live "prod" directory only retains ~10 days) -- not part of the recurring
6-hourly cron cycle. Saves through the exact same pipeline/storage.py path (local +
S3) as the live production runs, per instruction.
"""
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import torch

import pipeline  # noqa: F401
from pipeline.model_io import load_model
from pipeline.nomads_etl import (
    fetch_gfs_subset, parse_gfs_subset, subsample_to_hres_levels, ARCHIVE_BASE,
)
from pipeline.live_input import build_live_input_tensor
from pipeline.postprocess import to_dataset, save_netcdf
from pipeline.validate import validate_dataset
from pipeline.storage import local_path, cycle_dir, cycle_label, upload_to_s3, s3_enabled
from pipeline.cyclone_track import load_best_track, interpolate_track_position, find_pressure_min
from utils import levels_gfs, levels_hres, to_unix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INIT_DATE, INIT_HOUR = "20260703", "00"  # ~60-66h before Bavi's peak intensity (Jul 5 18z)
LEAD_HOURS = list(range(6, 217, 6))  # +6h .. +216h (9 days), through the Jul 11-12 China landfall
BEST_TRACK_PATH = "docs/bavi_2026_besttrack.json"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    t_start = time.time()
    storm_name, source, track = load_best_track(BEST_TRACK_PATH)
    print(f"Loaded best track for {storm_name} ({len(track)} points) from {source}")

    model = load_model(device=DEVICE)
    print(f"Model loaded in {time.time() - t_start:.1f}s on {DEVICE}")

    init_time = to_unix(datetime.strptime(f"{INIT_DATE}{INIT_HOUR}", "%Y%m%d%H").replace(tzinfo=timezone.utc))

    grib_path = f"nomads_cache/gfs_archive_{INIT_DATE}{INIT_HOUR}_f000.grib2"
    t_fetch = time.time()
    fetch_gfs_subset(INIT_DATE, INIT_HOUR, forecast_hour=0, out_path=grib_path, base=ARCHIVE_BASE)
    print(f"Archived NOMADS/AWS fetch took {time.time() - t_fetch:.1f}s")

    pr_gfs, sfc_core, extras = parse_gfs_subset(grib_path)
    pr_hres = subsample_to_hres_levels(pr_gfs)

    gfs_mesh = model.encoders[0].mesh
    hres_mesh = model.encoders[1].mesh
    era_mesh = model.decoders[0].mesh

    x_gfs = build_live_input_tensor(pr_gfs, sfc_core, extras, levels_gfs, gfs_mesh, DEVICE)
    x_hres = build_live_input_tensor(pr_hres, sfc_core, extras, levels_hres, hres_mesh, DEVICE)
    t0 = torch.tensor([float(init_time)], device=DEVICE)
    x = [x_gfs, x_hres, t0]

    print(f"Running rollout for {len(LEAD_HOURS)} lead hours: {LEAD_HOURS[0]}..{LEAD_HOURS[-1]}")
    t_run = time.time()
    with torch.no_grad():
        outputs = model.forward(x, LEAD_HOURS, send_to_cpu=True)
    print(f"Rollout took {time.time() - t_run:.1f}s")

    predicted, actual, errors = [], [], []
    all_issues = []
    for dt in LEAD_HOURS:
        y = outputs[dt]
        ds = to_dataset(y[0], era_mesh, init_time=init_time, forecast_hour=dt)
        all_issues += validate_dataset(ds, label=f"+{dt}h")

        path = local_path(init_time, dt)
        save_netcdf(ds, path)
        if s3_enabled():
            upload_to_s3(path, remote_key=f"{cycle_label(init_time)}/wm3_f{dt:03d}.nc")

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
        predicted.append((valid_time, pred_lat, pred_lon, pred_pressure))
        actual.append((valid_time, true_lat, true_lon))
        errors.append(err_km)
        print(f"+{dt:>3}h  valid {valid_time:%Y-%m-%d %H:%M}Z  "
              f"pred=({pred_lat:.1f},{pred_lon:.1f},{pred_pressure:.0f}hPa)  "
              f"true=({true_lat:.1f},{true_lon:.1f})  track_error={err_km:.0f}km")

    print(f"\nTrack error: mean={np.mean(errors):.0f}km, median={np.median(errors):.0f}km, "
          f"max={np.max(errors):.0f}km over {len(errors)} matched lead times")
    print(f"Validation issues across rollout: {len(all_issues)}")

    return predicted, actual, errors, init_time, storm_name


if __name__ == "__main__":
    main()
