"""End-to-end: pull latest GFS cycle from NOAA NOMADS, preprocess, run WM-3, save NetCDF.
This is the live-data path (replaces WindBorne API access, which wasn't available).
"""
import argparse
import sys
import time

import torch

import pipeline  # noqa: F401
from pipeline.model_io import load_model
from pipeline.nomads_etl import find_latest_cycle, fetch_gfs_subset, parse_gfs_subset, subsample_to_hres_levels
from pipeline.live_input import build_live_input_tensor
from pipeline.postprocess import to_dataset, save_netcdf
from utils import levels_gfs, levels_hres, to_unix
from datetime import datetime, timezone

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_cycle(lead_hours, out_dir="outputs", grib_cache="nomads_cache"):
    t_start = time.time()
    model = load_model(device=DEVICE)
    print(f"Model loaded in {time.time() - t_start:.1f}s on {DEVICE}")

    date_str, hour = find_latest_cycle()
    print(f"Latest GFS cycle: {date_str} {hour}z")
    init_time = to_unix(datetime.strptime(f"{date_str}{hour}", "%Y%m%d%H").replace(tzinfo=timezone.utc))

    grib_path = f"{grib_cache}/gfs_{date_str}{hour}_f000.grib2"
    t_fetch = time.time()
    fetch_gfs_subset(date_str, hour, forecast_hour=0, out_path=grib_path)
    print(f"NOMADS fetch took {time.time() - t_fetch:.1f}s")

    pr_gfs, sfc_core, extras = parse_gfs_subset(grib_path)
    pr_hres = subsample_to_hres_levels(pr_gfs)
    print(f"pr_gfs {pr_gfs.shape}, pr_hres (subsampled) {pr_hres.shape}")

    gfs_mesh = model.encoders[0].mesh
    hres_mesh = model.encoders[1].mesh

    x_gfs = build_live_input_tensor(pr_gfs, sfc_core, extras, levels_gfs, gfs_mesh, DEVICE)
    x_hres = build_live_input_tensor(pr_hres, sfc_core, extras, levels_hres, hres_mesh, DEVICE)
    t0 = torch.tensor([float(init_time)], device=DEVICE)
    x = [x_gfs, x_hres, t0]

    print(f"Running rollout for lead hours: {lead_hours}")
    t_run = time.time()
    with torch.no_grad():
        outputs = model.forward(x, lead_hours)
    print(f"Rollout took {time.time() - t_run:.1f}s")

    era_mesh = model.decoders[0].mesh
    saved = []
    for dt, y in outputs.items():
        if dt == "latent_l2":
            continue
        y0 = y[0]
        has_nan = torch.isnan(y0).any().item()
        has_inf = torch.isinf(y0).any().item()
        print(f"lead +{dt}h: nan={has_nan} inf={has_inf} min/max={y0.min().item():.3f}/{y0.max().item():.3f}")
        ds = to_dataset(y0, era_mesh, init_time=init_time, forecast_hour=dt)
        path = f"{out_dir}/wm3_{init_time}_f{dt:03d}.nc"
        save_netcdf(ds, path)
        saved.append(path)
        print(f"Saved {path}")

    print(f"Total cycle time: {time.time() - t_start:.1f}s")
    return saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lead_hours", nargs="*", type=int, default=[6, 12, 24])
    args = ap.parse_args()
    run_cycle(args.lead_hours or [6, 12, 24])
