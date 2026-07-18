"""Full live-data rollout across a representative set of lead times, plausibility
validation on every output, and the 3 "convincing artifact" plots from the brief.
"""
import os
import time

import numpy as np
import torch
import xarray as xr

import pipeline  # noqa: F401
from pipeline.model_io import load_model
from pipeline.nomads_etl import find_latest_cycle, fetch_gfs_subset, parse_gfs_subset, subsample_to_hres_levels
from pipeline.live_input import build_live_input_tensor
from pipeline.postprocess import to_dataset, save_netcdf
from pipeline.validate import validate_dataset
from pipeline.plots import plot_temperature_map, plot_pressure_wind_map, plot_hour0_vs_input
from utils import levels_gfs, levels_hres, to_unix, core_sfc_vars
from datetime import datetime, timezone

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEAD_HOURS = [0, 6, 24, 72, 168, 360]  # hour-0 reconstruction + spread across the full range


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    t_start = time.time()
    model = load_model(device=DEVICE)
    print(f"Model loaded in {time.time() - t_start:.1f}s on {DEVICE}")

    date_str, hour = find_latest_cycle()
    print(f"Latest GFS cycle: {date_str} {hour}z")
    init_time = to_unix(datetime.strptime(f"{date_str}{hour}", "%Y%m%d%H").replace(tzinfo=timezone.utc))

    grib_path = f"nomads_cache/gfs_{date_str}{hour}_f000.grib2"
    fetch_gfs_subset(date_str, hour, forecast_hour=0, out_path=grib_path)
    pr_gfs, sfc_core, extras = parse_gfs_subset(grib_path)
    pr_hres = subsample_to_hres_levels(pr_gfs)

    gfs_mesh = model.encoders[0].mesh
    hres_mesh = model.encoders[1].mesh
    era_mesh = model.decoders[0].mesh

    x_gfs = build_live_input_tensor(pr_gfs, sfc_core, extras, levels_gfs, gfs_mesh, DEVICE)
    x_hres = build_live_input_tensor(pr_hres, sfc_core, extras, levels_hres, hres_mesh, DEVICE)
    t0 = torch.tensor([float(init_time)], device=DEVICE)
    x = [x_gfs, x_hres, t0]

    print(f"Running rollout for lead hours: {LEAD_HOURS}")
    t_run = time.time()
    with torch.no_grad():
        outputs = model.forward(x, LEAD_HOURS)
    print(f"Rollout took {time.time() - t_run:.1f}s")

    all_issues = []
    datasets_by_dt = {}
    for dt, y in outputs.items():
        if dt == "latent_l2":
            continue
        ds = to_dataset(y[0], era_mesh, init_time=init_time, forecast_hour=dt)
        path = f"outputs/wm3_{init_time}_f{dt:03d}.nc"
        save_netcdf(ds, path)
        datasets_by_dt[dt] = ds
        issues = validate_dataset(ds, label=f"+{dt}h")
        all_issues += issues
        status = "PASS" if not issues else f"FAIL ({len(issues)} issues)"
        print(f"+{dt:>3}h  saved {path}  validate: {status}")
        for issue in issues:
            print("   ", issue)

    print()
    print(f"Validation summary: {len(all_issues)} issue(s) across {len(datasets_by_dt)} lead times")

    # --- Plots ---
    ds24 = datasets_by_dt.get(24) or datasets_by_dt[min(dt for dt in datasets_by_dt if dt > 0)]
    plot_temperature_map(ds24, "outputs/plots/01_temperature_map.png")
    plot_pressure_wind_map(ds24, "outputs/plots/02_pressure_wind_map.png")
    print("Saved outputs/plots/01_temperature_map.png, 02_pressure_wind_map.png")

    ds0 = datasets_by_dt[0]
    t2m_idx = core_sfc_vars.index("167_2t")
    gfs_t2m = sfc_core[:720, :, t2m_idx]  # drop south-pole row to match model's 720-row grid
    plot_hour0_vs_input(
        ds0["temperature_2m"].values, ds0.lat.values, ds0.lon.values,
        gfs_t2m, era_mesh.lats, era_mesh.lons,
        "outputs/plots/03_hour0_vs_input.png",
    )
    print("Saved outputs/plots/03_hour0_vs_input.png")

    diff = ds0["temperature_2m"].values - gfs_t2m
    print(f"Hour-0 reconstruction vs actual input: mean abs diff = {np.abs(diff).mean():.3f} K, "
          f"max abs diff = {np.abs(diff).max():.3f} K")

    print(f"\nTotal script time: {time.time() - t_start:.1f}s")
    return all_issues


if __name__ == "__main__":
    issues = main()
    raise SystemExit(1 if issues else 0)
