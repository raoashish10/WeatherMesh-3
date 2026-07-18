"""One full production forecast cycle: pull latest GFS cycle from NOAA NOMADS, preprocess,
run WM-3, validate, save NetCDF (local + optional GCS), prune old local cycles.

This is the live-data path that replaces WindBorne API access (not available for this
assignment) and is the entry point the cron/monitoring wrapper (scripts/cron_cycle.sh)
calls every ~6h.
"""
import argparse
import time
from datetime import datetime, timezone

import torch

import pipeline  # noqa: F401
from pipeline.model_io import load_model
from pipeline.nomads_etl import find_latest_cycle, fetch_gfs_subset, parse_gfs_subset, subsample_to_hres_levels
from pipeline.live_input import build_live_input_tensor
from pipeline.postprocess import to_dataset, save_netcdf
from pipeline.validate import validate_dataset
from pipeline.storage import local_path, prune_old_cycles, upload_to_gcs, gcs_enabled
from utils import levels_gfs, levels_hres, to_unix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Brief's revised (checkpoint-only-has-a-6h-processor) output schedule: 6-hourly out to
# 360h -> 60 files/cycle, instead of the originally assumed hourly+6-hourly 161 files.
DEFAULT_LEAD_HOURS = list(range(6, 361, 6))


def run_cycle(lead_hours=None, grib_cache="nomads_cache", keep_cycles=2):
    lead_hours = lead_hours or DEFAULT_LEAD_HOURS
    t_start = time.time()
    summary = {"ok": False, "issues": [], "saved": [], "uploaded": []}

    model = load_model(device=DEVICE)
    print(f"Model loaded in {time.time() - t_start:.1f}s on {DEVICE}")

    date_str, hour = find_latest_cycle()
    print(f"Latest GFS cycle: {date_str} {hour}z")
    init_time = to_unix(datetime.strptime(f"{date_str}{hour}", "%Y%m%d%H").replace(tzinfo=timezone.utc))
    summary["init_time"] = init_time

    grib_path = f"{grib_cache}/gfs_{date_str}{hour}_f000.grib2"
    t_fetch = time.time()
    fetch_gfs_subset(date_str, hour, forecast_hour=0, out_path=grib_path)
    print(f"NOMADS fetch took {time.time() - t_fetch:.1f}s")

    pr_gfs, sfc_core, extras = parse_gfs_subset(grib_path)
    pr_hres = subsample_to_hres_levels(pr_gfs)
    print(f"pr_gfs {pr_gfs.shape}, pr_hres (subsampled) {pr_hres.shape}")

    gfs_mesh = model.encoders[0].mesh
    hres_mesh = model.encoders[1].mesh
    era_mesh = model.decoders[0].mesh

    x_gfs = build_live_input_tensor(pr_gfs, sfc_core, extras, levels_gfs, gfs_mesh, DEVICE)
    x_hres = build_live_input_tensor(pr_hres, sfc_core, extras, levels_hres, hres_mesh, DEVICE)
    t0 = torch.tensor([float(init_time)], device=DEVICE)
    x = [x_gfs, x_hres, t0]

    print(f"Running rollout for lead hours: {lead_hours}")
    t_run = time.time()
    with torch.no_grad():
        # send_to_cpu=True: with many lead times in one call, decoded outputs would
        # otherwise all stay resident on GPU until the whole rollout finishes (~35GB for
        # the full 60-lead-time schedule) and OOM near the end. Offload each decoded
        # tensor to CPU immediately after its own decode step instead.
        outputs = model.forward(x, lead_hours, send_to_cpu=True)
    print(f"Rollout took {time.time() - t_run:.1f}s")

    for dt, y in outputs.items():
        if dt == "latent_l2":
            continue
        ds = to_dataset(y[0], era_mesh, init_time=init_time, forecast_hour=dt)
        issues = validate_dataset(ds, label=f"+{dt}h")
        summary["issues"] += issues

        path = local_path(init_time, dt)
        save_netcdf(ds, path)
        summary["saved"].append(path)

        if gcs_enabled():
            uri = upload_to_gcs(path, remote_blob_name=f"{init_time}/wm3_f{dt:03d}.nc")
            summary["uploaded"].append(uri)

    removed = prune_old_cycles(keep=keep_cycles)
    if removed:
        print(f"Pruned old local cycles: {removed}")

    summary["ok"] = True
    summary["wall_seconds"] = time.time() - t_start
    print(f"Total cycle time: {summary['wall_seconds']:.1f}s, "
          f"{len(summary['saved'])} files, {len(summary['issues'])} validation issue(s)")
    return summary


def _write_status(status, path="logs/last_run_status.json"):
    import json
    import os as _os

    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(status, f, indent=2, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lead_hours", nargs="*", type=int, default=None)
    ap.add_argument("--keep-cycles", type=int, default=2)
    args = ap.parse_args()

    run_started = datetime.now(timezone.utc).isoformat()
    try:
        summary = run_cycle(args.lead_hours, keep_cycles=args.keep_cycles)
        summary["run_started"] = run_started
        summary["run_finished"] = datetime.now(timezone.utc).isoformat()
        _write_status(summary)
        if summary["issues"]:
            raise SystemExit(f"Completed with {len(summary['issues'])} validation issue(s)")
    except Exception as e:
        _write_status({
            "ok": False, "error": str(e),
            "run_started": run_started, "run_finished": datetime.now(timezone.utc).isoformat(),
        })
        raise
