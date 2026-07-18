# WeatherMesh-3

This repo defines [WindBorne](https://www.windbornesystems.com)'s WeatherMesh-3
architecture ([ICLR paper](https://arxiv.org/abs/2503.22235)) — architecture only, no
weights/data pipeline. `pipeline/` and `scripts/` add a full inference pipeline: pull real
GFS initial conditions, run WM-3, validate output, run unattended on a GPU box.

## Deviations from a literal WindBorne-API-based pipeline

- **Input source**: WindBorne's API needs an approved account, so ICs come from **NOAA
  NOMADS** instead (public GFS 0.25° grib2, no key needed). Output *format* still follows
  WindBorne's conventions.
- **HRES encoder path**: no ECMWF HRES data used. GFS feeds *both* encoder paths — the
  25-level GFS array is linearly subsampled to the HRES encoder's 20 levels
  (`pipeline/nomads_etl.py: subsample_to_hres_levels`).
- **Rollout schedule**: `weights/WeatherMesh3.pt` only has a **6-hour processor** (checked
  its state_dict) — output is 6-hourly +6h..+360h (60 files/cycle).
- **Storage**: local disk + S3 (GCS also wired up, unused).
- **Docker build not verified** — this box can't run nested Docker, see below.

## Setup & running

### Docker
```
docker build -t weathermesh3-pipeline .
docker run --gpus all \
    -v /path/to/WeatherMesh3.pt:/app/weights/WeatherMesh3.pt \
    -v $(pwd)/outputs:/app/outputs -v $(pwd)/logs:/app/logs \
    -v $(pwd)/nomads_cache:/app/nomads_cache \
    weathermesh3-pipeline
```
`WeatherMesh3.pt` (~1.2GB) isn't baked into the image — mount it in; it's not distributed
in this repo. Default `CMD` runs the full production cycle
(`scripts/run_live_rollout.py`); override with any command from "Running" below, e.g.
`docker run --gpus all ... weathermesh3-pipeline python3 scripts/run_sample_rollout.py 6 12`.

*Not build-verified*: this box is itself an unprivileged container and `dockerd` can't
start on it (`iptables ... Permission denied`). The Dockerfile mirrors the exact steps
verified bare-metal here, but was never actually built/run — test on a plain
(non-nested) Docker host first.

### Running
```
PYTHONPATH=. python3 scripts/run_sample_rollout.py 6 12   # sanity check vs known-good sample data
PYTHONPATH=. python3 scripts/run_live_rollout.py 6 12     # live NOMADS data, a couple of lead hours
PYTHONPATH=. python3 scripts/run_live_rollout.py          # full production cycle: +6h..+360h, 60 files
PYTHONPATH=. python3 scripts/validate_and_plot.py         # rollout + plausibility checks + 3 summary plots
PYTHONPATH=. python3 scripts/validate_cyclone.py           # retrospective: Typhoon Bavi track validation
```

## Pipeline layout

- `pipeline/model_io.py` — loads `ForecastModel` + weights.
- `pipeline/sample_input.py` — input tensors from sample npz data.
- `pipeline/nomads_etl.py` — fetches a GFS cycle (NOMADS or AWS archive for older dates).
- `pipeline/live_input.py` — normalizes + interpolates onto the model's 28-level grid.
- `pipeline/postprocess.py` — denormalizes decoder output, saves int16-packed NetCDF.
- `pipeline/validate.py` — plausibility bounds checks.
- `pipeline/cyclone_track.py` — tracks a storm's MSL pressure minimum for verification.
- `pipeline/plots.py` — summary plots.
- `pipeline/storage.py` — local layout + retention pruning + S3/GCS upload.

## Output format

NetCDF via `xarray`, one file per lead time: `(lat, lon)` surface, `(level, lat, lon)`
upper-air (28 levels). Attrs: `model`, `init_time`, `forecast_hour`, `valid_time`.
Variable names/units follow WindBorne's convention (`temperature_2m` [K], `pressure_msl`
[Pa], `wind_speed_10m` [m/s], `geopotential` [m²/s²], `specific_humidity` [kg/kg]). int16
packed + zlib compressed (~170MB vs ~650MB float32) — needed since 60 files/cycle won't
fit a modest disk otherwise.

## Output storage

Local: `outputs/<YYYYMMDD_HHz>/wm3_f<NNN>.nc`, 2 most recent cycles kept
(`prune_old_cycles`). **S3 active**:
`s3://windbornesystem-mlops-assignment/<YYYYMMDD_HHz>/wm3_f<NNN>.nc` (us-east-2),
confirmed with a real upload verified against the bucket listing; credentials in
`~/.aws/credentials` (outside this repo). GCS also implemented (`GCS_BUCKET` env var) but
unused. Both no-op (logged) when their bucket env var isn't set.

## Automation + monitoring

`scripts/install_cron.sh` installs a crontab entry running `scripts/cron_cycle.sh` every
6h. Each run logs to `logs/cron_<timestamp>.log` (last 20 kept), writes
`logs/last_run_status.json`, and alerts to `logs/alerts.log` (+ optional
`$ALERT_WEBHOOK_URL`) on failure. `find_latest_cycle()` falls back to the previous GFS
cycle if the latest isn't published yet.

## Benchmarked timing (RTX 4090, this box)

Measured from a real completed full-schedule run (60 lead times):

| stage | time |
|---|---|
| Model load | 4.0s |
| NOMADS fetch (133 grib2 fields, parallel) | 1.5s |
| **GPU inference, full 60-lead-time rollout** | **114.4s** (~1.9s/step) |
| NetCDF postprocess+save, all 60 files | ~776s (~13s/file) |
| **Total cycle wall-clock** | **892.0s (~14.9 min)** |

Re-confirmed on two later runs (894.1s, 968.8s). NetCDF postprocessing, not GPU inference,
is the dominant cost — single-threaded, the first thing to optimize.
