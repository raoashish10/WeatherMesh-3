# WeatherMesh-3

This repo defines [WindBorne](https://www.windbornesystems.com)'s WeatherMesh-3 model
architecture ([ICLR paper](https://arxiv.org/abs/2503.22235)). The original repo ships
architecture only (no weights, no data pipeline) — `pipeline/` and `scripts/` here add a
full inference pipeline on top: pull real GFS initial conditions, run WM-3, validate
output, run unattended on a GPU box.

## Known deviations from a literal WindBorne-API-based pipeline

- **Input source**: WindBorne's API needs an approved account with no fast self-serve
  path, so initial conditions come from **NOAA NOMADS** instead (public GFS 0.25° grib2,
  no key needed). Output *format* still follows WindBorne's documented conventions.
- **HRES encoder path**: no ECMWF HRES data is used. Per the assignment's instructions,
  GFS is fed into *both* encoder paths — the 25-level GFS array is linearly subsampled to
  the HRES encoder's 20 levels (`pipeline/nomads_etl.py: subsample_to_hres_levels`),
  interpolating the one level (20 hPa) not already in the GFS set.
- **Rollout schedule**: `weights/WeatherMesh3.pt` only contains a **6-hour processor**
  (verified from its state_dict keys) — no 1h/24h processor exists. Output is 6-hourly
  +6h..+360h (60 files/cycle), not the hourly-then-6-hourly schedule assumed early on.
- **Storage**: outputs go to local disk + S3 (see "Output storage" below); GCS is also
  wired up as an alternative but unused.
- **Docker build not verified** — see the "Docker" section under Setup.

## Setup

### Docker

> **Not build-verified.** This box is itself an unprivileged container, and `dockerd`
> can't start on it (`iptables --wait -t nat -N DOCKER: ... Permission denied`, confirmed
> by directly attempting `apt-get install docker.io && dockerd`). The Dockerfile mirrors
> the exact steps verified working bare-metal here, but the image itself was never built
> or run. **Test `docker build` / `--gpus all` on a plain (non-nested) host first.**

```
docker build -t weathermesh3-pipeline .
docker run --gpus all \
    -v /path/to/WeatherMesh3.pt:/app/weights/WeatherMesh3.pt \
    -v $(pwd)/outputs:/app/outputs -v $(pwd)/logs:/app/logs \
    -v $(pwd)/nomads_cache:/app/nomads_cache \
    weathermesh3-pipeline
```
`WeatherMesh3.pt` (~1.2GB) isn't baked into the image or distributed in this repo — mount it in.

### Bare metal
Requires **exactly PyTorch 2.4.0 + CUDA 12.1** (the NATTEN wheel is built for that tag) —
the main install risk, not GPU generation (NATTEN supports back to Volta/SM70).
```
pip install numpy xarray netCDF4 cartopy matplotlib cfgrib ecmwflibs requests gdown google-cloud-storage
pip install --index-url https://test.pypi.org/simple/ matepoint
pip install --find-links https://shi-labs.com/natten/wheels/ natten==0.17.3+torch240cu121
```
(If `shi-labs.com`'s TLS cert has expired, `curl -k` the wheel directly and `pip install`
the local file — see `Dockerfile` for the exact command.)

Place `WeatherMesh3.pt` at `weights/WeatherMesh3.pt`.

## Running

```
PYTHONPATH=. python3 scripts/run_sample_rollout.py 6 12       # sanity check vs known-good sample data
PYTHONPATH=. python3 scripts/run_live_rollout.py 6 12         # live NOMADS data, a couple of lead hours
PYTHONPATH=. python3 scripts/run_live_rollout.py              # full production cycle: +6h..+360h, 60 files
PYTHONPATH=. python3 scripts/validate_and_plot.py             # rollout + plausibility checks + 3 summary plots
PYTHONPATH=. python3 scripts/validate_cyclone.py               # retrospective: Typhoon Bavi track validation
```

## Pipeline layout

- `pipeline/model_io.py` — loads `ForecastModel` + weights.
- `pipeline/sample_input.py` — builds input tensors from the sample npz data.
- `pipeline/nomads_etl.py` — fetches a GFS cycle (NOMADS or the AWS archive for older
  dates), range-requesting only the ~130 grib2 fields WM-3 needs.
- `pipeline/live_input.py` — normalizes raw arrays and interpolates onto the model's
  28-level working grid.
- `pipeline/postprocess.py` — denormalizes decoder output (no such function ships in the
  repo) and saves int16-packed NetCDF in WindBorne-style names/units.
- `pipeline/validate.py` — plausibility bounds checks (ranges, dewpoint≤temp, cloud
  cover, geopotential monotonicity, NaN/Inf).
- `pipeline/cyclone_track.py` — tracks a storm's MSL pressure minimum through a rollout
  for ground-truth verification.
- `pipeline/plots.py` — summary plots (temp map, pressure+wind, hour-0 vs input).
- `pipeline/storage.py` — local layout + retention pruning + S3/GCS upload.

## Output format

NetCDF via `xarray`, one file per lead time: `(lat, lon)` for surface vars, `(level, lat,
lon)` for upper-air (28 levels — WM-3's own `levels_medium`, a superset of the brief's 25
standard levels). Attrs: `model`, `init_time`, `forecast_hour`, `valid_time`. Variable
names/units follow WindBorne's convention (`temperature_2m` [K], `pressure_msl` [Pa],
`wind_u_10m`/`wind_speed_10m` [m/s], `geopotential` [m²/s², not height], `specific_humidity`
[kg/kg]). int16 scale/offset-packed + zlib compressed (~170MB vs ~650MB float32
uncompressed) — needed since 60 files/cycle otherwise won't fit a modest disk.

## Output storage

Local: `outputs/<YYYYMMDD_HHz>/wm3_f<NNN>.nc`, only the 2 most recent cycles kept
(`pipeline/storage.py: prune_old_cycles`) — a remote bucket is the durable store, not the
instance disk.

**S3 active**: `s3://windbornesystem-mlops-assignment/<YYYYMMDD_HHz>/wm3_f<NNN>.nc`
(us-east-2). `scripts/cron_cycle.sh` exports `S3_BUCKET`; AWS credentials live in
`~/.aws/credentials` (outside this repo). Confirmed with a real upload verified against
the bucket listing.

GCS is also implemented (`GCS_BUCKET` + `GOOGLE_APPLICATION_CREDENTIALS` or `gcloud auth
application-default login`) but unused since S3 covers this submission. Both are no-ops
(logged, not fatal) when their bucket env var isn't set.

## Automation + monitoring

`scripts/install_cron.sh` installs a crontab entry running `scripts/cron_cycle.sh` every
6h (offset ~4.5h after 00/06/12/18z so NOMADS has finished publishing). Each run logs to
`logs/cron_<timestamp>.log` (last 20 kept), writes `logs/last_run_status.json`, and
appends to `logs/alerts.log` (+ optional `$ALERT_WEBHOOK_URL` POST) on failure.
`find_latest_cycle()` falls back to the previous GFS cycle automatically if the latest
isn't published yet, so exact cron timing isn't correctness-critical.

## Benchmarked timing (RTX 4090, this box)

Measured from a real completed full-schedule run (60 lead times, GFS cycle 2026-07-18 12z):

| stage | time |
|---|---|
| Model load | 4.0s |
| NOMADS fetch (133 grib2 fields, parallel) | 1.5s |
| **GPU inference, full 60-lead-time rollout** | **114.4s** (~1.9s/step) |
| NetCDF postprocess+save, all 60 files | ~776s (~13s/file) |
| **Total cycle wall-clock** | **892.0s (~14.9 min)** |

Re-confirmed on two later runs (894.1s, 968.8s) — consistent. NetCDF postprocessing, not
GPU inference, is the dominant cost and the first thing to optimize (currently
single-threaded).
