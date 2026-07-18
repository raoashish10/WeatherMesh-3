# WeatherMesh-3

This repository defines the architecture of [WindBorne](https://www.windbornesystems.com)'s
WeatherMesh-3 model, as described in the ICLR submission
[WeatherMesh-3](https://arxiv.org/abs/2503.22235). The original repo ships architecture only
(no weights, no data pipeline); everything under `pipeline/` and `scripts/` here is an
inference pipeline built on top of it for a take-home assignment: pull real GFS initial
conditions, run WM-3, validate the output, and keep it running unattended on a GPU box.

## Known deviations from a literal WindBorne-API-based pipeline

- **Input data source**: the WindBorne API requires an approved account with no fast
  self-serve path. Initial conditions are pulled directly from **NOAA NOMADS**
  (`nomads.ncep.noaa.gov`, public GFS 0.25° grib2, no key required) instead. Output
  *format* still follows the WindBorne API's documented variable/unit conventions.
- **HRES encoder path**: WM-3 has two encoders (GFS-native, 25 pressure levels; ECMWF
  HRES-native, 20 levels), but no ECMWF HRES data is used here. Per the assignment's own
  instructions, GFS data is fed into *both* encoder paths — the 25-level GFS array is
  linearly subsampled down to the HRES encoder's 20 levels (`pipeline/nomads_etl.py:
  subsample_to_hres_levels`), interpolating the one level (20 hPa) that isn't already a
  GFS level.
- **Rollout schedule**: the actual checkpoint (`weights/WeatherMesh3.pt`) contains **only
  a 6-hour processor** (verified by inspecting its state_dict keys) — no 1-hour or 24-hour
  processor exists in this weights file. So the output schedule is 6-hourly from +6h to
  +360h (60 files/cycle), not the hourly-then-6-hourly schedule assumed in early planning.
- **Storage**: outputs are written to local disk first; Google Cloud Storage upload is
  wired up (`pipeline/storage.py`) but only activates once `GCS_BUCKET` (+ GCP
  credentials) are set — see "Output storage" below.
- **Docker build not verified** — see the callout in the "Docker" section under Setup
  below for the specific failure and what was and wasn't tested.

## Setup

### Docker

> **Not build-verified.** `docker build` / `docker run --gpus all` below have **not**
> actually been run to completion anywhere in this submission. The GPU box this pipeline
> was developed and run on is itself an unprivileged container, and `dockerd` cannot start
> on it — it fails at network-controller init (`iptables --wait -t nat -N DOCKER: ...
> Permission denied`), confirmed by directly attempting `apt-get install docker.io &&
> dockerd`. The Dockerfile mirrors the exact install steps that *are* verified working
> bare-metal on this box (same pip installs, same NATTEN wheel resolution logic, same
> `PYTHONPATH`), but the image itself has never been built or run. **Test `docker build`
> and a `--gpus all` run on a plain (non-nested) Docker host before relying on this.**

```
docker build -t weathermesh3-pipeline .
docker run --gpus all \
    -v /path/to/WeatherMesh3.pt:/app/weights/WeatherMesh3.pt \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/nomads_cache:/app/nomads_cache \
    weathermesh3-pipeline
```
`WeatherMesh3.pt` (~1.2GB) is not baked into the image — mount it in. It is not
distributed in this repo; obtain it separately (see the assignment brief / WindBorne).

### Bare metal
Requires **exactly PyTorch 2.4.0 + CUDA 12.1** (the NATTEN wheel is built for that exact
tag) — this is the main installation risk, not GPU generation; NATTEN's kernels support
back to Volta/SM70.
```
pip install numpy xarray netCDF4 cartopy matplotlib cfgrib ecmwflibs requests gdown google-cloud-storage
pip install --index-url https://test.pypi.org/simple/ matepoint
pip install --find-links https://shi-labs.com/natten/wheels/ natten==0.17.3+torch240cu121
```
(If `shi-labs.com`'s TLS cert has expired, as it did during this build, download the wheel
directly with `curl -k` and `pip install` the local file — see `Dockerfile` for the exact
command, which resolves the right wheel for whatever Python version is installed.)

Place `WeatherMesh3.pt` at `weights/WeatherMesh3.pt`.

## Running

```
PYTHONPATH=. python3 scripts/run_sample_rollout.py 6 12       # sanity check against the known-good sample data
PYTHONPATH=. python3 scripts/run_live_rollout.py 6 12         # live NOMADS data, a couple of lead hours
PYTHONPATH=. python3 scripts/run_live_rollout.py              # full production cycle: +6h..+360h, 60 files
PYTHONPATH=. python3 scripts/validate_and_plot.py             # rollout + plausibility checks + the 3 summary plots
```

## Pipeline layout

- `pipeline/model_io.py` — loads `ForecastModel` + weights.
- `pipeline/sample_input.py` — builds model input tensors from the provided sample npz
  data (already normalized at native levels; only needs level interpolation).
- `pipeline/nomads_etl.py` — fetches a GFS cycle from NOMADS (range-requests only the
  ~130 grib2 fields WM-3 needs, not the full ~700MB file) and parses it into physical-unit
  arrays.
- `pipeline/live_input.py` — normalizes raw physical NOMADS arrays and interpolates them
  onto the model's 28-level working grid (`utils.interp_levels`).
- `pipeline/postprocess.py` — denormalizes decoder output (the repo doesn't ship a
  denormalize function; this reimplements the z-score inverse) and saves int16-packed
  NetCDF in WindBorne-style variable names/units.
- `pipeline/validate.py` — plausibility bounds checks (temp/pressure/wind ranges,
  dewpoint≤temp, cloud cover, geopotential monotonicity, NaN/Inf).
- `pipeline/plots.py` — the 3 summary plots (2m temp map, MSL pressure + wind, hour-0
  reconstruction vs actual input).
- `pipeline/storage.py` — local output layout + local retention pruning + optional GCS
  upload.

## Output format

NetCDF (`.nc`) via `xarray`, one file per lead time, dims `(lat, lon)` for surface
variables and `(level, lat, lon)` for upper-air variables (28 pressure levels — WM-3's own
working level set, `levels_medium`, a superset of the brief's 25 standard levels). Every
file carries `model`, `init_time`, `forecast_hour`, `valid_time` attrs. Variable names/units
follow the WindBorne API convention (`temperature_2m` [K], `pressure_msl` [Pa],
`wind_u_10m`/`wind_v_10m`/`wind_speed_10m` [m/s], `geopotential` [m²/s², not height],
`specific_humidity` [kg/kg], etc). Files are int16 scale/offset-packed + zlib compressed
(~170MB vs ~650MB float32 uncompressed) — 60 files/cycle otherwise doesn't fit a modest
disk.

## Output storage

Local: `outputs/<init_time_unix>/wm3_f<NNN>.nc`. Only the most recent 2 cycles are kept
locally (`pipeline/storage.py: prune_old_cycles`) since each cycle is ~10GB packed and this
runs unattended — a remote bucket is the durable store, not the instance disk.

**S3 is wired up and active**: `s3://windbornesystem-mlops-assignment/<init_time_unix>/
wm3_f<NNN>.nc` (us-east-2). `scripts/cron_cycle.sh` exports `S3_BUCKET` for every cron
run; AWS credentials live in `~/.aws/credentials` (outside this repo, not
version-controlled). Confirmed working with a real upload verified against the bucket
listing, not just a dry run.

GCS upload is also implemented as an alternative (`GCS_BUCKET` + either
`GOOGLE_APPLICATION_CREDENTIALS` or `gcloud auth application-default login`) but isn't
currently used since S3 covers this submission's storage requirement. Both are no-ops
(logged, not fatal) when their respective bucket env var isn't set.

## Automation + monitoring

`scripts/install_cron.sh` installs a crontab entry running `scripts/cron_cycle.sh` every
6h (offset ~4.5h after 00/06/12/18z so NOMADS has finished publishing). Each run:
- logs to `logs/cron_<timestamp>.log` (last 20 kept),
- writes `logs/last_run_status.json` (timing, file count, validation issue count),
- appends to `logs/alerts.log` and optionally POSTs to `$ALERT_WEBHOOK_URL` on failure.

`find_latest_cycle()` in `pipeline/nomads_etl.py` falls back to the previous cycle
automatically if the latest one isn't published yet, so exact cron timing isn't
correctness-critical.

## Benchmarked timing (RTX 4090, this box)

Measured directly from a real full-schedule run (`scripts/run_live_rollout.py`, no args,
60 lead times +6h..+360h, GFS cycle 2026-07-18 12z):

| stage | time |
|---|---|
| Model load | 4.0s |
| NOMADS fetch (parallel range requests, 133 grib2 fields) | 1.5s |
| **GPU inference, full 60-lead-time rollout** (chained via the 6-hour processor) | **114.4s** (~1.9s/step) |
| NetCDF postprocess+save, all 60 files (denormalize, int16 pack, zlib compress) | ~776s (~13s/file) |
| **Total cycle wall-clock** | **892.0s (~14.9 min)** |

NetCDF postprocessing (not GPU inference) is the dominant cost — comfortably inside the
6h cron window regardless, but the first place to optimize with more time (it's
single-threaded; parallelizing file writes across CPU cores would help most).
