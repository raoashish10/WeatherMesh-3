# Time Log

- 2026-07-18 17:2x — Start. Confirmed repo architecture facts against source; switched GFS
  source from WindBorne API (no access) to NOAA NOMADS; outputs to local disk first, GCS
  later.
- 2026-07-18 17:5x — Env working (natten/matepoint installed), weights + sample data
  downloaded, checkpoint only has a 6-hour processor (no hourly), first multi-lead-time
  rollout on sample data succeeds with no NaN/Inf (~3s for 3 lead times).
- 2026-07-18 18:0x — Denormalize + NetCDF postprocessing working, values physically
  plausible (2m temp 215-316K, msl 95.6-103.4kPa). Switched to int16-packed NetCDF
  (~170MB/file vs ~650MB float32) since 161 files/cycle wouldn't fit disk otherwise.
- 2026-07-18 18:1x — NOAA NOMADS ETL working end-to-end (range-fetch only needed grib2
  fields, ~130MB/cycle in a few seconds). Fixed a real bug: GFS reports cloud cover as
  0-100%, model expects 0-1 fraction. Hardened fetch against NOMADS rate-limiting
  (redirects to an HTML error page that `requests` follows silently).
- 2026-07-18 18:1x-18:2x — Validation script + 3 key plots done. Live rollout across
  0-360h: only minor near-boundary plausibility flags (e.g. atm temp ~321K vs 320K bound,
  plausible desert extremes), no NaN/Inf/blowups. Hour-0 reconstruction vs actual GFS
  input: mean abs diff 0.74K, visually near-identical — strong end-to-end correctness
  signal.
- 2026-07-18 18:3x — cron installed (dockerd can't run here, no NAT/iptables privilege
  in this nested-container box — documented, not chased further). Full production cycle
  (60 files, +6h..+360h) benchmarked end-to-end: 114.4s GPU inference, 892.0s total
  (dominated by NetCDF postprocess/save, ~13s/file). Fixed a real OOM bug found during
  this run: decoded outputs for all 60 lead times were staying resident on GPU
  simultaneously (~35GB) — fixed via `forward(..., send_to_cpu=True)`.
- 2026-07-18 19:3x — S3 wired up (windbornesystem-mlops-assignment, us-east-2), confirmed
  end-to-end. Switched local/S3 cycle naming from raw unix timestamps to readable
  YYYYMMDD_HHz. Ran one full production cycle (60 files + 2 eye-check plots) for the
  most recent GFS init, verified all 62 artifacts landed in S3 via bucket listing
  (10.24GB total, 968.8s wall-clock).
- 2026-07-18 18:4x — Full-schedule validation: 91 boundary flags, all in just 2 checks
  (atm temp slightly >320K, specific humidity slightly >0.025 kg/kg), stable across all
  60 lead times (not diverging). Geo-clustering: all humidity flags in Arabian
  Peninsula/Persian Gulf (known extreme-dewpoint region); temp flags cluster in
  Sahara/Arabian Peninsula plus Turkmenistan/Uzbekistan (Karakum desert) and Xinjiang's
  Turpan Depression (China's hottest recorded spot) — all at 950-1000hPa (near-surface).
  Real July desert heat, not a model artifact.
