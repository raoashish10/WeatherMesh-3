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
