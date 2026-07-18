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
