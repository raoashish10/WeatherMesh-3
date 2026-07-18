# Time Log

- 2026-07-18 17:2x — Start. Confirmed repo architecture facts against source; switched GFS
  source from WindBorne API (no access) to NOAA NOMADS; outputs to local disk first, GCS
  later.
- 2026-07-18 17:5x — Env working (natten/matepoint installed), weights + sample data
  downloaded, checkpoint only has a 6-hour processor (no hourly), first multi-lead-time
  rollout on sample data succeeds with no NaN/Inf (~3s for 3 lead times).
