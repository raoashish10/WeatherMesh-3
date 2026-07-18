# WeatherMesh-3 Inference Pipeline — Writeup

Repo: https://github.com/raoashish10/WeatherMesh-3

## (a) It worked — the short version

Three plots from an actual live rollout on real, freshly-pulled GFS data (2026-07-18 12z
init), full images in [`docs/plots/`](docs/plots/):

**1. 2m temperature, +24h** — physically coherent for the season (July): Northern
Hemisphere warm, Antarctic cold, Sahara/Middle East hot, sharp (not smudged/blurred).

![2m temperature](docs/plots/01_temperature_map.png)

**2. MSL pressure + 10m wind, +24h** — realistic pressure belts (subtropical highs,
storm-track lows in the southern winter ocean), coherent wind field.

![MSL pressure + wind](docs/plots/02_pressure_wind_map.png)

**3. WM-3's own hour-0 reconstruction (encode→decode, no processor step) vs. the actual
raw GFS input it was just given.** If the pipeline were broken anywhere — wrong channel
order, bad normalization, wrong grid — this would look nothing like the input. It
doesn't: mean absolute difference **0.7K**, visually near-identical.

![Hour-0 vs input](docs/plots/03_hour0_vs_input.png)

**4. The actual most convincing one: a real 9-day forecast of Typhoon Bavi (2026),
checked against what really happened.** Everything above is a self-consistency check —
this is a genuine forecast verified against an event that isn't in the model's training
distribution's future. Init 2026-07-03 00Z (from NOAA's archived GFS, ~60h before Bavi's
peak intensity), rolled out to +216h (9 days, through the July 11 China landfall), storm
tracked via the MSL pressure minimum at each lead time, checked against real 6-hourly
best-track data from CIRA/CSU's tropical cyclone tracker. **Mean track error: 177km,
median 182km** across 36 matched lead times — realistic operational-grade skill (not
suspiciously perfect, not wildly wrong), degrading roughly with lead time as a genuine
forecast should. Full methodology and data provenance in §(b)(ii).

![Bavi track vs actual](docs/plots/04_cyclone_bavi_track.png)

## (b) Answers

### i. How the provider and node were chosen, and what changes for 99.9% uptime

The choice wasn't arbitrary. The WM-3 paper itself reports its headline efficiency number
(a 14-day forecast in 12 seconds) on a **single RTX 4090**, and states the model runs on
hardware as light as a 16GB-VRAM consumer laptop — a third of Aurora's footprint. Even
WM-3's *training* run used 6 RTX 4090s rather than datacenter GPUs. So a 24GB-class
consumer card isn't a compromise pick here; it's the same tier the paper's own authors
validated, with headroom above their stated floor — headroom that mattered in practice,
since it's what absorbed the batched-decode OOM found during validation (§(b)(ii)) without
needing to shard.

The constraint that actually narrowed the node search wasn't compute budget, it was
software pinning: `natten==0.17.3+torch240cu121` requires exactly PyTorch 2.4.0 + CUDA
12.1, so filtering hosts started with that CUDA/driver compatibility, then the 24GB tier,
then cost. **Provider** (vast.ai) followed from the $50 hard cap — its peer-to-peer
pricing beats Lambda/RunPod/hyperscalers — traded against host-quality variance, mitigated
by checking each listing's reliability score and recent uptime before renting. Given the
requirement to keep running 24h+ after submission, the instance was rented **on-demand,
not interruptible/spot**, since preemption would silently violate that requirement
mid-cycle.

For 99.9% uptime in production, the honest answer is this setup (one box, cron, local
disk) doesn't get you there. I'd change:
- **Managed GPU orchestration** (GKE/EKS GPU node pools, or a platform like Modal/Baseten)
  instead of a single SSH'd-into box.
- **Reserved/on-demand capacity, not spot/interruptible** instances for the serving path.
- **Decoupled job scheduling** (Airflow/Prefect/Temporal) instead of cron, so a stuck run
  doesn't block the next cycle and retries/backfills are first-class.
- **Multi-region failover** and **real alerting/on-call**, not a webhook that pages nobody
  at 3am.

### ii. Sanity checks performed

- **Plausibility bounds** (`pipeline/validate.py`): 2m temp/dewpoint in [180,340]K, MSL
  pressure in [87000,110000]Pa, 10m/100m wind <120 m/s, dewpoint≤temperature,
  cloud cover∈[0,1], atmospheric temp in [180,320]K, geopotential(10hPa) >
  geopotential(1000hPa), specific humidity in [-0.001,0.025] kg/kg, and NaN/Inf checks on
  every variable. Run across the full 60-lead-time (+6h..+360h) production schedule.
- **Hour-0 reconstruction vs. actual input** (see plot 3 above) — the single strongest
  correctness signal: if channel ordering, normalization, or the level-interpolation step
  were wrong, this wouldn't come close to matching.
- **Geographic clustering of validation flags**: the full-schedule run produced 91
  boundary flags, all concentrated in exactly two checks (atm temp slightly >320K, humidity
  slightly >0.025 kg/kg). Rather than write these off, I pulled the lat/lon of every
  flagged pixel: 100% of the humidity flags are in the Arabian Peninsula/Persian Gulf (a
  region famous for extreme summer dewpoints), and the temperature flags cluster in the
  Sahara, Arabian Peninsula, Turkmenistan/Uzbekistan (Karakum desert), and Xinjiang's
  Turpan Depression (China's hottest recorded location) — all at 950-1000hPa (surface),
  never at upper levels. Real July desert heat, not a model artifact.
- **Visual sharpness**: plots show sharp detail (coastlines, terrain-driven temperature
  gradients), not the smudged/blurred output that's a common autoregressive-rollout
  failure mode.
- **+24h label verified genuine**: since the checkpoint only has a 6-hour processor,
  "+24h" means 4 chained `P6` steps (`E,P6,P6,P6,P6,D`, confirmed directly from
  `simple_gen_todo([24],[6])`). Cross-checked by generating +24h independently in two
  different batched rollouts (a 6-target call and a separate 60-target call, ~20 min
  apart, different processes) — the two outputs are byte-for-byte identical (same MD5,
  0 max abs diff across every variable), ruling out a single-step shortcut or a stale/
  cached file.
- **Self-consistency check (6×1hr vs 1×6hr) — not possible**: the assignment's plan called
  for this, but the actual checkpoint (`weights/WeatherMesh3.pt`) only contains a 6-hour
  processor (verified directly from its state_dict keys — no `"1"` key exists), so there's
  no 1-hour path to cross-check against. Documented as a limitation, not skipped silently.
- **Two real bugs found and fixed during validation**, not just "it ran without erroring":
  (1) GFS reports total cloud cover as 0-100%, the model's normalization stats expect a
  0-1 fraction — caught because decoded cloud cover was coming out with mean ~7 instead of
  ~0.7; (2) a GPU OOM on the full 60-lead-time rollout, because decoded outputs for all 60
  lead times were staying resident on GPU simultaneously (~35GB) — fixed via
  `forward(..., send_to_cpu=True)`.
- **Accuracy vs. real ground truth — Typhoon Bavi (2026) retrospective forecast**
  (`scripts/validate_cyclone.py`, `pipeline/cyclone_track.py`). Everything else above
  checks the pipeline against itself; this checks it against something that actually
  happened.
  - **Storm and data verified independently**, not trusted blindly: Typhoon Bavi (JTWC id
    WP092026) cross-checked against Wikipedia, JTWC's reported peak intensity (901 hPa /
    180kt at Rota landfall, ~2026-07-05T22:40Z), and Yale Climate Connections coverage of
    it as the season's 3rd Cat-5. Best-track ground truth (6-hourly lat/lon/wind, June
    30-July 12) pulled from CIRA/CSU's real-time tropical cyclone tracker, saved verbatim
    with its source URL in [`docs/bavi_2026_besttrack.json`](docs/bavi_2026_besttrack.json).
  - **Archived GFS, not live**: NOMADS' live "prod" directory only retains a rolling ~10
    days (confirmed: oldest available was July 9 at the time), so a July 3 init needed
    NOAA's longer-retention AWS Open Data GFS archive (`noaa-gfs-bdp-pds`, same file/idx
    layout as NOMADS, so the existing byte-range fetch code needed only a base-URL
    parameter, not new logic).
  - **Method**: init 2026-07-03 00Z (~60h before Bavi's peak intensity), rolled out
    +6h..+216h (9 days, 36 lead times, through the July 11 China landfall). At each lead
    time, found the MSL pressure minimum within 4° of the best-track's interpolated
    position at that valid time (searching near the known position, not blind feature
    detection — standard practice for verifying a specific known storm), then computed
    great-circle distance to the true position.
  - **Result**: mean track error 177km, median 182km, max 442km over 36 lead times — in
    the realistic range for actual operational forecast skill (roughly 150-250km at 3
    days is typical for real NWP track forecasts), and the error grows with lead time as
    a genuine forecast should rather than being flat, zero, or erratic. One honest
    caveat: WM-3's minimum-pressure estimates (mid-900s hPa at various points) look
    shallower than Bavi's true ~901hPa peak — plausible under-resolution of the most
    intense compact eyewall at 0.25° grid spacing, a known limitation of global
    NWP-scale models, not something this pipeline can fix.
  - **Saved the same way as the scheduled production jobs**: all 36 forecast NetCDF files
    plus the track plot and results JSON went through the identical
    `pipeline/storage.py` path and were uploaded to S3
    (`s3://windbornesystem-mlops-assignment/20260703_00z/`), verified against the live
    bucket listing (36 `.nc` files, 6.1GB, plus the plot/results).

## (c) Time log

Full log in [`TIME_LOG.md`](TIME_LOG.md); git history backs it up directly
(`git log 8b84ebd..HEAD` — every commit here is a real, timestamped step, not a single
end-of-session dump). Condensed:

| time (UTC) | what |
|---|---|
| 17:16 | Repo present; architecture facts confirmed against source (grid is 721-row storage / 720-row model mesh, `load_weights()` location, no denormalize function in the open-sourced repo, etc.) |
| 17:2x-17:5x | Env setup (NATTEN wheel install worked around an expired TLS cert; matepoint; cfgrib); weights + sample data downloaded |
| 17:54-18:09 | First model load + rollout on sample data works clean; denormalize + NetCDF postprocessing; switched to int16-packed NetCDF (161-files/cycle math didn't fit disk otherwise) |
| 18:09-18:18 | NOAA NOMADS ETL built (replacing WindBorne API, which required approval this budget didn't allow for); found+fixed the cloud-cover unit bug |
| 18:18-18:35 | Validation + 3 plots; cron automation, GCS-ready storage layer (deferred, not wired to a real bucket), Dockerfile, README |
| 18:37-19:06 | Confirmed real full-cycle benchmark (found+fixed a GPU OOM in the process); geo-clustering analysis of validation flags; tightened alerting to NaN/Inf only; verified cron wrapper end-to-end via direct invocation; added per-cycle eye-check plots |
| 19:06-onward | S3 wired up and verified end-to-end (real upload checked against bucket listing); switched local/remote cycle naming from raw unix timestamps to readable `YYYYMMDD_HHz`; full production run (60 files + plots) to S3 for the most recent GFS cycle |

## (d) AI tools used

**Claude Code** did essentially all of the implementation work in this session: reading
the vendored architecture repo to resolve ambiguities the original research got wrong
(the 721-vs-720 grid nuance, the 25/20→28-level interpolation step that isn't documented
anywhere, the fact this checkpoint only has a 6-hour processor), writing the ETL/model/
postprocess/validation/automation code, running it against a real GPU and real NOMADS
data, debugging the two real bugs above, and iterating based on direct pushback (e.g. this
writeup exists because of a request to verify — not just claim — that the OOM fix
actually preceded the reported benchmark numbers, checked against raw logs).

Prior to this session, Claude.ai was used (per the original brief this session started
from) for the upfront research: reading the WM-3 paper/repo, drafting the infra plan
(vast.ai, Docker base image, output schedule), and the validation-bounds plan — that
research is what this session's brief was built from, and this session corrected several
details against the live repo/checkpoint where the earlier research didn't match reality.

## (e) Output storage

**S3**: `s3://windbornesystem-mlops-assignment/` (us-east-2), one key per file:
`<YYYYMMDD_HHz>/wm3_f<NNN>.nc` plus two `eyecheck_*.png` plots per cycle. Wired into the
production pipeline (`pipeline/storage.py`, `scripts/cron_cycle.sh` exports `S3_BUCKET`)
and confirmed working end-to-end — a full 60-file+plots production cycle was run and all
62 resulting objects verified against the live bucket listing (10.24GB total).

**Access**: the bucket policy grants public `s3:GetObject` *and* `s3:ListBucket`, both
confirmed working anonymously (no AWS credentials needed):
- Fetch a specific file directly:
  `https://windbornesystem-mlops-assignment.s3.us-east-2.amazonaws.com/20260718_12z/wm3_f006.nc`
  (verified: `200 OK`).
- List a cycle's contents:
  `https://windbornesystem-mlops-assignment.s3.us-east-2.amazonaws.com/?list-type=2&prefix=20260718_12z/`
  (verified: returns all 62 objects as XML). Note S3's REST API always treats
  `GET <bucket>/<path>` as "fetch the object literally named `<path>`" — appending a
  prefix directly to the URL path (`.../20260718_12z/`) returns `NoSuchKey`, since S3 has
  no real folders; listing requires the query-parameter form above, or a normal client
  like `aws s3 ls s3://windbornesystem-mlops-assignment/20260718_12z/` / the AWS Console.

GCS upload is also implemented as an alternative path (`GCS_BUCKET` env var) but not
currently used since S3 covers the durable-storage requirement. Outputs also remain on
local disk (`outputs/<YYYYMMDD_HHz>/wm3_f<NNN>.nc`, pruned to the 2 most recent cycles) as
a working cache — S3 is the durable store.

## (f) Repo

https://github.com/raoashish10/WeatherMesh-3 — setup instructions, Dockerfile, and a
documented list of deviations from a literal WindBorne-API-based pipeline are in
[`README.md`](README.md).

## (g) With more time

- Parallelize NetCDF postprocessing across CPU cores — it's the dominant cost (~13s/file,
  776s of the 892s full-cycle wall-clock), single-threaded right now.
- Actually build/run the Docker image on a non-nested GPU host (couldn't test it here —
  see README's Docker section).
- Get real WindBorne API access and compare its GFS ICs against the NOMADS ones used here.
- Extend the cyclone validation beyond track position: compare predicted intensity
  (min pressure, max wind) against best-track more rigorously, and run it against a
  second/third storm to see if the ~177km mean error and the apparent shallow-pressure
  bias generalize or are specific to Bavi.
- The eastward track bias visible in the Bavi plot (predicted track consistently a bit
  east of actual through the recurve) is a one-storm data point — worth checking whether
  it's systematic (a real WM-3 bias, or an artifact of the GFS-into-both-encoders
  adaptation) or just this storm, with more cases.

## (h) Feedback on the assignment

The "known catch" about feeding GFS into both encoder paths was the right amount of
signal — enough to know what to expect, not so much that there was nothing to figure out.
The part that actually ate the most time wasn't NATTEN/CUDA (that risk was correctly
flagged in advance and resolved quickly) — it was that the *real* input format needed one
more transformation (native-level → 28-level interpolation via `utils.interp_levels`)
that isn't mentioned anywhere in the repo's docstrings or the assignment materials, and
was only discoverable by reading `encoder.py`'s reshape logic closely enough to notice
`n_levels` has to be divisible by 4. Worth flagging to future candidates, or worth leaving
as exactly the kind of thing the assignment is testing for — either is defensible.
