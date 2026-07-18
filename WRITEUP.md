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

**3. The most convincing one: WM-3's own hour-0 reconstruction (encode→decode, no
processor step) vs. the actual raw GFS input it was just given.** If the pipeline were
broken anywhere — wrong channel order, bad normalization, wrong grid — this would look
nothing like the input. It doesn't: mean absolute difference **0.7K**, visually
near-identical.

![Hour-0 vs input](docs/plots/03_hour0_vs_input.png)

## (b) Answers

### i. Cloud provider / node, and what changes for 99.9% uptime

The GPU box this pipeline was built and run on was already provisioned when this session
started: an RTX 4090 (47GB reported to the OS), torch 2.4.0+cu121, Ubuntu 22.04 — matching
the profile researched beforehand (vast.ai, a single 24GB+-class consumer GPU; WM-3's
~1.2GB weights and 720×1440 grid don't need multi-GPU or A100s). I didn't personally shop
for and rent an instance in this session since one was already available, but the specs
line up with that plan: CUDA 12.1 to match the pinned NATTEN wheel (`natten==0.17.3
+torch240cu121` — the real installation risk, not GPU generation), on-demand rather than
interruptible given the "run 24h+ unattended" requirement.

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

Note: the architecture research and infra decisions (vast.ai choice, output schedule
modeled on WM-6's API, etc.) referenced throughout were done in a separate research pass
*before* this implementation session — this log covers the hands-on build.

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

**S3**: `s3://windbornesystem-mlops-assignment/` (public bucket, us-east-2), one key per
file: `<YYYYMMDD_HHz>/wm3_f<NNN>.nc`. Wired into the production pipeline
(`pipeline/storage.py`, `scripts/cron_cycle.sh` exports `S3_BUCKET`) and confirmed working
end-to-end — verified with a real upload through `run_live_rollout.py` (not just a
standalone `boto3` call), listed back from the bucket to confirm the objects and sizes
match. GCS upload is also implemented as an alternative path (`GCS_BUCKET` env var) but
not currently used since S3 covers the durable-storage requirement.

Outputs also remain on local disk (`outputs/<YYYYMMDD_HHz>/wm3_f<NNN>.nc`, pruned to the 2
most recent cycles) as a working cache — S3 is the durable store.

## (f) Repo

https://github.com/raoashish10/WeatherMesh-3 — setup instructions, Dockerfile, and a
documented list of deviations from a literal WindBorne-API-based pipeline are in
[`README.md`](README.md).

## (g) With more time

- Parallelize NetCDF postprocessing across CPU cores — it's the dominant cost (~13s/file,
  776s of the 892s full-cycle wall-clock), single-threaded right now.
- Actually build/run the Docker image on a non-nested GPU host (couldn't test it here —
  see README's Docker section).
- Wire up the GCS bucket for real and confirm the upload path end-to-end.
- Get real WindBorne API access and compare its GFS ICs against the NOMADS ones used here.
- Add accuracy-vs-ground-truth verification (compare a past-init forecast against observed
  conditions), which the original validation plan called for but time didn't allow.

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
