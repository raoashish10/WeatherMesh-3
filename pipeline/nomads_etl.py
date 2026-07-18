"""Pull a GFS 0.25deg initial-condition cycle from NOAA NOMADS (no API key needed) and
build WM-3-format input arrays, replacing the (inaccessible) WindBorne API for ICs.

Strategy: NOMADS serves per-field byte ranges via a companion .idx file for each grib2
file, so instead of downloading the full ~700MB pgrb2.0p25 file we range-fetch only the
~130 fields WM-3 actually needs and concatenate them into one small multi-message grib2
file (concatenated GRIB2 messages are independently self-delimited, so this is valid).
"""
import concurrent.futures
import io
import random
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
import xarray as xr

import pipeline  # noqa: F401
from utils import levels_gfs, levels_hres, core_pressure_vars

BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
# NOMADS' own "prod" directory only keeps a rolling ~10 day window (confirmed empirically:
# oldest available was 9 days back at time of writing). NOAA's public AWS Open Data GFS
# archive mirrors the exact same file/idx layout with much longer retention -- used for
# retrospective validation against real past events (e.g. cyclone track verification).
ARCHIVE_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
PRESSURE_GRIB_VARS = {"129_z": "HGT", "130_t": "TMP", "131_u": "UGRD", "132_v": "VGRD", "133_q": "SPFH"}
G0 = 9.80665  # m/s^2, converts geopotential height (gpm) -> geopotential (m^2/s^2)

SURFACE_FIELDS = [
    # (grib var, level string, our var code)
    ("TMP", "2 m above ground", "167_2t"),
    ("DPT", "2 m above ground", "168_2d"),
    ("UGRD", "10 m above ground", "165_10u"),
    ("VGRD", "10 m above ground", "166_10v"),
    ("UGRD", "100 m above ground", "246_100u"),
    ("VGRD", "100 m above ground", "247_100v"),
    ("PRMSL", "mean sea level", "151_msl"),
    ("TCDC", "entire atmosphere", "45_tcc"),
]


def _http1_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "weathermesh3-pipeline/1.0"})
    return s


def _looks_like_idx(resp):
    return resp.status_code == 200 and resp.text.lstrip().startswith("1:")


def _looks_like_grib_bytes(resp):
    # NOMADS silently 302-redirects rate-limited requests to an HTML "Over Rate" page;
    # `requests` follows redirects by default so this can come back as a 200 with HTML
    # content instead of grib bytes. Reject anything that looks like a text/html body.
    if resp.status_code not in (200, 206):
        return False
    head = resp.content[:20].lstrip().lower()
    return not (head.startswith(b"<!doctype") or head.startswith(b"<html"))


def _get_with_retry(session, url, validate, headers=None, timeout=30, retries=8):
    last_status = None
    for attempt in range(retries):
        try:
            r = session.get(url, headers=headers, timeout=timeout)
            last_status = r.status_code
            if validate(r):
                return r
        except requests.RequestException as e:
            last_status = str(e)
        time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts (last: {last_status})")


def find_latest_cycle(session=None, max_days_back=2):
    """Returns (date_str YYYYMMDD, hour_str HH) for the most recent GFS cycle whose
    f000 pgrb2.0p25 file is available on NOMADS."""
    session = session or _http1_session()
    now = datetime.now(timezone.utc)
    for days_back in range(max_days_back + 1):
        day = now - timedelta(days=days_back)
        date_str = day.strftime("%Y%m%d")
        for hour in ["18", "12", "06", "00"]:
            url = f"{BASE}/gfs.{date_str}/{hour}/atmos/gfs.t{hour}z.pgrb2.0p25.f000.idx"
            try:
                r = _get_with_retry(session, url, _looks_like_idx, timeout=15, retries=3)
                if r:
                    return date_str, hour
            except RuntimeError:
                continue  # this cycle isn't published yet (or truly unavailable); try older
    raise RuntimeError("No recent GFS cycle found on NOMADS")


def _parse_idx(idx_text):
    records = []
    for line in idx_text.strip().splitlines():
        parts = line.split(":")
        records.append({"seq": int(parts[0]), "start": int(parts[1]), "var": parts[3], "level": parts[4]})
    return records


def _wanted_records(records):
    wanted = []
    for r in records:
        for code, gribvar in PRESSURE_GRIB_VARS.items():
            for lvl in levels_gfs:
                if r["var"] == gribvar and r["level"] == f"{lvl} mb":
                    wanted.append((r, f"{code}_{lvl}"))
        for gribvar, levelstr, code in SURFACE_FIELDS:
            if r["var"] == gribvar and r["level"] == levelstr:
                wanted.append((r, code))
    return wanted


def fetch_gfs_subset(date_str, hour, forecast_hour=0, out_path=None, max_workers=16, base=BASE):
    """Downloads only the grib2 messages WM-3 needs for one GFS init time.
    Returns the raw concatenated grib2 bytes (also written to out_path if given).
    Pass base=ARCHIVE_BASE for init times older than NOMADS' rolling retention window."""
    session = _http1_session()
    url = f"{base}/gfs.{date_str}/{hour}/atmos/gfs.t{hour}z.pgrb2.0p25.f{forecast_hour:03d}"

    idx = _get_with_retry(session, url + ".idx", _looks_like_idx, timeout=30)
    records = _parse_idx(idx.text)
    records_sorted = sorted(records, key=lambda r: r["start"])
    starts = {r["seq"]: i for i, r in enumerate(records_sorted)}

    wanted = _wanted_records(records)
    assert len(wanted) == len(levels_gfs) * len(PRESSURE_GRIB_VARS) + len(SURFACE_FIELDS), (
        f"expected {len(levels_gfs) * len(PRESSURE_GRIB_VARS) + len(SURFACE_FIELDS)} fields, got {len(wanted)}"
    )

    def fetch_one(rec_code):
        rec, code = rec_code
        i = starts[rec["seq"]]
        start = records_sorted[i]["start"]
        end = records_sorted[i + 1]["start"] - 1 if i + 1 < len(records_sorted) else ""
        resp = _get_with_retry(
            session, url, _looks_like_grib_bytes,
            headers={"Range": f"bytes={start}-{end}"}, timeout=60,
        )
        return code, resp.content

    t0 = time.time()
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for code, content in ex.map(fetch_one, wanted):
            results[code] = content
    print(f"Fetched {len(results)} GFS fields in {time.time() - t0:.1f}s")

    buf = io.BytesIO()
    for _, content in results.items():
        buf.write(content)
    data = buf.getvalue()
    if out_path:
        with open(out_path, "wb") as f:
            f.write(data)
    return data


def parse_gfs_subset(path):
    """Parses the concatenated grib2 subset into physical-unit arrays matching WM-3's
    layout, at GFS's *native* 25 pressure levels (not yet interpolated to the model's
    28-level working grid -- see pipeline/live_input.py for that step).

    Returns:
        pr (np.ndarray): (721, 1440, 5, 25) float32, var order = core_pressure_vars,
            level order = levels_gfs (ascending, ends at 1000 hPa)
        sfc_core (np.ndarray): (721, 1440, 4) float32, order = core_sfc_vars
        extras (dict[str, np.ndarray]): each (721, 1440) float32, keyed by var code
    """
    import cfgrib

    datasets = cfgrib.open_datasets(path)
    by_vars = {frozenset(ds.data_vars): ds for ds in datasets}

    def find(*names):
        for varset, ds in by_vars.items():
            if set(names) <= varset:
                return ds
        raise KeyError(f"No dataset with vars {names} among {[set(v) for v in by_vars]}")

    pr_ds = find("t", "u", "v", "q", "gh").sortby("isobaricInhPa")  # ascending, matches levels_gfs
    assert list(pr_ds.isobaricInhPa.values.astype(int)) == levels_gfs

    assert list(core_pressure_vars) == ["129_z", "130_t", "131_u", "132_v", "133_q"]
    per_var = [  # each (level=25, lat=721, lon=1440)
        pr_ds["gh"].values * G0,
        pr_ds["t"].values,
        pr_ds["u"].values,
        pr_ds["v"].values,
        pr_ds["q"].values,
    ]
    pr = np.stack(per_var, axis=0).astype(np.float32)  # (5, 25, 721, 1440)
    pr = np.transpose(pr, (2, 3, 0, 1))  # (721, 1440, 5, 25)

    sfc10 = find("u10", "v10")
    sfc2 = find("t2m", "d2m")
    sfcmsl = find("prmsl")
    sfc_core = np.stack([
        sfc10["u10"].values, sfc10["v10"].values, sfc2["t2m"].values, sfcmsl["prmsl"].values,
    ], axis=-1).astype(np.float32)

    # GFS reports total cloud cover as a percent (0-100); WM-3's normalization stats
    # (and the sample data) use the ERA5-style 0-1 fraction convention.
    tcc = find("tcc")["tcc"].values.astype(np.float32) / 100.0
    d2m = sfc2["d2m"].values.astype(np.float32)
    u100 = find("u100", "v100")["u100"].values.astype(np.float32)
    v100 = find("u100", "v100")["v100"].values.astype(np.float32)
    extras = {"45_tcc": tcc, "168_2d": d2m, "246_100u": u100, "247_100v": v100}

    return pr, sfc_core, extras


def subsample_to_hres_levels(pr_gfs):
    """Linearly interpolates the 25-level GFS-native pressure array down to the 20
    HRES levels. levels_hres is a subset of levels_gfs except for 20 hPa, which needs
    real interpolation between the 10 and 30 hPa GFS levels -- this is the
    "GFS-into-both-encoder-paths" adaptation the assignment calls out explicitly.
    utils.interp_levels() refuses this direction (it asserts no genuine interpolation
    when going to *fewer* levels), so this is a plain linear interpolant instead.
    """
    out = np.zeros(pr_gfs.shape[:-1] + (len(levels_hres),), dtype=pr_gfs.dtype)
    for i, lvl in enumerate(levels_hres):
        if lvl in levels_gfs:
            out[..., i] = pr_gfs[..., levels_gfs.index(lvl)]
        else:
            j = np.searchsorted(levels_gfs, lvl)
            lo, hi = levels_gfs[j - 1], levels_gfs[j]
            w = (lvl - lo) / (hi - lo)
            out[..., i] = pr_gfs[..., j - 1] * (1 - w) + pr_gfs[..., j] * w
    return out
