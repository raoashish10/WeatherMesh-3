"""Track a cyclone's MSL pressure minimum through a WM-3 rollout and compare against
real best-track ground truth -- the "accuracy vs. observed reality" validation check.
"""
import json
from datetime import datetime, timezone

import numpy as np


def load_best_track(path):
    with open(path) as f:
        data = json.load(f)
    track = [
        {**pt, "time": datetime.fromisoformat(pt["time"].replace("Z", "+00:00"))}
        for pt in data["track"]
    ]
    return data["storm"], data["source"], track


def interpolate_track_position(track, valid_time):
    """Linear interpolation between the two best-track points bracketing valid_time.
    Returns None if valid_time is outside the track's observed range."""
    if valid_time < track[0]["time"] or valid_time > track[-1]["time"]:
        return None
    for i in range(len(track) - 1):
        t0, t1 = track[i]["time"], track[i + 1]["time"]
        if t0 <= valid_time <= t1:
            frac = (valid_time - t0).total_seconds() / (t1 - t0).total_seconds()
            lat = track[i]["lat"] + frac * (track[i + 1]["lat"] - track[i]["lat"])
            lon = track[i]["lon"] + frac * (track[i + 1]["lon"] - track[i]["lon"])
            return lat, lon
    return None


def find_pressure_min(ds, center_lat, center_lon, radius_deg=4.0):
    """Finds the MSL pressure minimum within radius_deg of (center_lat, center_lon).
    Uses the best-track position as the search center (standard practice for verifying
    a *known* storm's predicted intensity/position -- this isn't blind feature detection,
    it's "given we know roughly where the storm should be, what did the model predict
    there"), so a wrong prediction still gets found as long as it's within radius_deg.
    """
    msl = ds["pressure_msl"]
    lat_mask = np.abs(ds.lat.values - center_lat) <= radius_deg
    lon_diff = np.abs(((ds.lon.values - center_lon + 180) % 360) - 180)
    lon_mask = lon_diff <= radius_deg

    sub = msl.values[np.ix_(lat_mask, lon_mask)]
    if sub.size == 0:
        return None
    sub_lats = ds.lat.values[lat_mask]
    sub_lons = ds.lon.values[lon_mask]
    idx = np.unravel_index(np.argmin(sub), sub.shape)
    return float(sub_lats[idx[0]]), float(sub_lons[idx[1]]), float(sub[idx] / 100.0)  # Pa -> hPa
