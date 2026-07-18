"""Plausibility checks from the assignment brief's validation plan (§7)."""
import numpy as np


def validate_dataset(ds, label=""):
    """Returns a list of human-readable issue strings; empty list = all checks passed."""
    issues = []

    def check(name, ok, detail=""):
        if not ok:
            issues.append(f"[{label}] {name}: {detail}")

    for var in ds.data_vars:
        arr = ds[var].values
        if np.isnan(arr).any():
            issues.append(f"[{label}] {var}: contains NaN")
        if np.isinf(arr).any():
            issues.append(f"[{label}] {var}: contains Inf")

    t2m = ds["temperature_2m"].values
    check("temperature_2m in [180,340]K", 180 <= t2m.min() and t2m.max() <= 340,
          f"{t2m.min():.1f}-{t2m.max():.1f}")

    msl = ds["pressure_msl"].values
    check("pressure_msl in [87000,110000]Pa", 87000 <= msl.min() and msl.max() <= 110000,
          f"{msl.min():.0f}-{msl.max():.0f}")

    for wv in ("wind_speed_10m", "wind_speed_100m"):
        if wv in ds:
            w = ds[wv].values
            check(f"{wv} < 120 m/s", w.max() < 120, f"max {w.max():.1f}")

    dew = ds["dewpoint_2m"].values
    frac_bad = float((dew > t2m + 0.5).mean())  # small tolerance for decoder noise
    check("dewpoint_2m <= temperature_2m (>99% of grid)", frac_bad < 0.01, f"{frac_bad * 100:.2f}% violate")

    tcc = ds["total_cloud_cover"].values
    frac_out = float(((tcc < -0.05) | (tcc > 1.05)).mean())
    check("total_cloud_cover in [0,1] (>95% of grid)", frac_out < 0.05, f"{frac_out * 100:.2f}% out of range")

    temp = ds["temperature"].values
    check("atmospheric temperature in [180,320]K", 180 <= temp.min() and temp.max() <= 320,
          f"{temp.min():.1f}-{temp.max():.1f}")

    z10 = ds["geopotential"].sel(level=10).values
    z1000 = ds["geopotential"].sel(level=1000).values
    frac_correct = float((z10 > z1000).mean())
    check("geopotential(10hPa) > geopotential(1000hPa) (>99% of grid)", frac_correct > 0.99,
          f"{frac_correct * 100:.2f}% correct")

    q = ds["specific_humidity"].values
    check("specific_humidity in [-0.001,0.025] kg/kg", -0.001 <= q.min() and q.max() <= 0.025,
          f"{q.min():.5f}-{q.max():.5f}")

    q_high = ds["specific_humidity"].sel(level=slice(10, 300)).values
    check("specific_humidity near-zero above 300hPa (<0.005)", np.abs(q_high).max() < 0.005,
          f"max {np.abs(q_high).max():.5f}")

    return issues
