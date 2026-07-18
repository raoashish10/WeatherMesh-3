"""Normalize raw physical-unit NOMADS-derived arrays and interpolate them onto the
model's 28-level working grid -- the live-data equivalent of pipeline/sample_input.py
(which reads already-normalized sample npz files instead).
"""
import json

import numpy as np
import torch

import pipeline  # noqa: F401
from utils import interp_levels, levels_full, core_pressure_vars, core_sfc_vars

EXTRA_SFC_VARS_INPUT = ["45_tcc", "168_2d", "246_100u", "247_100v"]


def load_norms():
    with open("constants/normalization.json") as f:
        return json.load(f)


def normalize_pr(pr_physical, native_levels, norms):
    """pr_physical: (H, W, 5, len(native_levels)) -> z-scored, same shape."""
    which_level = [levels_full.index(lvl) for lvl in native_levels]
    out = np.empty_like(pr_physical, dtype=np.float32)
    for i, var in enumerate(core_pressure_vars):
        mean = np.array(norms[var]["mean"], dtype=np.float32)[which_level]
        std = np.array(norms[var]["std"], dtype=np.float32)[which_level]
        out[:, :, i, :] = (pr_physical[:, :, i, :] - mean) / std
    return out


def normalize_scalar_vars(physical_by_var, norms):
    """physical_by_var: {var_code: (H, W) array} -> stacked+normalized (H, W, len(vars))."""
    return np.stack(
        [(physical_by_var[v] - norms[v]["mean"]) / norms[v]["std"] for v in physical_by_var],
        axis=-1,
    ).astype(np.float32)


def build_live_input_tensor(pr_physical, sfc_core_physical, extras_physical, native_levels, mesh, device="cuda"):
    """
    Args:
        pr_physical: (721, 1440, 5, len(native_levels)) physical units
        sfc_core_physical: (721, 1440, 4) physical units, order = core_sfc_vars
        extras_physical: dict of 4x (721, 1440) physical arrays, keys = EXTRA_SFC_VARS_INPUT
        native_levels: pressure levels of pr_physical's last axis
        mesh: model.encoders[i].mesh
    Returns:
        x (torch.Tensor): (1, 720, 1440, mesh.n_vars), normalized + interpolated to mesh.levels
    """
    norms = load_norms()

    pr_norm = normalize_pr(pr_physical, native_levels, norms)
    sfc_core_by_var = {v: sfc_core_physical[:, :, i] for i, v in enumerate(core_sfc_vars)}
    sfc_norm = normalize_scalar_vars(sfc_core_by_var, norms)
    extras_norm = normalize_scalar_vars({v: extras_physical[v] for v in EXTRA_SFC_VARS_INPUT}, norms)

    # Drop the south-pole row: our arrays are 721 rows, model mesh is 720.
    pr_norm = pr_norm[:720]
    sfc_norm = sfc_norm[:720]
    extras_norm = extras_norm[:720]

    H, W, C_pr, D_native = pr_norm.shape
    pr_flat = pr_norm.reshape(H, W, C_pr * D_native)
    zeropad = np.zeros((H, W, mesh.extra_sfc_pad), dtype=np.float32)

    x_native = np.concatenate([pr_flat, sfc_norm, extras_norm, zeropad], axis=-1)
    x_native = torch.from_numpy(x_native.astype(np.float32)).unsqueeze(0)

    x = interp_levels(x_native, mesh, levels_in=native_levels, levels_out=mesh.levels)
    assert x.shape[-1] == mesh.n_vars, f"{x.shape[-1]} vs mesh.n_vars={mesh.n_vars}"
    return x.to(device)
