"""Build model-ready input tensors from the WM-3 sample npz format
(proc/haoxing_data/wm3/data/{neogfs,neohres}/{f000,extra/*}/YYYYMM/<unix>.npz).

The sample data is already z-score normalized at its *native* pressure levels
(25 for GFS, 20 for HRES). The model's encoders expect input on the 28-level
`levels_medium` grid (see meshes.py / model.py), so we interpolate the level
axis with utils.interp_levels() after assembling the raw channel-packed array.
"""
import numpy as np
import torch

import pipeline  # noqa: F401
from utils import interp_levels, to_unix
from datetime import datetime, timezone

EXTRA_SFC_VARS_INPUT = ['45_tcc', '168_2d', '246_100u', '247_100v']


def _load_npz_first(path):
    with np.load(path) as d:
        return d[list(d.keys())[0]]


def build_input_tensor(data_dir, source_prefix, timestamp, mesh, native_levels, device="cuda"):
    """
    Args:
        data_dir: root of the preprocessed data tree, e.g. "sample_data/huge/proc/haoxing_data/wm3/data"
        source_prefix: "neogfs" or "neohres"
        timestamp: unix timestamp (int) matching the npz filename, e.g. 1741305600
        mesh: LatLonGrid for this encoder path (model.encoders[i].mesh)
        native_levels: pressure levels of the raw 'pr' array (levels_gfs or levels_hres)
        device: torch device string

    Returns:
        x (torch.Tensor): shape (1, 720, 1440, mesh.n_vars), interpolated to mesh.levels
        t0 (torch.Tensor): shape (1,), unix timestamp of this input
    """
    yyyymm = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y%m")
    base = f"{data_dir}/{source_prefix}"

    f0 = np.load(f"{base}/f000/{yyyymm}/{timestamp}.npz")
    pr = f0["pr"]          # (721, 1440, 5, n_native_levels)
    sfc = f0["sfc"]        # (721, 1440, 4)

    extras = [
        _load_npz_first(f"{base}/extra/{v}/{yyyymm}/{timestamp}.npz")
        for v in EXTRA_SFC_VARS_INPUT
    ]  # each (721, 1440)

    # Drop the south-pole row: raw storage is 721 rows, model mesh is 720.
    pr = pr[:720]
    sfc = sfc[:720]
    extras = [e[:720] for e in extras]

    H, W, C_pr, D_native = pr.shape
    pr_flat = pr.reshape(H, W, C_pr * D_native)  # var-major, level-minor

    extra_stack = np.stack(extras, axis=-1)  # (720, 1440, 4)
    zeropad = np.zeros((H, W, mesh.extra_sfc_pad), dtype=pr.dtype)

    x_native = np.concatenate([pr_flat, sfc, extra_stack, zeropad], axis=-1)
    x_native = torch.from_numpy(x_native.astype(np.float32)).unsqueeze(0)  # (1,720,1440,C)

    x = interp_levels(x_native, mesh, levels_in=native_levels, levels_out=mesh.levels)
    assert x.shape[-1] == mesh.n_vars, f"{x.shape[-1]} vs mesh.n_vars={mesh.n_vars}"

    t0 = torch.tensor([timestamp], dtype=torch.float32, device=device)
    return x.to(device), t0
