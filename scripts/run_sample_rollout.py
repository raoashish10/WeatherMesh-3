"""First end-to-end rollout using the known-good sample npz data.
De-risks model loading + forward-pass mechanics before any ETL code exists.
"""
import sys
import time

import torch

import pipeline  # noqa: F401
from pipeline.model_io import load_model
from pipeline.sample_input import build_input_tensor
from pipeline.postprocess import to_dataset, save_netcdf
from utils import levels_gfs, levels_hres

DATA_DIR = "sample_data/huge/proc/haoxing_data/wm3/data"
TIMESTAMP = 1741305600
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    t_start = time.time()
    model = load_model(device=DEVICE)
    print(f"Model loaded in {time.time() - t_start:.1f}s on {DEVICE}")

    gfs_mesh = model.encoders[0].mesh
    hres_mesh = model.encoders[1].mesh

    x_gfs, t0 = build_input_tensor(DATA_DIR, "neogfs", TIMESTAMP, gfs_mesh, levels_gfs, DEVICE)
    x_hres, _ = build_input_tensor(DATA_DIR, "neohres", TIMESTAMP, hres_mesh, levels_hres, DEVICE)
    print("x_gfs", x_gfs.shape, "x_hres", x_hres.shape, "t0", t0)

    x = [x_gfs, x_hres, t0]

    lead_hours = [int(h) for h in (sys.argv[1:] or ["6", "12"])]
    print(f"Running rollout for lead hours: {lead_hours}")

    # Note: the vendored model_latlon/vars.py get_additional_vars() always returns
    # float32 (its `hours` tensor forces a dtype promotion in torch.cat), which breaks
    # a fully-fp16 forward pass. Running the whole model in fp32 sidesteps that.
    t_run = time.time()
    with torch.no_grad():
        outputs = model.forward(x, lead_hours)
    print(f"Rollout took {time.time() - t_run:.1f}s")

    for dt, y in outputs.items():
        if dt == "latent_l2":
            print("latent_l2:", y.item())
            continue
        y0 = y[0]  # single decoder output
        print(f"lead +{dt}h -> output shape {tuple(y0.shape)}, dtype {y0.dtype}")
        print(f"  nan: {torch.isnan(y0).any().item()}, inf: {torch.isinf(y0).any().item()}")
        print(f"  min/max/mean: {y0.min().item():.4f} / {y0.max().item():.4f} / {y0.mean().item():.4f}")

    torch.save({str(k): v for k, v in outputs.items() if k != "latent_l2"}, "outputs/sample_rollout_raw.pt")
    print("Saved raw (normalized) outputs to outputs/sample_rollout_raw.pt")

    era_mesh = model.decoders[0].mesh
    for dt, y in outputs.items():
        if dt == "latent_l2":
            continue
        ds = to_dataset(y[0], era_mesh, init_time=TIMESTAMP, forecast_hour=dt)
        path = f"outputs/wm3_{TIMESTAMP}_f{dt:03d}.nc"
        save_netcdf(ds, path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
