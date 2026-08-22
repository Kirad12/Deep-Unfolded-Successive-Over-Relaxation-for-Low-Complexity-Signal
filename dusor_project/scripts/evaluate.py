"""
Evaluate a trained DU-SOR checkpoint against the classical/deep-learning
baselines across a range of SNRs, reproducing the style of Fig. 4.1
(BER vs SNR) and Fig. 4.4 (BLER vs SNR).

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/du_sor.pt --out results/ber_vs_snr.csv
"""
import argparse
import os
import sys

import numpy as np
import torch
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import OpticalMIMODataset
from src.detectors.du_sor import DUSORNet
from src.detectors.linear import zf_detect, mmse_detect
from src.detectors.iterative import sor_detect, gauss_seidel_detect
from src.detectors.ml_detector import ml_detect
from src.utils.modulation import PAM
from src.utils.metrics import compute_ber, compute_bler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/du_sor.pt")
    p.add_argument("--config", type=str, default=None,
                    help="Override config (defaults to the one stored in the checkpoint)")
    p.add_argument("--snr_points", type=int, default=None)
    p.add_argument("--test_samples", type=int, default=2000,
                    help="Samples per SNR point (kept modest by default; ML is O(M^K))")
    p.add_argument("--detectors", nargs="+",
                    default=["mmse", "zf", "sor", "gs", "du_sor"],
                    help="Add 'ml' only for small K (<= benchmark.ml_max_K)")
    p.add_argument("--out", type=str, default="results/ber_vs_snr.csv")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"] if args.config is None else yaml.safe_load(open(args.config))
    K = ckpt["K"]
    num_layers = ckpt["num_layers"]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DUSORNet(K=K, num_layers=num_layers).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pam = PAM(cfg["system"]["modulation"])
    snr_lo, snr_hi = cfg["system"]["snr_db_range"]
    n_points = args.snr_points or cfg["system"]["snr_db_points"]
    snr_list = np.linspace(snr_lo, snr_hi, n_points)

    eval_ds = OpticalMIMODataset(
        K=K, modulation_order=cfg["system"]["modulation"],
        channel_cfg=cfg["channel"], noise_cfg=cfg["noise"],
        snr_db_range=cfg["system"]["snr_db_range"],
        n_samples=1, seed=cfg["training"]["seed"] + 100, precompute=False,
    )

    sor_omega = cfg["sor"]["omega"]
    sor_iters = cfg["sor"]["max_iterations"]
    sor_tol = cfg["sor"]["tolerance"]
    ml_max_K = cfg["benchmark"]["ml_max_K"]

    rows = []
    for snr_db in snr_list:
        x, H, y, sigma2 = eval_ds.generate_batch(args.test_samples, fixed_snr_db=snr_db)
        print(f"[eval] SNR={snr_db:.1f} dB ...")

        for det in args.detectors:
            if det == "mmse":
                x_hat = mmse_detect(y, H, sigma2)
            elif det == "zf":
                x_hat = zf_detect(y, H)
            elif det == "sor":
                x_hat, _ = sor_detect(y, H, sigma2, omega=sor_omega,
                                       max_iterations=sor_iters, tol=sor_tol)
            elif det == "gs":
                x_hat, _ = gauss_seidel_detect(y, H, sigma2,
                                                max_iterations=sor_iters, tol=sor_tol)
            elif det == "ml":
                if K > ml_max_K:
                    print(f"  skipping ML: K={K} > ml_max_K={ml_max_K}")
                    continue
                x_hat = ml_detect(y, H, sigma2, pam, max_K=ml_max_K)
            elif det == "du_sor":
                with torch.no_grad():
                    x_t = torch.tensor(y, dtype=torch.float32, device=device)
                    H_t = torch.tensor(H, dtype=torch.float32, device=device)
                    s2_t = torch.tensor(sigma2, dtype=torch.float32, device=device)
                    x_hat = model(x_t, H_t, s2_t).cpu().numpy()
            else:
                raise ValueError(f"Unknown detector '{det}'")

            ber = compute_ber(x_hat, x, pam)
            bler = compute_bler(x_hat, x, pam)
            rows.append({"snr_db": snr_db, "detector": det, "ber": ber, "bler": bler})
            print(f"  {det:8s} BER={ber:.3e}  BLER={bler:.3e}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[eval] results written to {args.out}")


if __name__ == "__main__":
    main()
