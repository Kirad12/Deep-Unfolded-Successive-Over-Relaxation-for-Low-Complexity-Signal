"""
Ablation study: impact of the number of unfolded layers L on BER/complexity
(Chapter 4, Sec. 4.5.1; Fig. 4.9). Trains a small DU-SOR model for each L in
the sweep and reports BER at a fixed SNR, plus the theoretical FLOP count.

Usage:
    python scripts/run_ablation_layers.py --config configs/default.yaml
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import OpticalMIMODataset
from src.detectors.du_sor import DUSORNet
from src.utils.modulation import PAM
from src.utils.metrics import compute_ber, theoretical_flops


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--layer_values", type=int, nargs="+", default=[2, 5, 10, 15, 20])
    p.add_argument("--eval_snr_db", type=float, default=15.0,
                    help="Fixed SNR (dB) at which BER is reported, per Sec. 4.5.1")
    p.add_argument("--epochs", type=int, default=30, help="Reduced epochs for the ablation sweep")
    p.add_argument("--train_samples", type=int, default=20000)
    p.add_argument("--val_samples", type=int, default=4000)
    p.add_argument("--test_samples", type=int, default=4000)
    p.add_argument("--out", type=str, default="results/ablation_layers.csv")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def train_one(K, num_layers, cfg, args, device):
    train_ds = OpticalMIMODataset(
        K=K, modulation_order=cfg["system"]["modulation"],
        channel_cfg=cfg["channel"], noise_cfg=cfg["noise"],
        snr_db_range=cfg["system"]["snr_db_range"],
        n_samples=args.train_samples, seed=cfg["training"]["seed"],
    )
    val_ds = OpticalMIMODataset(
        K=K, modulation_order=cfg["system"]["modulation"],
        channel_cfg=cfg["channel"], noise_cfg=cfg["noise"],
        snr_db_range=cfg["system"]["snr_db_range"],
        n_samples=args.val_samples, seed=cfg["training"]["seed"] + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    model = DUSORNet(K=K, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    criterion = nn.MSELoss()

    best_state, best_val = None, float("inf")
    for epoch in range(args.epochs):
        model.train()
        for x, H, y, sigma2 in train_loader:
            x, H, y, sigma2 = x.to(device), H.to(device), y.to(device), sigma2.to(device)
            optimizer.zero_grad()
            loss = criterion(model(y, H, sigma2), x)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, H, y, sigma2 in val_loader:
                x, H, y, sigma2 = x.to(device), H.to(device), y.to(device), sigma2.to(device)
                val_losses.append(criterion(model(y, H, sigma2), x).item())
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    K = cfg["system"]["K"]
    pam = PAM(cfg["system"]["modulation"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = OpticalMIMODataset(
        K=K, modulation_order=cfg["system"]["modulation"],
        channel_cfg=cfg["channel"], noise_cfg=cfg["noise"],
        snr_db_range=cfg["system"]["snr_db_range"],
        n_samples=1, seed=999, precompute=False,
    )

    rows = []
    for L in args.layer_values:
        print(f"[ablation] training DU-SOR with L={L} layers ...")
        model = train_one(K, L, cfg, args, device)
        model.eval()

        x, H, y, sigma2 = test_ds.generate_batch(args.test_samples, fixed_snr_db=args.eval_snr_db)
        with torch.no_grad():
            x_t = torch.tensor(y, dtype=torch.float32, device=device)
            H_t = torch.tensor(H, dtype=torch.float32, device=device)
            s2_t = torch.tensor(sigma2, dtype=torch.float32, device=device)
            x_hat = model(x_t, H_t, s2_t).cpu().numpy()

        ber = compute_ber(x_hat, x, pam)
        flops = theoretical_flops(K, "du_sor", num_layers=L)
        print(f"  L={L:2d}  BER={ber:.3e}  FLOPs~{flops:.2e}")
        rows.append({"num_layers": L, "ber": ber, "flops": flops, "eval_snr_db": args.eval_snr_db})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[ablation] results written to {args.out}")


if __name__ == "__main__":
    main()
