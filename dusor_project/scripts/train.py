"""
Train the DU-SOR network (Chapter 3, Sec. 3.5.4; Appendix A.5).

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --K 32 --num_layers 10
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import OpticalMIMODataset
from src.detectors.du_sor import DUSORNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--K", type=int, default=None, help="Override system.K")
    p.add_argument("--num_layers", type=int, default=None, help="Override training.num_layers")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--train_samples", type=int, default=None)
    p.add_argument("--val_samples", type=int, default=None)
    p.add_argument("--out", type=str, default="checkpoints/du_sor.pt")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    K = args.K or cfg["system"]["K"]
    num_layers = args.num_layers or cfg["training"]["num_layers"]
    epochs = args.epochs or cfg["training"]["epochs"]
    train_samples = args.train_samples or cfg["training"]["train_samples"]
    val_samples = args.val_samples or cfg["training"]["val_samples"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["learning_rate"]
    patience = cfg["training"]["early_stopping_patience"]
    seed = cfg["training"]["seed"]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    print(f"[train] K={K} num_layers={num_layers} device={device} "
          f"train_samples={train_samples} val_samples={val_samples}")

    train_ds = OpticalMIMODataset(
        K=K, modulation_order=cfg["system"]["modulation"],
        channel_cfg=cfg["channel"], noise_cfg=cfg["noise"],
        snr_db_range=cfg["system"]["snr_db_range"],
        n_samples=train_samples, seed=seed,
    )
    val_ds = OpticalMIMODataset(
        K=K, modulation_order=cfg["system"]["modulation"],
        channel_cfg=cfg["channel"], noise_cfg=cfg["noise"],
        snr_db_range=cfg["system"]["snr_db_range"],
        n_samples=val_samples, seed=seed + 1,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DUSORNet(K=K, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    criterion = nn.MSELoss()

    best_val = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_losses = []
        for x, H, y, sigma2 in train_loader:
            x, H, y, sigma2 = x.to(device), H.to(device), y.to(device), sigma2.to(device)
            optimizer.zero_grad()
            x_hat = model(y, H, sigma2)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, H, y, sigma2 in val_loader:
                x, H, y, sigma2 = x.to(device), H.to(device), y.to(device), sigma2.to(device)
                x_hat = model(y, H, sigma2)
                val_losses.append(criterion(x_hat, x).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"epoch {epoch:3d}/{epochs} | train_loss={train_loss:.6f} "
              f"val_loss={val_loss:.6f} | {elapsed:.1f}s")

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "K": K, "num_layers": num_layers,
                "config": cfg, "history": history,
            }, args.out)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[train] early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    print(f"[train] best val_loss={best_val:.6f}, checkpoint saved to {args.out}")


if __name__ == "__main__":
    main()
