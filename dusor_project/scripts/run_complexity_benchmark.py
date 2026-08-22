"""
Computational complexity & runtime benchmark (Chapter 4, Sec. 4.4.2 - 4.4.3;
Table 4.1; Fig. 4.8). Measures empirical per-frame runtime for each detector
across a range of MIMO sizes K, and reports the theoretical FLOP estimate
alongside it.

Usage:
    python scripts/run_complexity_benchmark.py --config configs/default.yaml
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
from src.detectors.iterative import sor_detect
from src.utils.metrics import theoretical_flops, benchmark_runtime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--K_values", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    p.add_argument("--num_layers", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1, help="Runtime is per-frame (Sec. 4.2.5)")
    p.add_argument("--n_repeats", type=int, default=30)
    p.add_argument("--out", type=str, default="results/complexity_benchmark.csv")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for K in args.K_values:
        print(f"[complexity] K={K} ...")
        ds = OpticalMIMODataset(
            K=K, modulation_order=cfg["system"]["modulation"],
            channel_cfg=cfg["channel"], noise_cfg=cfg["noise"],
            snr_db_range=cfg["system"]["snr_db_range"],
            n_samples=1, seed=123, precompute=False,
        )
        _, H, y, sigma2 = ds.generate_batch(args.batch_size, fixed_snr_db=15.0)

        # ---- MMSE (O(K^3)) ----
        rt_mmse = benchmark_runtime(lambda y_, H_, s_: mmse_detect(y_, H_, s_), y, H, sigma2,
                                     n_repeats=args.n_repeats, device="cpu")
        rows.append({"K": K, "detector": "mmse", "runtime_ms": rt_mmse,
                      "flops": theoretical_flops(K, "mmse")})

        # ---- ZF (O(K^3)) ----
        rt_zf = benchmark_runtime(lambda y_, H_, s_: zf_detect(y_, H_), y, H, sigma2,
                                   n_repeats=args.n_repeats, device="cpu")
        rows.append({"K": K, "detector": "zf", "runtime_ms": rt_zf,
                      "flops": theoretical_flops(K, "zf")})

        # ---- Classical SOR (O(K^2) per iter, ~50 iters) ----
        sor_omega = cfg["sor"]["omega"]
        rt_sor = benchmark_runtime(
            lambda y_, H_, s_: sor_detect(y_, H_, s_, omega=sor_omega,
                                           max_iterations=cfg["sor"]["max_iterations"],
                                           tol=cfg["sor"]["tolerance"]),
            y, H, sigma2, n_repeats=max(5, args.n_repeats // 5), device="cpu",
        )
        rows.append({"K": K, "detector": "sor", "runtime_ms": rt_sor,
                      "flops": theoretical_flops(K, "sor")})

        # ---- DU-SOR (O(K^2) per layer, fixed L layers) ----
        model = DUSORNet(K=K, num_layers=args.num_layers).to(device)
        model.eval()
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        H_t = torch.tensor(H, dtype=torch.float32, device=device)
        s2_t = torch.tensor(sigma2, dtype=torch.float32, device=device)

        with torch.no_grad():
            rt_dusor = benchmark_runtime(
                lambda y_, H_, s_: model(y_, H_, s_), y_t, H_t, s2_t,
                n_repeats=args.n_repeats, device=device,
            )
        rows.append({"K": K, "detector": "du_sor", "runtime_ms": rt_dusor,
                      "flops": theoretical_flops(K, "du_sor", num_layers=args.num_layers)})

        for r in rows[-4:]:
            print(f"  {r['detector']:8s} runtime={r['runtime_ms']:.4f} ms  flops~{r['flops']:.2e}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[complexity] results written to {args.out}")


if __name__ == "__main__":
    main()
