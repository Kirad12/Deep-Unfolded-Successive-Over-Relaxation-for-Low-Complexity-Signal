"""
Convenience wrapper that trains a DU-SOR model (if no checkpoint is given)
and immediately runs the BER/BLER-vs-SNR evaluation, reproducing the overall
flow behind Fig. 4.1 - 4.6. Thin orchestration only; see train.py and
evaluate.py for the actual logic.

Usage:
    python scripts/run_ber_vs_snr.py --config configs/default.yaml
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, default="checkpoints/du_sor.pt")
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--out", type=str, default="results/ber_vs_snr.csv")
    return p.parse_args()


def main():
    args = parse_args()
    here = os.path.dirname(__file__)

    if args.force_retrain or not os.path.exists(args.checkpoint):
        print("[run_ber_vs_snr] training DU-SOR network ...")
        subprocess.run(
            [sys.executable, os.path.join(here, "train.py"),
             "--config", args.config, "--out", args.checkpoint],
            check=True,
        )
    else:
        print(f"[run_ber_vs_snr] using existing checkpoint at {args.checkpoint}")

    print("[run_ber_vs_snr] evaluating against baselines ...")
    subprocess.run(
        [sys.executable, os.path.join(here, "evaluate.py"),
         "--checkpoint", args.checkpoint, "--out", args.out],
        check=True,
    )


if __name__ == "__main__":
    main()
