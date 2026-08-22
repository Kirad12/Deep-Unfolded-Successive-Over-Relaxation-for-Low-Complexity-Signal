"""
Plot helper: turns the CSV outputs of evaluate.py / run_ablation_layers.py /
run_complexity_benchmark.py into the figure styles used in Chapter 4
(BER vs SNR, BLER vs SNR, BER vs num_layers, runtime vs K).

Usage:
    python scripts/plot_results.py --ber_csv results/ber_vs_snr.csv --out results/ber_vs_snr.png
    python scripts/plot_results.py --ablation_csv results/ablation_layers.csv --out results/ablation.png
    python scripts/plot_results.py --complexity_csv results/complexity_benchmark.csv --out results/runtime.png
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_ber_snr(csv_path: str, out_path: str, metric: str = "ber"):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for det, sub in df.groupby("detector"):
        sub = sub.sort_values("snr_db")
        ax.semilogy(sub["snr_db"], sub[metric], marker="o", label=det.upper())
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs SNR")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def plot_ablation(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path).sort_values("num_layers")
    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    ax1.semilogy(df["num_layers"], df["ber"], marker="o", color="tab:blue", label="BER")
    ax1.set_xlabel("Number of unfolded layers (L)")
    ax1.set_ylabel("BER", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(df["num_layers"], df["flops"], marker="s", color="tab:red", label="FLOPs")
    ax2.set_ylabel("Theoretical FLOPs", color="tab:red")
    fig.suptitle("Impact of unfolded layers L on BER and complexity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def plot_complexity(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for det, sub in df.groupby("detector"):
        sub = sub.sort_values("K")
        ax.plot(sub["K"], sub["runtime_ms"], marker="o", label=det.upper())
    ax.set_xlabel("Number of antennas (K)")
    ax.set_ylabel("Runtime per frame (ms)")
    ax.set_yscale("log")
    ax.set_title("Runtime vs MIMO size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ber_csv", type=str, default=None)
    p.add_argument("--ablation_csv", type=str, default=None)
    p.add_argument("--complexity_csv", type=str, default=None)
    p.add_argument("--metric", type=str, default="ber", choices=["ber", "bler"])
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    if args.ber_csv:
        plot_ber_snr(args.ber_csv, args.out, metric=args.metric)
    elif args.ablation_csv:
        plot_ablation(args.ablation_csv, args.out)
    elif args.complexity_csv:
        plot_complexity(args.complexity_csv, args.out)
    else:
        raise SystemExit("Provide one of --ber_csv / --ablation_csv / --complexity_csv")


if __name__ == "__main__":
    main()
