# Enhanced Low-Complexity Signal Detection in Large-Scale Optical MIMO
### Deep Unfolded Successive Over-Relaxation (DU-SOR) Network

This repository is a from-scratch PyTorch/NumPy implementation of the
**DU-SOR** detector described in the accompanying thesis, *"Enhanced
Low-Complexity Signal Detection in Large-Scale Optical MIMO Systems"*
(Chapters 3–4 and Appendix). It targets signal detection for large-scale
optical MIMO under intensity-modulation/direct-detection (IM/DD)
constraints, for both indoor Visible Light Communication (VLC) and outdoor
Free-Space Optical (FSO) links.

DU-SOR "unfolds" the classical iterative Successive Over-Relaxation (SOR)
algorithm into a fixed-depth neural network: each layer is one SOR
iteration, but the relaxation factor `omega_l` (and the initial estimate
`x^(0)`) become **learnable parameters** trained end-to-end, with a ReLU
enforcing the non-negativity required by IM/DD signaling. The result
targets near-ML detection accuracy at `O(L*K^2)` complexity, versus
`O(K^3)` for MMSE/ZF and `O(M^K)` for exhaustive ML search.

> **Note on provenance:** the original uploaded project contained only the
> thesis chapters (`.docx`) — no source code. This implementation was
> built from scratch based on the system model, pseudo-code (Appendix C),
> and simulation parameters (Appendix A) described in those chapters. It
> is a faithful-effort reference implementation, not code recovered from
> an original project repository. See [Implementation notes](#implementation-notes-vs-the-thesis)
> below for the small set of modeling choices this required.

---

## Directory structure

```
dusor_project/
├── README.md                          <- this file
├── requirements.txt
├── configs/
│   └── default.yaml                   <- all simulation parameters (Appendix A)
├── src/
│   ├── channels/
│   │   ├── vlc_channel.py             <- Lambertian VLC channel model (Sec. 4.2.2a)
│   │   ├── fso_channel.py             <- FSO path loss + turbulence + pointing error (Sec. 4.2.2b)
│   │   └── noise.py                   <- shot + thermal noise model (Sec. 4.2.3)
│   ├── detectors/
│   │   ├── linear.py                  <- Zero-Forcing (ZF) and MMSE detectors
│   │   ├── ml_detector.py             <- brute-force Maximum Likelihood (small K only)
│   │   ├── iterative.py               <- classical SOR and Gauss-Seidel detectors (Appendix C.1)
│   │   └── du_sor.py                  <- the DU-SOR network itself (Sec. 3.4-3.5, Appendix C.2)
│   ├── data/
│   │   └── dataset.py                 <- Monte Carlo dataset generator (Sec. 3.5.4)
│   └── utils/
│       ├── modulation.py              <- M-PAM (Gray-coded, non-negative) mapper
│       └── metrics.py                 <- BER, BLER, FLOPs, runtime benchmarking
├── scripts/
│   ├── train.py                       <- train a DU-SOR model
│   ├── evaluate.py                    <- BER/BLER vs SNR, DU-SOR vs all baselines (Fig. 4.1, 4.4)
│   ├── run_ber_vs_snr.py              <- convenience wrapper: train (if needed) + evaluate
│   ├── run_ablation_layers.py         <- ablation over number of layers L (Sec. 4.5.1, Fig. 4.9)
│   ├── run_complexity_benchmark.py    <- runtime/FLOPs vs K for all detectors (Sec. 4.4.2, Fig. 4.8)
│   └── plot_results.py                <- turn the CSV outputs above into figures
├── tests/
│   └── test_detectors.py              <- correctness checks (PAM round-trip, ZF/MMSE recovery,
│                                          SOR update matches Appendix C.1 pseudo-code exactly,
│                                          DU-SOR non-negativity / omega range)
├── checkpoints/                       <- trained model checkpoints land here
└── results/                           <- CSV/PNG outputs from the scripts land here
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.1+. GPU is optional — everything falls back
to CPU automatically (`--device cpu`), though training with the full
`training.train_samples: 100000` from Appendix A.5 is much faster on GPU.

Run the correctness tests first:

```bash
python -m pytest tests/ -v
# or, without pytest installed:
python tests/test_detectors.py
```

---

## Quick start

**1. Train a DU-SOR model** (defaults come from `configs/default.yaml`,
i.e. K=64, L=10, VLC channel, 4-PAM, 100 epochs, matching Appendix A):

```bash
python scripts/train.py --config configs/default.yaml --out checkpoints/du_sor.pt
```

Useful overrides for a faster first run:

```bash
python scripts/train.py --config configs/default.yaml \
    --K 32 --num_layers 10 --epochs 20 --train_samples 20000 --val_samples 4000 \
    --out checkpoints/du_sor_k32.pt
```

**2. Evaluate against all baselines (Fig. 4.1 / 4.4 style BER/BLER vs SNR):**

```bash
python scripts/evaluate.py --checkpoint checkpoints/du_sor.pt \
    --detectors mmse zf sor gs du_sor \
    --out results/ber_vs_snr.csv

python scripts/plot_results.py --ber_csv results/ber_vs_snr.csv --out results/ber_vs_snr.png
python scripts/plot_results.py --ber_csv results/ber_vs_snr.csv --metric bler --out results/bler_vs_snr.png
```

Add `--detectors ... ml` only for small systems (`K <= benchmark.ml_max_K`,
default 4) since ML brute-force search is `O(M^K)`.

**3. Layer-count ablation (Fig. 4.9, Sec. 4.5.1):**

```bash
python scripts/run_ablation_layers.py --config configs/default.yaml \
    --layer_values 2 5 10 15 20 --out results/ablation_layers.csv
python scripts/plot_results.py --ablation_csv results/ablation_layers.csv --out results/ablation.png
```

**4. Complexity / runtime benchmark (Table 4.1, Fig. 4.8, Sec. 4.4.2-4.4.3):**

```bash
python scripts/run_complexity_benchmark.py --config configs/default.yaml \
    --K_values 8 16 32 64 128 --out results/complexity_benchmark.csv
python scripts/plot_results.py --complexity_csv results/complexity_benchmark.csv --out results/runtime.png
```

**5. One-shot pipeline** (trains if no checkpoint exists, then evaluates):

```bash
python scripts/run_ber_vs_snr.py --config configs/default.yaml
```

### Switching to the FSO channel

Edit `configs/default.yaml`:

```yaml
channel:
  type: fso     # was 'vlc'
```

or copy the file (`configs/fso.yaml`) and pass `--config configs/fso.yaml`
to any script. The `channel.fso.turbulence_strength` field toggles between
`moderate` and `strong` Gamma-Gamma turbulence (Fig. 4.6).

---

## What's implemented, mapped to the thesis

| Thesis section | Code |
|---|---|
| Sec. 3.2 – optical MIMO system model, non-negativity constraint | `src/utils/modulation.py` (non-negative PAM), `src/data/dataset.py` |
| Sec. 3.3 / Appendix C.1 – classical SOR | `src/detectors/iterative.py` (`sor_detect`, `gauss_seidel_detect`) |
| Sec. 3.4 / Appendix C.2 – DU-SOR unfolding, learnable `omega_l`, learnable `x^(0)`, ReLU non-negativity | `src/detectors/du_sor.py` |
| Sec. 3.5.3 – MSE training loss | `scripts/train.py` (`nn.MSELoss`) |
| Sec. 3.5.4 / Appendix A.5 – dataset generation, Adam optimizer, LR-on-plateau, 100 epochs | `src/data/dataset.py`, `scripts/train.py`, `configs/default.yaml` |
| Sec. 3.6.1 / Sec. 4.4.2 – `O(K^3)` MMSE/ZF vs `O(L*K^2)` DU-SOR complexity | `src/utils/metrics.py::theoretical_flops` |
| Sec. 4.2.2(a) – VLC Lambertian channel | `src/channels/vlc_channel.py` |
| Sec. 4.2.2(b) – FSO path loss + Log-Normal/Gamma-Gamma turbulence + pointing error | `src/channels/fso_channel.py` |
| Sec. 4.2.3 – shot + thermal noise | `src/channels/noise.py` |
| Sec. 4.2.4 – ML / ZF / MMSE / GS / SOR / DU-SOR benchmark set | `src/detectors/*`, `scripts/evaluate.py` |
| Sec. 4.2.5 – BER, BLER, FLOPs, runtime (ms/frame) metrics | `src/utils/metrics.py` |
| Sec. 4.5.1 / Fig. 4.9 – ablation over number of layers L | `scripts/run_ablation_layers.py` |
| Sec. 4.4.1 / Fig. 4.5, 4.12 – learned `omega_l` schedule, convergence trajectory | `DUSORNet.omega_schedule()`, `forward(..., return_all_layers=True)` |
| Appendix B.2 – learned vs. zero vs. random `x^(0)` initialization | `DUSORNet.x0` is a learnable `nn.Parameter` by default; swap for a fixed buffer to reproduce the zero/random-init ablation |
| Appendix D – SOR convergence guaranteed for `0 < omega < 2` | `DUSORLayer.omega` uses a `2*sigmoid(...)` parameterization so learned values can never leave this range |

Not implemented (out of scope for a reference implementation, called out
explicitly rather than silently skipped):
- **MMNet** and **DPST** (Sec. 4.2.4, 4.4.1) — third-party deep-learning
  baselines cited from prior work in the thesis; only sketched at a high
  level in the chapters (no architecture details), so they are not
  reproduced here. `scripts/evaluate.py --detectors` intentionally lists
  only the baselines that are *fully specified* in Chapters 3–4 and
  Appendix C (ML, ZF, MMSE, GS, SOR, DU-SOR).
- Full room radiosity / diffuse-reflection VLC modeling — the VLC model
  implements the Lambertian line-of-sight term exactly per Sec. 4.2.2(a),
  plus a simplified constant diffuse-floor term scaled by wall
  reflectivity, rather than a full multi-bounce radiosity simulation.

---

## Implementation notes vs. the thesis

The thesis chapters describe the system model, algorithm, and parameters
in full mathematical/textual detail, but (being a written thesis, not a
code repository) leave a handful of concrete engineering choices
unspecified. These were resolved as follows, and are called out in code
comments at each location:

1. **Channel normalization.** Raw physical VLC/FSO gains are extremely
   small (`~1e-5` or smaller), which is physically correct but numerically
   ill-conditions the SOR normal equations. Each sampled channel matrix is
   normalized to unit average per-link gain (`src/data/dataset.py
   ::_normalize_channel`), equivalent to an automatic-gain-control stage;
   only the *relative* channel structure that the detector conditions on
   is affected, not the detection problem itself.
2. **SOR/DU-SOR update rule.** Appendix C gives a component-wise
   (Gauss-Seidel-style) pseudo-code. This is implemented via the
   mathematically equivalent matrix-splitting form
   `(D + w*L) x^(k+1) = w*b - (w*U + (w-1)*D) x^(k)`, solved with a batched
   lower-triangular solve — exact, differentiable, and GPU-batchable. A
   unit test (`tests/test_detectors.py
   ::test_sor_matches_component_wise_pseudocode`) verifies the two forms
   agree bit-for-bit on a hand-built example.
3. **Learnable `omega_l` parameterization.** Implemented as
   `2 * sigmoid(raw_omega)` so that gradient descent can never push a
   layer's relaxation factor outside the `(0, 2)` convergence guarantee
   from Appendix D, without needing a hard clamp.
4. **PAM constellation.** Levels are placed at `{0, 1, ..., M-1}` (Gray
   coded), scaled to unit average energy, consistent with the
   non-negativity constraint of Sec. 3.2.1 and the M-PAM scheme of
   Sec. 4.2.1.
5. **FSO MIMO extension.** The thesis's FSO model (Sec. 4.2.2b) is written
   for a general channel gain product `L_path * h_turb * h_pointing`; this
   is applied independently per transmit/receive aperture pair to build
   the full `K x K` channel matrix for a MIMO array of FSO apertures.

---

## Reproducing headline thesis numbers

The thesis's specific reported figures (e.g. "0.21 ms per frame", "7x fewer
FLOPs than MMSE", "5x faster convergence than classical SOR") were obtained
on the hardware and full-scale settings described in Sec. 4.2.5 (NVIDIA
RTX 4090 / Intel i9-14900K, `K=128`, 100k training samples, 100 epochs).
Running `scripts/run_complexity_benchmark.py` and
`scripts/run_ablation_layers.py` with the full `configs/default.yaml`
settings on comparable hardware should reproduce these trends (DU-SOR
converging within ~10 layers, quadratic-vs-cubic scaling, etc.), though
exact numbers will vary with hardware, PyTorch version, and random seed.
