"""
Evaluation metrics used throughout Chapter 4:
  - Bit Error Rate (BER)
  - Block Error Rate (BLER)
  - Theoretical FLOPs for O(K^2) / O(K^3) detectors
  - Empirical runtime benchmarking helper
"""

from __future__ import annotations
import time
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from src.utils.modulation import PAM


def compute_ber(x_hat: np.ndarray, x_true: np.ndarray, pam: PAM) -> float:
    """Symbol-level estimates -> hard decision -> bit error rate."""
    sym_hat = pam.nearest_symbol(x_hat)
    bits_hat = pam.demodulate_hard(sym_hat)
    bits_true = pam.demodulate_hard(x_true)
    return float(np.mean(bits_hat != bits_true))


def compute_bler(x_hat: np.ndarray, x_true: np.ndarray, pam: PAM) -> float:
    """A 'block' = one full transmitted vector (K symbols). BLER = fraction
    of blocks with at least one symbol error."""
    sym_hat = pam.nearest_symbol(x_hat)
    sym_true = pam.nearest_symbol(x_true)
    block_has_error = np.any(sym_hat != sym_true, axis=-1)
    return float(np.mean(block_has_error))


def theoretical_flops(K: int, detector: str, num_layers: int = 10) -> int:
    """Rough theoretical FLOP counts used for the complexity comparisons in
    Sec. 4.4.2 / Table 4.1. These are order-of-magnitude estimates
    (dominant matrix-multiplication terms), consistent with the thesis:
        - ML:      O(M^K)            (exponential, only meaningful for small K)
        - ZF/MMSE: O(K^3)            (matrix inversion)
        - GS/SOR:  O(K^2) per iteration
        - DU-SOR:  O(K^2) per layer, L layers -> O(L*K^2)
    """
    detector = detector.lower()
    if detector == "zf" or detector == "mmse":
        return int(2 * K ** 3)                      # inversion + matmuls
    if detector in ("sor", "gs"):
        return int(2 * K ** 2 * 50)                  # ~50 iterations to converge (Sec. 4.4.1)
    if detector == "du_sor":
        return int(2 * K ** 2 * num_layers)          # fixed L layers
    if detector == "ml":
        return None  # exponential; reported separately, not plotted linearly
    raise ValueError(f"Unknown detector '{detector}'")


def benchmark_runtime(detect_fn, y, H, sigma2, n_repeats: int = 50, device: str = "cpu") -> float:
    """Average wall-clock runtime (ms) per call of detect_fn(y, H, sigma2)."""
    if torch is not None and device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    # Warmup
    for _ in range(5):
        detect_fn(y, H, sigma2)
    if torch is not None and device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_repeats):
        detect_fn(y, H, sigma2)
    if torch is not None and device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / n_repeats
