"""
Maximum Likelihood (ML) detector via brute-force search over all M^K
candidate symbol vectors (Chapter 2, Sec. 2.3.1; Chapter 4, Sec. 4.2.4).

This serves as the performance *upper bound* in Fig. 4.1, but its
exponential complexity O(M^K) makes it tractable only for small systems
(the thesis benchmarks ML for K <= 4, see Appendix / benchmark.ml_max_K).
"""

from __future__ import annotations
import itertools
import numpy as np

from src.utils.modulation import PAM


def ml_detect(y: np.ndarray, H: np.ndarray, sigma2, pam: PAM, max_K: int = 4) -> np.ndarray:
    """y: (batch, K), H: (batch, K, K) -> x_hat: (batch, K).

    Exhaustively evaluates every candidate vector in {levels}^K and picks the
    one minimizing ||y - Hx||^2 (equivalent to ML under AWGN).
    """
    batch, K = y.shape
    if K > max_K:
        raise ValueError(
            f"ML brute-force requested for K={K} > max_K={max_K}. "
            "This is exponential in K and intentionally gated; reduce K or "
            "raise benchmark.ml_max_K if you really want to wait."
        )

    candidates = np.array(list(itertools.product(pam.levels, repeat=K)))  # (M^K, K)
    x_hat = np.empty((batch, K))
    for b in range(batch):
        residual = y[b][None, :] - candidates @ H[b].T                     # (M^K, K)
        dist2 = np.sum(residual ** 2, axis=-1)
        best = np.argmin(dist2)
        x_hat[b] = candidates[best]
    return x_hat
