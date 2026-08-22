"""
Classical linear detectors: Zero-Forcing (ZF) and Minimum Mean Square Error
(MMSE). Both require an O(K^3) matrix inversion (Chapter 2, Sec. 2.3.2;
Chapter 3, Sec. 3.2.2).

    ZF:   x_hat = H^+ y                      (Moore-Penrose pseudo-inverse)
    MMSE: x_hat = (H^T H + sigma^2 I)^-1 H^T y
"""

from __future__ import annotations
import numpy as np


def zf_detect(y: np.ndarray, H: np.ndarray, sigma2=None) -> np.ndarray:
    """y: (batch, K), H: (batch, K, K) -> x_hat: (batch, K)."""
    H_pinv = np.linalg.pinv(H)                       # (batch, K, K)
    x_hat = np.einsum("bij,bj->bi", H_pinv, y)
    return x_hat


def mmse_detect(y: np.ndarray, H: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """y: (batch, K), H: (batch, K, K), sigma2: (batch,) -> x_hat: (batch, K)."""
    batch, K, _ = H.shape
    Ht = np.transpose(H, (0, 2, 1))
    HtH = np.einsum("bij,bjk->bik", Ht, H)
    A = HtH + sigma2[:, None, None] * np.eye(K)[None, :, :]
    b = np.einsum("bij,bj->bi", Ht, y)
    x_hat = np.linalg.solve(A, b[..., None])[..., 0]
    return x_hat
