"""
Classical Successive Over-Relaxation (SOR) and Gauss-Seidel (GS, the
special case omega=1) iterative detectors, following Appendix C.1 exactly.

Pseudo-code (Appendix C.1) precomputes:
    A = H^T H + sigma^2 I
    b = H^T y
and performs the component-wise update, for i = 1..K:
    x_i^(k+1) = (1-w) x_i^(k) + (w / A_ii) * (b_i - sum_{j<i} A_ij x_j^(k+1)
                                                    - sum_{j>i} A_ij x_j^(k))

This is mathematically identical to the matrix splitting A = D + L + U
(diagonal / strictly-lower / strictly-upper) form:
    (D + w L) x^(k+1) = w b - (w U + (w-1) D) x^(k)

We use the matrix-splitting form because it is exact, numerically
equivalent to the pseudo-code, and vectorizes over a batch of samples via a
batched lower-triangular solve -- which is exactly the operation reused
(with *learnable* w per layer) in the DU-SOR network (src/detectors/du_sor.py).
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_triangular


def _split(A: np.ndarray):
    """A: (K, K) -> D (diag matrix), L (strictly lower), U (strictly upper)."""
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    return D, L, U


def sor_detect(y: np.ndarray, H: np.ndarray, sigma2: np.ndarray,
               omega: float = 1.5, max_iterations: int = 50, tol: float = 1e-6):
    """y: (batch, K), H: (batch, K, K), sigma2: (batch,).

    Returns (x_hat, iters_used) where iters_used is an array (batch,) with
    the number of iterations each sample actually took before convergence
    (used for the convergence-speed comparisons in Sec. 4.4.1).
    """
    batch, K, _ = H.shape
    Ht = np.transpose(H, (0, 2, 1))
    A = np.einsum("bij,bjk->bik", Ht, H) + sigma2[:, None, None] * np.eye(K)[None, :, :]
    b = np.einsum("bij,bj->bi", Ht, y)

    x = np.zeros((batch, K))
    iters_used = np.full(batch, max_iterations)
    active = np.ones(batch, dtype=bool)

    for k in range(max_iterations):
        if not active.any():
            break
        x_prev = x.copy()
        for idx in np.where(active)[0]:
            D, L, U = _split(A[idx])
            M = D + omega * L                                  # lower triangular
            rhs = omega * b[idx] - (omega * U + (omega - 1) * D) @ x_prev[idx]
            x[idx] = solve_triangular(M, rhs, lower=True)

        diff = np.linalg.norm(x - x_prev, axis=-1)
        newly_converged = active & (diff < tol)
        iters_used[newly_converged] = k + 1
        active = active & (diff >= tol)

    return x, iters_used


def gauss_seidel_detect(y, H, sigma2, max_iterations: int = 50, tol: float = 1e-6):
    """Gauss-Seidel is the special case of SOR with omega = 1."""
    return sor_detect(y, H, sigma2, omega=1.0, max_iterations=max_iterations, tol=tol)
