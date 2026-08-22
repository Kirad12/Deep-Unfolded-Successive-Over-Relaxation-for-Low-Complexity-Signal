"""
Deep Unfolded Successive Over-Relaxation (DU-SOR) network.

Implements Chapter 3 (Sec. 3.4 - 3.5) and the forward-pass pseudo-code of
Appendix C.2: each of the L layers performs one SOR-type update with a
*layer-specific, learnable* relaxation factor omega_l, followed by a ReLU
to enforce the IM/DD non-negativity constraint (Sec. 3.4.4). The initial
estimate x^(0) is also a learnable parameter (Sec. 3.4.3 / Appendix B.2).

Matrix-splitting formulation (equivalent to the component-wise pseudo-code,
see src/detectors/iterative.py for the derivation):

    A = H^T H + sigma^2 I  = D + L + U
    x^(l+1) = ReLU( (D + w_l L)^-1 [ w_l b - (w_l U + (w_l-1) D) x^(l) ] )

The batched lower-triangular solve is implemented with
`torch.linalg.solve_triangular`, which is differentiable, GPU-friendly, and
avoids an explicit K-length Python loop per layer -- giving the O(K^2)
per-layer cost claimed in Sec. 3.6.1 / Sec. 4.4.2 (the triangular solve
itself is O(K^2) given the pre-formed matrix, and forming D+w*L, and the
matrix-vector product for the RHS, are also O(K^2)).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class DUSORLayer(nn.Module):
    """A single unfolded SOR iteration with a learnable relaxation factor."""

    def __init__(self, init_omega: float = 1.2):
        super().__init__()
        # Unconstrained parameter; mapped through a scaled sigmoid to keep
        # omega inside the theoretically-guaranteed convergence range
        # (0, 2) for symmetric positive-definite A (Appendix D).
        raw_init = torch.logit(torch.tensor(init_omega / 2.0).clamp(1e-3, 1 - 1e-3))
        self.raw_omega = nn.Parameter(raw_init)

    @property
    def omega(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.raw_omega)

    def forward(self, x_prev, D, L, U, b):
        """
        x_prev: (batch, K, 1)
        D, L, U: (batch, K, K)   diagonal / strictly-lower / strictly-upper parts of A
        b: (batch, K, 1)
        """
        w = self.omega
        M = D + w * L                                  # (batch, K, K), lower triangular
        rhs = w * b - torch.bmm(w * U + (w - 1.0) * D, x_prev)   # (batch, K, 1)
        x_new = torch.linalg.solve_triangular(M, rhs, upper=False)
        x_new = torch.relu(x_new)                       # enforce IM/DD non-negativity (Sec. 3.4.4)
        return x_new


class DUSORNet(nn.Module):
    """L-layer Deep Unfolded SOR network (Fig. 3.1)."""

    def __init__(self, K: int, num_layers: int = 10, init_omega: float = 1.2):
        super().__init__()
        self.K = K
        self.num_layers = num_layers

        # Learnable initial estimate x^(0), shared across the batch and
        # broadcast at inference time (Sec. 3.4.3 / Appendix B.2).
        self.x0 = nn.Parameter(torch.zeros(K, 1))

        self.layers = nn.ModuleList(
            [DUSORLayer(init_omega=init_omega) for _ in range(num_layers)]
        )

    @staticmethod
    def _split(A: torch.Tensor):
        """A: (batch, K, K) -> D, L, U (diagonal / strictly-lower / strictly-upper)."""
        K = A.shape[-1]
        eye = torch.eye(K, device=A.device, dtype=A.dtype)
        diag_vals = torch.diagonal(A, dim1=-2, dim2=-1)          # (batch, K)
        D = diag_vals.unsqueeze(-1) * eye.unsqueeze(0)
        lower_mask = torch.tril(torch.ones(K, K, device=A.device, dtype=torch.bool), diagonal=-1)
        upper_mask = torch.triu(torch.ones(K, K, device=A.device, dtype=torch.bool), diagonal=1)
        L = torch.where(lower_mask.unsqueeze(0), A, torch.zeros_like(A))
        U = torch.where(upper_mask.unsqueeze(0), A, torch.zeros_like(A))
        return D, L, U

    def forward(self, y: torch.Tensor, H: torch.Tensor, sigma2: torch.Tensor,
                return_all_layers: bool = False):
        """
        y: (batch, K)          received signal
        H: (batch, K, K)       channel matrix
        sigma2: (batch,)       noise variance per sample

        Returns x_hat: (batch, K), or a list of per-layer estimates if
        return_all_layers=True (used for the convergence-speed plots,
        Fig. 4.5 / Fig. 4.7).
        """
        batch = y.shape[0]
        K = self.K
        eye = torch.eye(K, device=H.device, dtype=H.dtype).unsqueeze(0)

        Ht = H.transpose(-1, -2)
        A = torch.bmm(Ht, H) + sigma2.view(-1, 1, 1) * eye         # (batch, K, K)
        b = torch.bmm(Ht, y.unsqueeze(-1))                          # (batch, K, 1)
        D, L, U = self._split(A)

        x = self.x0.unsqueeze(0).expand(batch, -1, -1)              # learned x^(0)
        trajectory = [x.squeeze(-1)]

        for layer in self.layers:
            x = layer(x, D, L, U, b)
            trajectory.append(x.squeeze(-1))

        if return_all_layers:
            return trajectory  # list of (batch, K) tensors, length num_layers+1
        return x.squeeze(-1)

    def omega_schedule(self):
        """Return the learned omega_l values across layers (for Fig. 4.12)."""
        return [layer.omega.item() for layer in self.layers]
