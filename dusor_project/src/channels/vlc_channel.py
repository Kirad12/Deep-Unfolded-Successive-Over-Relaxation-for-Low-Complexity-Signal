"""
Indoor Visible Light Communication (VLC) channel model based on the
Lambertian radiation pattern (Chapter 4, Sec. 4.2.2(a); Appendix A.2).

    h_ij = (m+1) * A_r / (2*pi*d_ij^2) * cos(phi_ij)^m * T_s * g * cos(psi_ij)

where:
    m      : Lambertian order (from the transmitter semi-angle)
    A_r    : photodetector area
    d_ij   : distance between LED i and photodetector j
    phi_ij : irradiance angle (at the transmitter)
    psi_ij : incidence angle (at the receiver)
    T_s    : optical filter gain
    g      : concentrator gain

A first-order diffuse (wall-reflection) term is added as a small constant
offset scaled by the wall reflectivity, approximating the multipath
contribution discussed in Sec. 4.2.2 without requiring a full radiosity
simulation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class VLCChannelConfig:
    room_dims_m: tuple = (5.0, 5.0, 3.0)     # (L, W, H)
    lambertian_order: float = 1.0             # m
    photodetector_area_cm2: float = 1.0       # A_r
    optical_filter_gain: float = 1.0          # T_s
    concentrator_gain: float = 1.0            # g
    wall_reflectivity: float = 0.8


def _grid_positions(n: int, dims_xy, z: float, rng: np.random.Generator, jitter=0.15):
    """Place n transmitters/receivers on an (approximately) square grid within
    the given (x, y) footprint at fixed height z, with small random jitter so
    that channel realizations differ across Monte Carlo samples."""
    side = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(dims_xy[0] * 0.1, dims_xy[0] * 0.9, side)
    ys = np.linspace(dims_xy[1] * 0.1, dims_xy[1] * 0.9, side)
    grid = np.array([(x, y) for x in xs for y in ys])[:n]
    grid = grid + rng.normal(scale=jitter, size=grid.shape)
    z_col = np.full((n, 1), z)
    return np.hstack([grid, z_col])


class VLCChannel:
    def __init__(self, cfg: VLCChannelConfig):
        self.cfg = cfg
        self.A_r = cfg.photodetector_area_cm2 * 1e-4  # cm^2 -> m^2
        self.m = cfg.lambertian_order

    def sample(self, K: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        """Return a batch of real-valued channel matrices H, shape (batch, K, K).

        Transmit LEDs (K of them) are placed on the ceiling, receive
        photodetectors (K of them) are placed on the receiver plane (desk
        height), consistent with a square MIMO configuration (Eq. 4.1).
        """
        L, W, H_room = self.cfg.room_dims_m
        H_batch = np.empty((batch_size, K, K), dtype=np.float64)

        for b in range(batch_size):
            tx_pos = _grid_positions(K, (L, W), z=H_room, rng=rng)          # ceiling, z = room height
            rx_pos = _grid_positions(K, (L, W), z=0.85, rng=rng)           # desk height

            # Pairwise geometry
            diff = tx_pos[:, None, :] - rx_pos[None, :, :]                  # (K, K, 3), tx i -> rx j
            d = np.linalg.norm(diff, axis=-1)                               # distance
            d = np.maximum(d, 1e-3)

            # Both LEDs (pointing straight down) and PDs (pointing straight up)
            # are assumed vertically oriented, so irradiance angle == incidence
            # angle == angle from vertical.
            vertical_drop = np.abs(diff[..., 2])
            cos_angle = vertical_drop / d
            cos_angle = np.clip(cos_angle, 0.0, 1.0)

            los_gain = (
                (self.m + 1) * self.A_r / (2 * np.pi * d ** 2)
                * cos_angle ** self.m
                * self.cfg.optical_filter_gain
                * self.cfg.concentrator_gain
                * cos_angle
            )

            # Simple first-order diffuse component: a small uniform floor
            # proportional to wall reflectivity, representing non-LoS energy
            # collected via wall reflections (approximation of Sec. 4.2.2).
            diffuse = self.cfg.wall_reflectivity * los_gain.mean() * 0.05
            H_batch[b] = los_gain + diffuse

        return H_batch
