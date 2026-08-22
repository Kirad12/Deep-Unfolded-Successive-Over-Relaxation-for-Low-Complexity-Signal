"""
Outdoor Free-Space Optical (FSO) channel model (Chapter 4, Sec. 4.2.2(b);
Appendix A.3).

    h_ij = L_path * h_turb_ij * h_pointing_ij

    L_path      : deterministic free-space path loss (attenuation over distance)
    h_turb      : turbulence-induced fading
                    - Log-Normal for weak turbulence
                    - Gamma-Gamma for moderate/strong turbulence
    h_pointing  : pointing-error loss from Gaussian transmitter/receiver
                  misalignment (jitter)

Extended here to a K x K MIMO array of apertures: each (i, j) transmit/receive
aperture pair gets an independent turbulence and pointing-error realization
sharing the same link distance and path loss.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class FSOChannelConfig:
    link_distance_km: float = 1.0
    wavelength_nm: float = 1550.0
    attenuation_db_per_km: float = 0.2
    turbulence_model: str = "gamma_gamma"       # 'lognormal' or 'gamma_gamma'
    gamma_gamma_moderate: tuple = (4.0, 1.9)     # (alpha, beta)
    gamma_gamma_strong: tuple = (2.0, 1.8)
    turbulence_strength: str = "moderate"        # 'moderate' or 'strong'
    pointing_error_jitter_mrad: tuple = (0.1, 0.6)


class FSOChannel:
    def __init__(self, cfg: FSOChannelConfig):
        self.cfg = cfg
        # Path loss: L = 10^(-alpha[dB/km] * d[km] / 10)
        self.path_loss = 10 ** (-(cfg.attenuation_db_per_km * cfg.link_distance_km) / 10)

    def _turbulence_fading(self, size, rng: np.random.Generator) -> np.ndarray:
        cfg = self.cfg
        if cfg.turbulence_model == "lognormal":
            sigma_r2 = 0.3  # weak-turbulence Rytov variance (representative)
            mu = -0.5 * sigma_r2
            X = rng.normal(mu, np.sqrt(sigma_r2), size=size)
            return np.exp(X)
        elif cfg.turbulence_model == "gamma_gamma":
            alpha, beta = (
                cfg.gamma_gamma_strong if cfg.turbulence_strength == "strong"
                else cfg.gamma_gamma_moderate
            )
            x = rng.gamma(shape=alpha, scale=1.0 / alpha, size=size)
            y = rng.gamma(shape=beta, scale=1.0 / beta, size=size)
            return x * y
        else:
            raise ValueError(f"Unknown turbulence model '{cfg.turbulence_model}'")

    def _pointing_loss(self, size, rng: np.random.Generator) -> np.ndarray:
        """Gaussian-beam pointing-error loss from a random radial displacement
        r ~ Rayleigh(sigma_jitter). Uses the standard approximation
        h_p ~ A0 * exp(-2 r^2 / w_eq^2), with w_eq normalized to 1 so that the
        jitter (in mrad, converted to an equivalent normalized displacement)
        directly controls the loss magnitude."""
        lo, hi = self.cfg.pointing_error_jitter_mrad
        sigma_jitter = rng.uniform(lo, hi, size=size)  # mrad, per-link jitter std
        r = rng.rayleigh(scale=sigma_jitter, size=size)
        w_eq = hi * 3.0  # equivalent beam-width scale relative to max jitter
        A0 = 1.0
        return A0 * np.exp(-2 * (r ** 2) / (w_eq ** 2))

    def sample(self, K: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        """Return a batch of real-valued FSO channel matrices, shape (batch, K, K)."""
        size = (batch_size, K, K)
        turb = self._turbulence_fading(size, rng)
        point = self._pointing_loss(size, rng)
        H = self.path_loss * turb * point
        return H
