"""
Receiver noise model: shot noise + thermal noise (Chapter 4, Sec. 4.2.3;
Appendix A.4).

    sigma_shot^2 = 2 * q * R * P_r * B          (signal-dependent)
    sigma_th^2   = 4 * k * T * B / R_L           (constant, AWGN)
    sigma_total^2 = sigma_shot^2 + sigma_th^2
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class NoiseConfig:
    electron_charge_C: float = 1.6e-19
    responsivity_A_per_W: float = 0.6
    bandwidth_Hz: float = 1.0e9
    boltzmann_J_per_K: float = 1.38e-23
    temperature_K: float = 300.0
    load_resistance_ohm: float = 50.0


class NoiseModel:
    def __init__(self, cfg: NoiseConfig):
        self.cfg = cfg
        # Thermal noise variance is constant (does not depend on received power)
        self.sigma_thermal2 = (
            4 * cfg.boltzmann_J_per_K * cfg.temperature_K * cfg.bandwidth_Hz
            / cfg.load_resistance_ohm
        )

    def shot_noise_variance(self, received_power: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        return 2 * cfg.electron_charge_C * cfg.responsivity_A_per_W * np.abs(received_power) * cfg.bandwidth_Hz

    def total_noise_variance(self, received_power: np.ndarray) -> np.ndarray:
        return self.shot_noise_variance(received_power) + self.sigma_thermal2

    def add_noise(self, y_clean: np.ndarray, snr_db: float, rng: np.random.Generator):
        """Add noise so that the *effective* SNR of the link matches `snr_db`
        (SNR defined as average signal power over average total-noise
        power, per Sec. 4.2.5). The relative split between shot noise and
        thermal noise (Sec. 4.2.3) is preserved -- only the overall scale is
        set by the target SNR, since the absolute physical noise power at
        the receiver's actual (very low) optical power levels is not, by
        itself, a meaningful quantity once the channel has been normalized
        (see OpticalMIMODataset._normalize_channel).

        Returns (y_noisy, sigma2) where sigma2 is the *scalar* per-sample
        noise variance used by the detectors' MMSE normal equations
        (A = H^T H + sigma^2 I)."""
        signal_power = np.mean(y_clean ** 2, axis=-1, keepdims=True)          # (batch, 1)

        # Relative shape of the noise (shot proportional to |y|, plus a
        # constant thermal floor), used only to distribute noise power
        # across antennas -- not its absolute scale.
        shape_var = self.total_noise_variance(np.abs(y_clean))                # (batch, K)
        shape_var = shape_var / np.maximum(np.mean(shape_var, axis=-1, keepdims=True), 1e-30)

        target_snr_linear = 10 ** (snr_db / 10)
        sigma2 = (signal_power / target_snr_linear).squeeze(-1)               # (batch,) -- mean noise variance

        noise_var_per_antenna = sigma2[:, None] * shape_var                   # (batch, K)
        noise = rng.normal(0.0, 1.0, size=y_clean.shape) * np.sqrt(noise_var_per_antenna)
        y_noisy = y_clean + noise
        return y_noisy, sigma2
