"""
Monte Carlo dataset generation for training/evaluating the DU-SOR network
(Chapter 3, Sec. 3.5.4 "Training Procedure"; Appendix A.5).

Each sample consists of:
    x  : (K,)  transmitted M-PAM symbol vector, x >= 0
    H  : (K,K) channel realization (VLC or FSO, per config)
    y  : (K,)  received signal, y = H x + n
    sigma2 : noise variance used to build the MMSE normal equations

SNR is drawn uniformly at random per-sample from the configured range so
that a single trained network generalizes across the whole SNR sweep used
for the BER/BLER curves (Fig. 4.1 - 4.6), consistent with "training across
diverse channel realizations" in Sec. 3.5.4.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object

from src.channels.vlc_channel import VLCChannel, VLCChannelConfig
from src.channels.fso_channel import FSOChannel, FSOChannelConfig
from src.channels.noise import NoiseModel, NoiseConfig
from src.utils.modulation import PAM


def build_channel(channel_cfg: dict):
    if channel_cfg["type"] == "vlc":
        return VLCChannel(VLCChannelConfig(**channel_cfg["vlc"]))
    elif channel_cfg["type"] == "fso":
        fso_kwargs = dict(channel_cfg["fso"])
        fso_kwargs["gamma_gamma_moderate"] = tuple(fso_kwargs.pop("gamma_gamma_params")["moderate"])
        fso_kwargs["gamma_gamma_strong"] = tuple(channel_cfg["fso"]["gamma_gamma_params"]["strong"])
        return FSOChannel(FSOChannelConfig(**fso_kwargs))
    else:
        raise ValueError(f"Unknown channel type '{channel_cfg['type']}'")


@dataclass
class OpticalMIMOSample:
    x: np.ndarray
    H: np.ndarray
    y: np.ndarray
    sigma2: float
    snr_db: float


class OpticalMIMODataset(Dataset):
    """Generates (or pre-generates) Monte Carlo samples of the optical MIMO
    link: random channel realization + random PAM symbol vector + noise at a
    randomly-drawn SNR within the configured range."""

    def __init__(self, K: int, modulation_order: int, channel_cfg: dict,
                 noise_cfg: dict, snr_db_range, n_samples: int, seed: int = 0,
                 precompute: bool = True):
        self.K = K
        self.pam = PAM(modulation_order)
        self.channel = build_channel(channel_cfg)
        self.noise_model = NoiseModel(NoiseConfig(**noise_cfg))
        self.snr_lo, self.snr_hi = snr_db_range
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

        self._data = None
        if precompute:
            self._data = self._generate(n_samples)

    def _normalize_channel(self, H: np.ndarray) -> np.ndarray:
        """Normalize each channel realization to unit average per-link gain.

        The raw physical VLC/FSO gains (Lambertian path loss, FSO path loss +
        turbulence) are extremely small in absolute terms (~1e-5 or less),
        which is physically correct but numerically ill-conditions the SOR
        normal equations (A_ii = H_i^T H_i + sigma^2 becomes tiny, amplifying
        rounding error in the triangular solve). We normalize per-sample so
        that E[H_ij^2] = 1, equivalent to an automatic-gain-control (AGC)
        stage at the receiver front end; the *relative* channel structure
        (which is what the detector actually needs, together with a
        correspondingly-scaled SNR/noise) is preserved exactly.
        """
        scale = np.sqrt(np.mean(H ** 2, axis=(-1, -2), keepdims=True))
        scale = np.maximum(scale, 1e-30)
        return H / scale

    def _generate(self, n: int):
        x = self.pam.random_symbols((n, self.K), self.rng)          # (n, K)
        H = self.channel.sample(self.K, n, self.rng)                 # (n, K, K)
        H = self._normalize_channel(H)
        y_clean = np.einsum("bij,bj->bi", H, x)
        snr_db = self.rng.uniform(self.snr_lo, self.snr_hi, size=n)

        y = np.empty_like(y_clean)
        sigma2 = np.empty(n)
        # add_noise expects a scalar snr_db per call; loop in chunks by unique
        # SNR would be faster, but per-sample SNR keeps the generator simple
        # and matches "wide-domain training across SNR" in Sec. 3.5.4.
        for i in range(n):
            y_i, s2_i = self.noise_model.add_noise(y_clean[i:i+1], snr_db[i], self.rng)
            y[i] = y_i[0]
            sigma2[i] = s2_i[0]

        return {"x": x, "H": H, "y": y, "sigma2": sigma2, "snr_db": snr_db}

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self._data is None:
            raise RuntimeError("Dataset was constructed with precompute=False; "
                                "use generate_batch() for on-the-fly sampling instead.")
        d = self._data
        if torch is not None:
            return (
                torch.tensor(d["x"][idx], dtype=torch.float32),
                torch.tensor(d["H"][idx], dtype=torch.float32),
                torch.tensor(d["y"][idx], dtype=torch.float32),
                torch.tensor(d["sigma2"][idx], dtype=torch.float32),
            )
        return d["x"][idx], d["H"][idx], d["y"][idx], d["sigma2"][idx]

    def generate_batch(self, batch_size: int, fixed_snr_db: float | None = None):
        """On-the-fly batch generation, optionally at a *fixed* SNR -- used
        by the evaluation scripts to sweep SNR points for BER/BLER curves."""
        x = self.pam.random_symbols((batch_size, self.K), self.rng)
        H = self.channel.sample(self.K, batch_size, self.rng)
        H = self._normalize_channel(H)
        y_clean = np.einsum("bij,bj->bi", H, x)

        if fixed_snr_db is not None:
            snr_db = np.full(batch_size, fixed_snr_db)
        else:
            snr_db = self.rng.uniform(self.snr_lo, self.snr_hi, size=batch_size)

        y = np.empty_like(y_clean)
        sigma2 = np.empty(batch_size)
        for i in range(batch_size):
            y_i, s2_i = self.noise_model.add_noise(y_clean[i:i+1], snr_db[i], self.rng)
            y[i] = y_i[0]
            sigma2[i] = s2_i[0]

        return x, H, y, sigma2
