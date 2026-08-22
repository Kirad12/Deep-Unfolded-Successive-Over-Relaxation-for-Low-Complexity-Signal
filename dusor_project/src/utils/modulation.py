"""
M-ary Pulse Amplitude Modulation (M-PAM) utilities for IM/DD optical MIMO.

Symbols are real-valued and non-negative, as required by intensity
modulation / direct detection (Chapter 3, Sec. 3.2.1: x >= 0).
Levels are Gray-coded and normalized to unit average energy so that a
given Eb/N0 / SNR maps consistently across modulation orders.
"""

from __future__ import annotations
import numpy as np


def gray_code(n_bits: int) -> np.ndarray:
    """Return the Gray-code sequence for n_bits bits, as an array of ints."""
    codes = np.arange(1 << n_bits)
    return codes ^ (codes >> 1)


class PAM:
    """M-ary PAM constellation with non-negative, Gray-mapped levels.

    Levels are placed at {0, 1, ..., M-1} * step, then shifted/scaled to have
    zero-offset non-negative amplitudes and unit average symbol energy.
    """

    def __init__(self, M: int):
        assert M >= 2 and (M & (M - 1)) == 0, "M must be a power of 2"
        self.M = M
        self.bits_per_symbol = int(np.log2(M))

        raw_levels = np.arange(M, dtype=np.float64)          # 0, 1, ..., M-1
        avg_power = np.mean(raw_levels ** 2)
        self.levels = raw_levels / np.sqrt(avg_power)         # unit average energy, still >= 0

        gray = gray_code(self.bits_per_symbol)
        self.gray_to_index = np.argsort(gray)                 # maps gray code -> natural index
        self.index_to_gray = gray

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """bits: (..., bits_per_symbol) array of 0/1 -> real PAM symbols (...)."""
        weights = 1 << np.arange(self.bits_per_symbol)[::-1]
        gray_idx = bits.astype(int) @ weights
        nat_idx = self.gray_to_index[gray_idx]
        return self.levels[nat_idx]

    def demodulate_hard(self, symbols: np.ndarray) -> np.ndarray:
        """Nearest-level hard decision -> bit array (..., bits_per_symbol)."""
        symbols = np.clip(symbols, self.levels.min(), self.levels.max())
        nat_idx = np.argmin(
            np.abs(symbols[..., None] - self.levels[None, :]), axis=-1
        )
        gray_idx = self.index_to_gray[nat_idx]
        bits = ((gray_idx[..., None] >> np.arange(self.bits_per_symbol)[::-1]) & 1)
        return bits.astype(int)

    def nearest_symbol(self, values: np.ndarray) -> np.ndarray:
        """Snap arbitrary real values to the nearest constellation level."""
        idx = np.argmin(np.abs(values[..., None] - self.levels[None, :]), axis=-1)
        return self.levels[idx]

    def random_symbols(self, shape, rng: np.random.Generator) -> np.ndarray:
        idx = rng.integers(0, self.M, size=shape)
        return self.levels[idx]
