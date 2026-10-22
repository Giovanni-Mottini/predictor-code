"""Inherited SPARC calculation with explicit rejection of degenerate spectra.

Adapted from Sivakumar Balasubramanian's SPARC scripts/smoothness.py.
https://github.com/siva82kb/SPARC (ISC; see LICENSE-SPARC.txt).
"""
from __future__ import annotations

import math
import numpy as np

from shared_code.reader import DataContractError


def spectral_arc_length(movement: np.ndarray, fs: float) -> tuple[float, dict]:
    """Return the legacy scalar and compact spectral selection evidence."""
    if len(movement) < 2:
        raise DataContractError("sparc_too_short", "SPARC requires at least two samples")
    nfft = int(2 ** np.ceil(np.log2(len(movement)) + 4))
    frequencies = np.arange(0, fs, fs / nfft)
    spectrum = np.abs(np.fft.fft(movement, nfft))
    maximum = float(np.max(spectrum))
    if not math.isfinite(maximum) or maximum <= 0:
        raise DataContractError("sparc_degenerate_spectrum", "Spectrum is zero or nonfinite")
    spectrum /= maximum
    retained = frequencies <= 10.0
    frequencies, spectrum = frequencies[retained], spectrum[retained]
    above = np.flatnonzero(spectrum >= 0.05)
    if len(above) == 0 or above[0] == above[-1]:
        raise DataContractError("sparc_degenerate_band", "Threshold-selected spectrum has fewer than two bins")
    frequencies = frequencies[above[0]:above[-1] + 1]
    spectrum = spectrum[above[0]:above[-1] + 1]
    value = -sum(np.sqrt((np.diff(frequencies) / (frequencies[-1] - frequencies[0])) ** 2
                         + np.diff(spectrum) ** 2))
    if not math.isfinite(value):
        raise DataContractError("sparc_nonfinite_output", "SPARC produced a nonfinite value")
    return float(value), {"nfft": nfft, "selected_bins": len(frequencies),
                          "first_frequency_hz": float(frequencies[0]),
                          "last_frequency_hz": float(frequencies[-1]),
                          "spectrum_max_before_normalization": maximum}
