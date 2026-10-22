"""First-wave notebook HRV, ported from archive/garmin_code/data_wearable_01-aggregation.ipynb.

One value per six-hour block from all of its beats in time order, exactly as the
notebook computed them: differences between every pair of rows (no gap check),
SD1/SD2 from SDNN and SDSD, sample entropy on 300 evenly spaced beats, and LF/HF
on a time axis built from the cumulative sum of the intervals. The notebook's
rounding to two decimals is not copied.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

from .garmin_hrv import ADVANCED_HRV

SOURCE = "archive/garmin_code/data_wearable_01-aggregation.ipynb (cells 32, 34, 36)"


def method():
    return {
        "source": SOURCE,
        "time_domain": "np.diff over all block beats in time order; SDNN, SDSD sample std; NN50/NN20 counts; pNN as % of differences",
        "poincare": "SD1 = SDSD/sqrt(2); SD2 = sqrt(2*SDNN^2 - 0.5*SDSD^2); ratio when SD1>0; at least 10 beats",
        "entropy": "m=2, r=0.2*sample std; >300 beats -> 300 at np.linspace positions; at least 50 beats; notebook template counts",
        "spectral": "at least 50 beats and 50 s; time = cumsum(BBI); scipy default (not-a-knot, extrapolating) cubic spline at 4 Hz; Welch nperseg=min(256,n), density, linear detrend; trapezoid over [low, high) bins",
        "bands_hz": {"VLF": [0.003, 0.04], "LF": [0.04, 0.15], "HF": [0.15, 0.4]},
    }


def time_domain(bbi):
    bbi = np.asarray(bbi, float)
    out = {"n_beats": len(bbi), "n_adjacent_pairs": max(0, len(bbi) - 1)}
    out.update({k: np.nan for k in ("SDNN", "RMSSD", "SDSD", "NN50", "pNN50", "NN20", "pNN20")})
    if len(bbi) == 0:
        return out
    diff = np.diff(bbi)
    out["SDNN"] = float(np.std(bbi, ddof=1)) if len(bbi) > 1 else np.nan
    if len(diff):
        out["RMSSD"] = float(np.sqrt(np.mean(diff ** 2)))
        out["NN50"], out["NN20"] = int((np.abs(diff) > 50).sum()), int((np.abs(diff) > 20).sum())
        out["pNN50"], out["pNN20"] = out["NN50"] * 100 / len(diff), out["NN20"] * 100 / len(diff)
    out["SDSD"] = float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan
    return out


def _sample(bbi, max_points=300):
    if len(bbi) > max_points:
        bbi = bbi[np.linspace(0, len(bbi) - 1, max_points, dtype=int)]
    return bbi


def approximate_entropy(bbi, m=2, r_multiplier=0.2, max_points=300):
    bbi = np.asarray(bbi, float)
    if len(bbi) < 50:
        return np.nan
    x = _sample(bbi, max_points)
    n, r = len(x), r_multiplier * np.std(x, ddof=1)

    def phi(m):
        templates = np.array([x[i:i + m] for i in range(n - m + 1)])
        dists = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
        return float(((dists <= r).sum(axis=1) / len(templates)).sum() / len(templates))

    a, b = phi(m), phi(m + 1)
    return np.nan if a == 0 or b == 0 else float(-np.log(b / a))


def sample_entropy(bbi, m=2, r_multiplier=0.2, max_points=300):
    bbi = np.asarray(bbi, float)
    if len(bbi) < 50:
        return np.nan
    x = _sample(bbi, max_points)
    n, r = len(x), r_multiplier * np.std(x, ddof=1)

    def phi(m):
        if n - m <= 0:
            return 0
        templates = np.array([x[i:i + m] for i in range(n - m)])
        dists = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
        return float(((dists <= r).sum(axis=1) - 1).sum() / len(templates))

    a, b = phi(m), phi(m + 1)
    return np.nan if a == 0 or b == 0 else float(-np.log(b / a))


def frequency(bbi):
    bbi = np.asarray(bbi, float)
    empty = {"VLF": np.nan, "LF": np.nan, "HF": np.nan, "LF_HF_ratio": np.nan}
    if len(bbi) < 50 or bbi.sum() / 1000 < 50:
        return empty
    time_s = np.insert(np.cumsum(bbi) / 1000, 0, 0)
    fs = 4.0
    grid = np.arange(0, time_s[-1], 1 / fs)
    sampled = CubicSpline(time_s[:-1], bbi)(grid)
    freqs, psd = signal.welch(sampled, fs=fs, nperseg=min(256, len(sampled)), scaling="density", detrend="linear")

    def power(low, high):
        idx = (freqs >= low) & (freqs < high)
        return float(np.trapezoid(psd[idx], freqs[idx])) if idx.any() else 0.0

    vlf, lf, hf = power(0.003, 0.04), power(0.04, 0.15), power(0.15, 0.4)
    return {"VLF": vlf, "LF": lf, "HF": hf, "LF_HF_ratio": lf / hf if hf > 0 else np.nan}


def block(bbi):
    """All notebook HRV fields for one six-hour block, with the shared status columns."""
    bbi = np.asarray(bbi, float)
    out = time_domain(bbi)
    out.update({name: np.nan for name in ADVANCED_HRV})
    out.update({f"{name}_review": False for name in ADVANCED_HRV})
    out.update({f"{name}_n_epochs": 0 for name in ADVANCED_HRV})
    if len(bbi) >= 10 and np.isfinite(out["SDSD"]):
        sd1 = out["SDSD"] / np.sqrt(2)
        sd2 = np.sqrt(2 * out["SDNN"] ** 2 - 0.5 * out["SDSD"] ** 2)
        out.update(SD1=sd1, SD2=sd2, SD2_SD1_ratio=sd2 / sd1 if sd1 > 0 and np.isfinite(sd2) else np.nan)
    out.update(frequency(bbi))
    out["ApEn"], out["SampEn"] = approximate_entropy(bbi), sample_entropy(bbi)
    for name in ADVANCED_HRV:
        out[f"{name}_status"] = "ok" if np.isfinite(out[name]) else "insufficient_beats"
    return out
