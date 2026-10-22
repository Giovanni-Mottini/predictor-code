"""Continuous-segment Garmin HRV; no joining gaps or decimating beat sequences.

These are numerical descriptors of screened wearable BBI, not certified ECG NN
intervals. Epoch results and source slices are retained before block summaries.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import CubicSpline
from scipy.signal import welch
from scipy.spatial import cKDTree

ADVANCED_HRV = ("VLF", "LF", "HF", "LF_HF_ratio", "SD1", "SD2", "SD2_SD1_ratio", "ApEn", "SampEn")
SPECTRAL_SECONDS = 300
VLF_SECONDS = 3600
INTERPOLATION_HZ = 4
EPOCH_COLUMNS = (
    "participant_key", "hrv_block_start", "kind", "segment", "epoch",
    "source_id", "source_table", "source_row_start", "source_row_end_inclusive",
    "utc_start", "utc_end", "analysis_start_utc", "analysis_end_utc", "duration_s", "n_beats", "bbi_sha256",
    "clock_mismatch_pairs", "review", "status", "entropy_r_ms",
    "entropy_matches_m", "entropy_matches_m1", *ADVANCED_HRV,
)


def method():
    return {
        "segment_rule": "eligible consecutive source rows, same source ID/table, increasing UTC with delta <= max_pair_gap_s; declared gap_duration_ms>0 or invalid populated gap information breaks continuity",
        "clock_rule": "the pipeline breaks pairs whose UTC step differs from the following BBI by more than hrv_clock_tolerance_ms; hrv_clock_mismatch_pairs counts any remaining in the supplied pairs; no correction",
        "spectral_epochs": "nonoverlapping complete 300s epochs anchored at each segment's first UTC observation; interpolate actual UTC, bracket endpoints, no extrapolation or synthetic cumulative clock",
        "spectral_estimator": "natural cubic spline at 4 Hz; Welch single-epoch periodic Hann, linear detrend, density, one-sided, nfft=nperseg=1200, no overlap; integrate piecewise-linear PSD including exact band edges",
        "spectral_bands_hz": {"VLF": [0.003, 0.04], "LF": [0.04, 0.15], "HF": [0.15, 0.4]},
        "vlf": "separate nonoverlapping complete 3600s epochs, 14400-point spectrum; >10 cycles at 0.003Hz; shorter segments null, no short-term VLF substitution; stationarity is unverified",
        "poincare": "sample std of (next-current)/sqrt(2) and (next+current)/sqrt(2) on accepted continuous pairs within each block; at least two pairs; ratio null when SD1=0",
        "entropy": "nonoverlapping fixed hrv_entropy_beats contiguous beat epochs; m=2, r=0.2*population std; Chebyshev distance <=r; natural logs; no point subsampling",
        "ApEn": "mean(log(C_i^m)) - mean(log(C_i^(m+1))); all dimension-specific templates, self matches included",
        "SampEn": "-log(A/B); same N-m extendable starting positions for both dimensions, unordered pairs, self matches excluded; zero B or A produces null plus status, not infinity",
        "constant_entropy": "ApEn=SampEn=0 under inclusive <=r convention",
        "block_summary": "median of finite epoch estimates per feature; LF/HF and entropy have their own counts, VLF has separate long-epoch count; any contributing reviewed epoch marks that block feature for review; no pooling disconnected tachograms",
        "references": {
            "Task_Force_1996": "https://www.escardio.org/static-file/Escardio/Guidelines/Scientific-Statements/guidelines-Heart-Rate-Variability-FT-1996.pdf",
            "Pincus_1991": "https://doi.org/10.1073/pnas.88.6.2297",
            "Richman_Moorman_2000": "https://doi.org/10.1152/ajpheart.2000.278.6.h2039",
            "PhysioNet_SampEn": "https://physionet.org/content/sampen/1.0.0/",
            "Brennan_2001": "https://doi.org/10.1109/10.959330",
            "Welch": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html",
        },
        "units": {"VLF_LF_HF": "ms^2", "SD1_SD2": "ms", "ratios_ApEn_SampEn": "dimensionless"},
        "limits": "rest/wear screening, continuity, sample size and numerical checks do not establish ECG quality, sinus rhythm, stationarity or clinical validity; median epochs are descriptive six-hour summaries, not clinical monitoring periods",
    }


def entropy(values):
    """Exact template counts with bounded-memory trees, including match evidence."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError("Entropy requires at least four finite consecutive beats")
    r = float(0.2 * np.std(x, ddof=0))
    if r == 0:
        n = len(x) - 2
        pairs = n * (n - 1) // 2
        return {"ApEn": 0.0, "SampEn": 0.0, "entropy_r_ms": r,
                "entropy_matches_m": pairs, "entropy_matches_m1": pairs, "status": "ok"}
    phi, pair_counts = [], []
    for dim in (2, 3):
        templates = sliding_window_view(x, dim)
        counts = cKDTree(templates).query_ball_point(templates, r, p=np.inf, return_length=True)
        phi.append(float(np.log(counts / len(templates)).mean()))
        # SampEn uses the same starting positions, each with an available next beat.
        extendable = templates[:len(x)-2]
        counts = cKDTree(extendable).query_ball_point(extendable, r, p=np.inf, return_length=True)
        pair_counts.append(int((counts.sum() - len(extendable)) // 2))
    b, a = pair_counts
    return {
        "ApEn": phi[0] - phi[1],
        "SampEn": float(-np.log(a / b)) if a and b else np.nan,
        "entropy_r_ms": r, "entropy_matches_m": b, "entropy_matches_m1": a,
        "status": "ok" if a and b else "no_m_matches" if not b else "no_m1_matches",
    }


def band_power(frequencies, density, low, high):
    """Integrate exact edges rather than silently dropping boundary-bin power."""
    inside = (frequencies > low) & (frequencies < high)
    f = np.r_[low, frequencies[inside], high]
    return float(np.trapezoid(np.interp(f, frequencies, density), f))


def spectrum(times_s, bbi_ms, seconds):
    """One fully bracketed epoch: seconds is 300 or 3600, time zero is its start."""
    t, x = np.asarray(times_s, float), np.asarray(bbi_ms, float)
    if (seconds not in (SPECTRAL_SECONDS, VLF_SECONDS) or len(t) < 4
            or len(t) != len(x) or not np.isfinite(t).all()
            or not np.isfinite(x).all() or np.any(np.diff(t) <= 0)
            or t[0] > 0 or t[-1] < seconds):
        raise ValueError("Spectrum needs a finite, increasing, bracketed complete epoch")
    grid = np.arange(seconds * INTERPOLATION_HZ) / INTERPOLATION_HZ
    if np.all(x == x[0]):
        return {"VLF": 0.0} if seconds == VLF_SECONDS else {"LF": 0.0, "HF": 0.0, "LF_HF_ratio": np.nan}
    sampled = CubicSpline(t, x, bc_type="natural", extrapolate=False)(grid)
    f, power = welch(sampled, fs=INTERPOLATION_HZ, window="hann", nperseg=len(grid),
                     noverlap=0, nfft=len(grid), detrend="linear", scaling="density",
                     return_onesided=True, average="mean")
    if not np.isfinite(power).all():
        raise ValueError("Nonfinite spectral estimate")
    if seconds == VLF_SECONDS:
        return {"VLF": band_power(f, power, 0.003, 0.04)}
    lf, hf = band_power(f, power, 0.04, 0.15), band_power(f, power, 0.15, 0.4)
    return {"LF": lf, "HF": hf, "LF_HF_ratio": lf / hf if hf > 0 else np.nan}


def continuity_mask(selected, adjacent):
    """Respect declared gaps without treating an absent gap marker as a gap.

    Fitrockr fills gap duration only for gap records. A populated malformed or
    negative duration is unresolved and breaks the incoming pair.
    """
    pairs = np.asarray(adjacent, bool).copy()
    declared = False
    if "gap_duration_ms" in selected:
        raw = selected["gap_duration_ms"]
        populated = raw.notna() & raw.astype(str).str.strip().ne("")
        declared = bool(populated.any())
        gap = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
        broken = populated.to_numpy() & (~np.isfinite(gap) | (gap != 0))
        pairs &= ~broken[1:]
    if "status" in selected:
        pairs &= ~selected["status"].fillna("").astype(str).str.lower().eq("gap").to_numpy()[1:]
    if len(selected) > 1:
        pairs &= selected["source_table"].to_numpy()[1:] == selected["source_table"].to_numpy()[:-1]
    return pairs, declared


def advanced_block(selected, bbi, adjacent, config):
    """Return nine block descriptors, validity/count metadata and epoch evidence.

    selected is UTC-sorted and already screened by GarminSubject. adjacent is the
    existing source-row/gap rule. Explicit source gap information adds breaks.
    """
    x = np.asarray(bbi, float)
    n = len(x)
    if len(selected) != n or len(adjacent) != max(0, n-1):
        raise ValueError("BBI, rows and adjacency differ in length")
    result = {name: np.nan for name in ADVANCED_HRV}
    result.update({f"{name}_review": False for name in ADVANCED_HRV})
    result.update({f"{name}_n_epochs": 0 for name in ADVANCED_HRV})
    result.update({f"{name}_status": "no_complete_epoch" for name in ADVANCED_HRV})
    t = selected["_utc"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    pairs, gap_available = continuity_mask(selected, adjacent)
    mismatch = np.abs(np.diff(t) / 1e9 - x[1:] / 1000) > config.hrv_clock_tolerance_ms / 1000 + 1e-12
    result.update(
        hrv_continuous_pairs=int(pairs.sum()),
        hrv_clock_mismatch_pairs=int((mismatch & pairs).sum()),
        hrv_declared_gap_information=gap_available,
        hrv_segments=0, hrv_longest_segment_s=np.nan,
        hrv_spectral_epochs=0, hrv_vlf_epochs=0, hrv_entropy_epochs=0,
    )
    if pairs.sum() >= 2:
        first, second = x[:-1][pairs], x[1:][pairs]
        sd1 = float(np.std((second-first) / np.sqrt(2), ddof=1))
        sd2 = float(np.std((second+first) / np.sqrt(2), ddof=1))
        result.update(SD1=sd1, SD2=sd2, SD2_SD1_ratio=sd2/sd1 if sd1 > 0 else np.nan)
        for name in ("SD1", "SD2", "SD2_SD1_ratio"):
            result[f"{name}_status"] = "ok" if np.isfinite(result[name]) else "zero_SD1"
    else:
        for name in ("SD1", "SD2", "SD2_SD1_ratio"):
            result[f"{name}_status"] = "fewer_than_two_continuous_pairs"
    bounds = np.r_[0, np.flatnonzero(~pairs)+1, n] if n else np.array([0])
    evidence = []
    durations = []
    for segment, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
        durations.append((t[hi-1]-t[lo]) / 1e9)

        def record(kind, epoch, first, last, values):
            part = selected.iloc[first:last]
            clock_count = int(mismatch[first:last-1].sum())
            return {
                "participant_key": part.iloc[0]["participant_key"],
                "kind": kind, "segment": segment, "epoch": epoch,
                "source_id": part.iloc[0]["source_id"], "source_table": part.iloc[0]["source_table"],
                "source_row_start": int(part.iloc[0]["source_row"]),
                "source_row_end_inclusive": int(part.iloc[-1]["source_row"]),
                "utc_start": part.iloc[0]["_utc"], "utc_end": part.iloc[-1]["_utc"],
                "analysis_start_utc": part.iloc[0]["_utc"], "analysis_end_utc": part.iloc[-1]["_utc"],
                "duration_s": (t[last-1]-t[first]) / 1e9, "n_beats": last-first,
                "bbi_sha256": hashlib.sha256(x[first:last].astype("<f8").tobytes()).hexdigest(),
                "clock_mismatch_pairs": clock_count, "review": False, "status": "ok",
                **values,
            }

        for kind, seconds in (("spectral", SPECTRAL_SECONDS), ("vlf", VLF_SECONDS)):
            count = int((t[hi-1]-t[lo]) // (seconds * 1_000_000_000))
            for epoch in range(count):
                begin = t[lo] + epoch * seconds * 1_000_000_000
                end = begin + seconds * 1_000_000_000
                # Include immediately bracketing beats, all within this segment.
                first = lo + max(0, np.searchsorted(t[lo:hi], begin, side="right")-1)
                last = lo + np.searchsorted(t[lo:hi], end, side="left")+1
                values = spectrum((t[first:last]-begin)/1e9, x[first:last], seconds)
                values.update(analysis_start_utc=pd.Timestamp(begin), analysis_end_utc=pd.Timestamp(end))
                if kind == "spectral" and not np.isfinite(values["LF_HF_ratio"]):
                    values["status"] = "zero_HF"
                evidence.append(record(kind, epoch+1, first, last, values))
        for epoch, first in enumerate(range(lo, hi-config.hrv_entropy_beats+1, config.hrv_entropy_beats), 1):
            last = first + config.hrv_entropy_beats
            evidence.append(record("entropy", epoch, first, last, entropy(x[first:last])))
    result["hrv_segments"] = len(durations)
    result["hrv_longest_segment_s"] = max(durations) if durations else np.nan
    for kind in ("spectral", "vlf", "entropy"):
        result[f"hrv_{kind}_epochs"] = sum(row["kind"] == kind for row in evidence)
    for name in ("VLF", "LF", "HF", "LF_HF_ratio", "ApEn", "SampEn"):
        finite = [row for row in evidence if np.isfinite(row.get(name, np.nan))]
        result[f"{name}_n_epochs"] = len(finite)
        if finite:
            result[name] = float(np.median([row[name] for row in finite]))
            result[f"{name}_review"] = any(row["review"] for row in finite)
            result[f"{name}_status"] = "ok"
        elif name == "SampEn" and result["hrv_entropy_epochs"]:
            result[f"{name}_status"] = "no_template_matches"
        elif name == "LF_HF_ratio" and result["hrv_spectral_epochs"]:
            result[f"{name}_status"] = "zero_HF"
    return result, evidence
