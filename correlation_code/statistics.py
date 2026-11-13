"""Correlation of each feature with each clinical scale, at two levels.

between   Spearman correlation of patient means: one point per patient (severity).
          Needs at least 4 patients and scale means that differ.
within    repeated-measures correlation (Bakdash & Marusich 2017): Pearson r of the values
          centred on each patient's own mean, patients with at least 2 visits; df = visits -
          patients - 1 (change). p from within-patient shuffles of the scale visits: the shuffles
          are drawn once per visit table, before any test, and shared by every feature and scale.
          Also the t-test p, a 95% CI and the range of r when one patient at a time is left out.
Holm and Benjamini-Hochberg over all features x scales of one table, separately for each level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

SEED, PERMUTATIONS = 20261112, 5000


def centred(x: pd.Series, y: pd.Series, g: pd.Series):
    ok = x.notna() & y.notna()
    x, y, g = x[ok], y[ok], g[ok]
    keep = g.map(g.value_counts()) >= 2
    x, y, g = x[keep], y[keep], g[keep]
    return x - x.groupby(g).transform("mean"), y - y.groupby(g).transform("mean"), g


def rmcorr(x: pd.Series, y: pd.Series, g: pd.Series) -> dict:
    xc, yc, g = centred(x, y, g)
    n, k = len(xc), g.nunique()
    df = n - k - 1
    den = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    if df < 2 or den == 0:
        return dict(r=np.nan, df=df, p=np.nan, lo=np.nan, hi=np.nan, n=n, k=k)
    r = (xc * yc).sum() / den
    t = r * np.sqrt(df / max(1 - r * r, 1e-12))
    p = 2 * stats.t.sf(abs(t), df)
    z, se = np.arctanh(r), 1 / np.sqrt(df - 1)
    return dict(r=r, df=df, p=p, lo=np.tanh(z - 1.96 * se), hi=np.tanh(z + 1.96 * se), n=n, k=k)


def rm_r(x: np.ndarray, y: np.ndarray, g: np.ndarray) -> float:
    return rmcorr(pd.Series(x), pd.Series(y), pd.Series(g))["r"]


def between(x: pd.Series, y: pd.Series, g: pd.Series) -> dict:
    ok = x.notna() & y.notna()
    means = pd.DataFrame({"x": x[ok], "y": y[ok], "g": g[ok]}).groupby("g").mean()
    if len(means) < 4 or means["x"].nunique() < 2:
        return dict(r=np.nan, p=np.nan, n=len(means))
    r, p = stats.spearmanr(means["x"], means["y"])
    return dict(r=r, p=p, n=len(means))


def shuffle_within(g: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    index = np.arange(len(g))
    for patient in np.unique(g):
        members = np.where(g == patient)[0]
        index[members] = rng.permutation(members)
    return index


def moving_patients(x: np.ndarray, y: np.ndarray, g: np.ndarray) -> list:
    """Patients with at least 2 valid visits in which the scale changes."""
    out = []
    for patient in np.unique(g):
        i = (g == patient) & ~np.isnan(x) & ~np.isnan(y)
        if i.sum() >= 2 and np.ptp(x[i]) > 0:
            out.append(patient)
    return out


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order]).clip(max=1)
    out = np.empty(m)
    out[order] = adjusted
    return out


def benjamini_hochberg(p) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.minimum.accumulate((m / np.arange(m, 0, -1)) * p[order][::-1])[::-1].clip(max=1)
    out = np.empty(m)
    out[order] = adjusted
    return out


def _test(scale: str, feature: str, x: np.ndarray, y: np.ndarray, g: np.ndarray, shuffles: np.ndarray) -> dict:
    ok = ~np.isnan(x) & ~np.isnan(y)
    b = between(pd.Series(x), pd.Series(y), pd.Series(g))
    rm = rmcorr(pd.Series(x), pd.Series(y), pd.Series(g))
    null = np.array([rm_r(x[i], y, g) for i in shuffles])
    p_shuffle = (1 + (np.abs(np.nan_to_num(null)) >= abs(rm["r"])).sum()) / (len(shuffles) + 1)
    loo = np.array([rm_r(x[g != p], y[g != p], g[g != p]) for p in np.unique(g[ok])], dtype=float)
    return dict(scale=scale, feature=feature, visits=int(ok.sum()), patients=len(np.unique(g[ok])),
                patients_scale_moves=len(moving_patients(x, y, g)),
                between=b["r"], between_p=b["p"], between_patients=b["n"],
                rmcorr=rm["r"], rmcorr_lo=rm["lo"], rmcorr_hi=rm["hi"], rmcorr_df=rm["df"],
                rmcorr_p_t=rm["p"], rmcorr_p_shuffle=p_shuffle,
                rmcorr_loo_min=np.nanmin(loo) if np.isfinite(loo).any() else np.nan,
                rmcorr_loo_max=np.nanmax(loo) if np.isfinite(loo).any() else np.nan)


def correlate(visits: pd.DataFrame, features, scales, seed: int = SEED, permutations: int = PERMUTATIONS,
              jobs: int = -1) -> pd.DataFrame:
    """One row per scale and feature (scale-major), with both levels and the corrections.

    The tests run in parallel; every test reads the same shuffles, so the result does not depend on `jobs`.
    """
    from joblib import Parallel, delayed

    g = visits["patient"].to_numpy()
    rng = np.random.default_rng(seed)
    shuffles = np.array([shuffle_within(g, rng) for _ in range(permutations)]).reshape(permutations, len(g))
    rows = Parallel(n_jobs=jobs)(delayed(_test)(scale, feature, visits[scale].to_numpy(float), visits[feature].to_numpy(float),
                                                g, shuffles) for scale in scales for feature in features)
    result = pd.DataFrame(rows)
    for column, name in (("between_p", "between_p"), ("rmcorr_p_shuffle", "rmcorr_p")):
        result[f"{name}_holm"] = holm(result[column])
        result[f"{name}_bh"] = benjamini_hochberg(result[column])
    return result
