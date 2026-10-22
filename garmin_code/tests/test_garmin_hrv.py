"""Independent equations, spectral targets and meaningful discontinuity cases."""
import numpy as np
import pandas as pd
import pytest

from garmin_code.garmin import GarminConfig, normalize
from garmin_code.garmin_hrv import advanced_block, band_power, entropy, spectrum
from garmin_code import garmin_hrv_notebook as notebook
from shared_code.reader import DataContractError


def brute_entropy(x):
    x = np.asarray(x, float)
    r = .2 * np.std(x, ddof=0)
    phi, matches = [], []
    for m in (2, 3):
        v = np.array([x[i:i+m] for i in range(len(x)-m+1)])
        matrix = np.max(np.abs(v[:, None]-v[None, :]), axis=2) <= r
        phi.append(np.mean(np.log(matrix.mean(axis=1))))
        extendable = matrix[:len(x)-2, :len(x)-2]
        matches.append(int(np.triu(extendable, 1).sum()))
    b, a = matches
    return phi[0]-phi[1], -np.log(a/b) if a and b else np.nan, b, a


@pytest.mark.parametrize("x", [
    np.full(20, 800.), np.tile([800., 900., 1000.], 15),
    np.random.default_rng(291).normal(800, 80, 140),
    np.array([700, 720, 780, 850, 940, 1100, 1450.]),
])
def test_entropy_matches_independent_template_equations(x):
    expected_apen, expected_sampen, b, a = brute_entropy(x)
    actual = entropy(x)
    assert actual["ApEn"] == pytest.approx(expected_apen, abs=1e-14)
    assert actual["entropy_matches_m"] == b and actual["entropy_matches_m1"] == a
    if np.isfinite(expected_sampen):
        assert actual["SampEn"] == pytest.approx(expected_sampen, abs=1e-14)
    else:
        assert np.isnan(actual["SampEn"]) and actual["status"] != "ok"


def test_apen_is_mean_of_logs_not_old_log_of_mean_match_probability():
    x = np.random.default_rng(76).integers(700, 1100, 60).astype(float)
    r = .2*np.std(x)
    means = []
    for m in (2, 3):
        v = np.array([x[i:i+m] for i in range(len(x)-m+1)])
        means.append((np.max(abs(v[:, None]-v[None, :]), axis=2) <= r).mean())
    assert abs(entropy(x)["ApEn"] + np.log(means[1]/means[0])) > 1e-3


def test_spectrum_recovers_known_lf_hf_variance_and_ratio():
    t = np.arange(0, 300.25, .25)
    x = 850+40*np.sin(2*np.pi*.1*t)+20*np.sin(2*np.pi*.25*t)
    result = spectrum(t, x, 300)
    assert result["LF"] == pytest.approx(800, rel=.002)
    assert result["HF"] == pytest.approx(200, rel=.002)
    assert result["LF_HF_ratio"] == pytest.approx(4, rel=.002)


def test_vlf_requires_and_recovers_separate_hour_epoch():
    t = np.arange(0, 3600.25, .25)
    result = spectrum(t, 850+30*np.sin(2*np.pi*.01*t), 3600)
    assert result["VLF"] == pytest.approx(450, rel=.002)
    with pytest.raises(ValueError):
        spectrum(t[:1201], np.full(1201, 850.), 3600)


def test_band_edges_are_integrated_and_constant_spectrum_ratio_is_null():
    assert band_power(np.arange(0, 1.1, .1), np.full(11, 10.), .04, .15) == pytest.approx(1.1)
    result = spectrum(np.arange(301.), np.full(301, 900.), 300)
    assert result["LF"] == result["HF"] == 0 and np.isnan(result["LF_HF_ratio"])


def selected(x, *, rows=None, timestamps=None, gap=None):
    n = len(x)
    t = np.r_[0., np.cumsum(x[1:])/1000] if timestamps is None else timestamps
    data = pd.DataFrame({
        "participant_key": "SANI/EXAMPLE", "source_id": "source", "source_table": "bbi-(raw).csv",
        "source_row": np.arange(n) if rows is None else rows,
        "_utc": pd.Timestamp("2030-05-01")+pd.to_timedelta(t, unit="s"),
    })
    if gap is not None:
        data["gap_duration_ms"] = gap
    dt = np.diff(data._utc.to_numpy(dtype="datetime64[ns]").astype(np.int64))/1e9
    adj = (np.diff(data.source_row)==1)&(dt>0)&(dt<=5)
    return data, adj


def test_excluded_source_beat_splits_epochs_no_implicit_join():
    x = np.full(600, 1000.)
    rows = np.arange(600)
    rows[300:] += 1
    data, adj = selected(x, rows=rows)
    result, epochs = advanced_block(data, x, adj, GarminConfig())
    assert result["hrv_segments"] == 2 and result["hrv_spectral_epochs"] == 0
    assert np.isnan(result["LF"]) and not epochs
    assert result["hrv_continuous_pairs"] == 598


@pytest.mark.parametrize("gap", [1., -1., "invalid"])
def test_populated_gap_field_breaks_at_positive_or_invalid_entry(gap):
    x = np.full(600, 1000.)
    gaps = np.zeros(600, dtype=object)
    gaps[300] = gap
    data, adj = selected(x, gap=gaps)
    result, _ = advanced_block(data, x, adj, GarminConfig())
    assert result["hrv_segments"] == 2 and result["hrv_spectral_epochs"] == 0


def test_raw_export_blank_gap_information_does_not_exclude_all_beats():
    x = np.full(1001, 1000.)
    data, adj = selected(x, gap=np.full(1001, np.nan))
    result, epochs = advanced_block(data, x, adj, GarminConfig())
    assert not result["hrv_declared_gap_information"]
    assert result["hrv_spectral_epochs"] == 3 and result["hrv_entropy_epochs"] == 1
    assert result["ApEn"] == result["SampEn"] == 0
    assert result["VLF_status"] == "no_complete_epoch"
    assert len(epochs) == 4


def test_entropy_keeps_all_beats_and_never_spans_segments():
    x = 850+np.random.default_rng(59).normal(0, 30, 2000)
    rows = np.arange(2000)
    rows[1100:] += 1
    data, adj = selected(x, rows=rows)
    result, evidence = advanced_block(data, x, adj, GarminConfig())
    ent = [r for r in evidence if r["kind"] == "entropy"]
    assert len(ent) == 1 and ent[0]["n_beats"] == 1000
    expected = brute_entropy(x[:1000])
    assert result["ApEn"] == pytest.approx(expected[0])
    assert result["SampEn"] == pytest.approx(expected[1])
    assert ent[0]["source_row_start"] == 0 and ent[0]["source_row_end_inclusive"] == 999


def test_poincare_rotates_only_valid_pairs_and_flags_clock_inconsistency():
    x = np.array([800., 900., 1100., 1300., 700., 750.])
    rows = np.array([0, 1, 2, 4, 5, 6])
    data, adj = selected(x, rows=rows)
    result, _ = advanced_block(data, x, adj, GarminConfig())
    u, v = x[:-1][adj], x[1:][adj]
    assert result["SD1"] == pytest.approx(np.std((v-u)/np.sqrt(2), ddof=1))
    assert result["SD2"] == pytest.approx(np.std((v+u)/np.sqrt(2), ddof=1))
    assert not result["SD1_review"]
    data.loc[3:, "_utc"] += pd.Timedelta(milliseconds=100)
    data.loc[1:, "_utc"] += pd.Timedelta(milliseconds=100)
    result, _ = advanced_block(data, x, adj, GarminConfig())
    assert not result["SD1_review"] and result["hrv_clock_mismatch_pairs"] == 1


def test_raw_clock_residual_is_diagnostic_without_review_rejection():
    x = np.random.default_rng(40).normal(900, 25, 1050)
    data, adj = selected(x)
    data.loc[100:, "_utc"] += pd.Timedelta(seconds=.1)
    result, evidence = advanced_block(data, x, adj, GarminConfig())
    assert result["hrv_clock_mismatch_pairs"] > 0
    assert not result["ApEn_review"] and not result["LF_review"]
    assert all(not row["review"] for row in evidence)


def test_blank_gap_markers_between_declared_gaps_preserve_other_pairs():
    x = np.full(1001, 1000.)
    gaps = np.full(1001, np.nan)
    gaps[500] = 1000.
    data, adj = selected(x, gap=gaps)
    result, _ = advanced_block(data, x, adj, GarminConfig())
    assert result["hrv_segments"] == 2
    assert result["hrv_continuous_pairs"] == 999


@pytest.mark.parametrize("config", [
    {"hrv_entropy_beats": 300}, {"hrv_entropy_beats": 10001},
    {"hrv_entropy_beats": 1000.0}, {"hrv_clock_tolerance_ms": float("nan")},
    {"hrv_clock_tolerance_ms": 0},
])
def test_hrv_configuration_limits_and_semantic_hash(config):
    with pytest.raises(DataContractError):
        GarminConfig(**config)


def test_hrv_settings_change_method_identity():
    assert GarminConfig().method()["method_sha256"] != GarminConfig(hrv_entropy_beats=1200).method()["method_sha256"]
    assert GarminConfig().method()["method_sha256"] != GarminConfig(hrv_clock_tolerance_ms=100).method()["method_sha256"]


def notebook_sampen(bbi_values, m=2, r_multiplier=0.2, max_points=300):
    """Literal copy of calculate_sample_entropy from data_wearable_01-aggregation.ipynb."""
    n = len(bbi_values)
    if n < 50:
        return np.nan
    if n > max_points:
        indices = np.linspace(0, n-1, max_points, dtype=int)
        bbi_values = bbi_values[indices]
        n = max_points
    r = r_multiplier * np.std(bbi_values, ddof=1)

    def _phi(m):
        N = n - m
        if N <= 0:
            return 0
        templates = np.array([bbi_values[i:i+m] for i in range(N)])
        match_count = 0
        for i in range(N):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            match_count += np.sum(dists <= r) - 1
        return match_count / N if N > 0 else 0

    phi_m, phi_m1 = _phi(m), _phi(m + 1)
    if phi_m == 0 or phi_m1 == 0:
        return np.nan
    return -np.log(phi_m1 / phi_m)


def notebook_lf_hf(bbi_values):
    """Literal copy of calculate_frequency_features (LF/HF part), notebook cell 34."""
    from scipy import signal
    from scipy.interpolate import CubicSpline
    time_s = np.insert(np.cumsum(bbi_values) / 1000, 0, 0)
    time_interp = np.arange(0, time_s[-1], 1/4.0)
    bbi_interp = CubicSpline(time_s[:-1], bbi_values)(time_interp)
    freqs, psd = signal.welch(bbi_interp, fs=4.0, nperseg=min(256, len(bbi_interp)), scaling='density', detrend='linear')
    band = lambda lo, hi: np.trapezoid(psd[(freqs >= lo) & (freqs < hi)], freqs[(freqs >= lo) & (freqs < hi)])
    return band(0.04, 0.15) / band(0.15, 0.4)


@pytest.mark.parametrize("n", [60, 299, 2500])
def test_notebook_port_matches_the_notebook_code(n):
    x = np.random.default_rng(n).normal(850, 60, n)
    assert notebook.sample_entropy(x) == pytest.approx(notebook_sampen(x), rel=1e-12)
    assert notebook.frequency(x)["LF_HF_ratio"] == pytest.approx(notebook_lf_hf(x), rel=1e-12)
    assert np.isnan(notebook.sample_entropy(x[:49]))
    sdnn, sdsd = np.std(x, ddof=1), np.std(np.diff(x), ddof=1)
    assert notebook.block(x)["SD2_SD1_ratio"] == pytest.approx(np.sqrt(2*sdnn**2 - 0.5*sdsd**2) / (sdsd/np.sqrt(2)))
