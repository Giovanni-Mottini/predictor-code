"""Geometric and integration checks for contralateral DTW coordinates."""
import json

import numpy as np
import pytest

from shared_code.io import atomic_json
from shared_code.reader import DataContractError, SubjectReader
from xsens_code.tests.test_xsens_features import full_subject, independent_dtw, pair_result
from xsens_code.tests.test_xsens_reader import opened, replace_column, rewrite_table
from xsens_code.xsens_cli import extract_subject
from xsens_code.xsens_dtw import (DTW_COORDINATE_METHOD, ORIENTATION_SHEET, align_and_mirror,
                                 hand_distance, initial_heading_frame, pair_hand_dtw)
from xsens_code.xsens_features import XsensConfig


def quaternion_heading(angle, count=40):
    return np.tile([np.cos(angle / 2), 0, 0, np.sin(angle / 2)], (count, 1))


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_mirrored_equivalent_paths_match_despite_independent_global_headings():
    t = np.linspace(0, 1, 40)
    right_body = np.column_stack((t, 0.2 * np.sin(np.pi * t), 0.3 * t*t))
    left_body = right_body * [1, -1, 1]
    right = right_body @ rotation(0.73).T + [4, 2, 1]
    left = left_body @ rotation(-1.12).T + [-7, 5, 1.4]
    canonical_right = align_and_mirror(right, initial_heading_frame(quaternion_heading(0.73)), "right")
    canonical_left = align_and_mirror(left, initial_heading_frame(quaternion_heading(-1.12)), "left")
    assert hand_distance(canonical_right, canonical_left) < 1e-12
    assert hand_distance(right, left) > 1
    assert hand_distance(canonical_right, canonical_left) == pytest.approx(
        independent_dtw(right_body, left_body * [1, -1, 1]), abs=1e-12)
    # Rotating around world Y would not implement the same sagittal reflection.
    assert hand_distance(right_body, left_body) > 0.1


def test_reflection_is_once_on_left_side_symmetric_and_preserves_amplitude():
    t = np.linspace(0, 1, 12)
    right_hand = np.column_stack((t, t*t, 0.5*t))
    left_hand = right_hand * [1, -1, 1]
    a, b = pair_result("right", "right"), pair_result("left", "left", "paretic")
    trajectories = {"right": {1: right_hand}, "left": {1: left_hand}}
    originals = {key: value[1].copy() for key, value in trajectories.items()}
    pairs = pair_hand_dtw([a, b], trajectories)
    assert len(pairs) == 1
    assert a["feature_rows"][0]["dtw_hand"] == b["feature_rows"][0]["dtw_hand"] == 0
    assert pairs[0]["mirror_a"] is False and pairs[0]["mirror_b"] is True
    assert pairs[0]["coordinate_method_id"] == DTW_COORDINATE_METHOD
    a, b = pair_result("right", "right"), pair_result("left", "left", "paretic")
    reverse = pair_hand_dtw([b, a], trajectories)
    assert reverse[0]["pair_id"] == pairs[0]["pair_id"] and reverse[0]["value"] == 0
    frame = initial_heading_frame(quaternion_heading(0))
    assert hand_distance(align_and_mirror(right_hand, frame, "right"),
                         align_and_mirror(2 * left_hand, frame, "left")) > 0
    assert all(np.array_equal(value[1], originals[key]) for key, value in trajectories.items())


def test_heading_uses_initial_samples_is_sign_invariant_and_keeps_later_trunk_motion():
    quaternions = quaternion_heading(0.4)
    frame = initial_heading_frame(quaternions)
    assert initial_heading_frame(-quaternions)["global_to_body_row_matrix"] == frame["global_to_body_row_matrix"]
    quaternions[10:] = quaternion_heading(1.4, len(quaternions) - 10)
    assert initial_heading_frame(quaternions) == frame
    assert frame["time_varying_trunk_motion_removed"] is False
    short = initial_heading_frame(quaternion_heading(0.4, 5))
    assert short["baseline_samples"] == 1
    unused_invalid = quaternion_heading(0.4).astype(object)
    unused_invalid[20, 0] = "unused invalid sample"
    assert initial_heading_frame(unused_invalid) == frame


@pytest.mark.parametrize("quaternion,code", [
    ([np.nan, 0, 0, 0], "dtw_orientation_nonfinite"),
    ([np.inf, 0, 0, 0], "dtw_orientation_nonfinite"),
    ([0, 0, 0, 0], "dtw_orientation_zero_norm"),
    ([1e-14, 0, 0, 0], "dtw_orientation_zero_norm"),
    ([np.sqrt(0.5), 0, np.sqrt(0.5), 0], "dtw_heading_undefined"),
])
def test_invalid_heading_does_not_fall_back_to_world_axes(quaternion, code):
    with pytest.raises(DataContractError) as error:
        initial_heading_frame(np.tile(quaternion, (12, 1)))
    assert error.value.code == code


def test_nonunit_heading_normalizes_preserves_input_and_records_original_components():
    original = quaternion_heading(.4)
    scaled = .8 * original
    before = scaled.copy()
    actual = initial_heading_frame(scaled)
    expected = initial_heading_frame(original)
    assert np.allclose(actual['global_to_body_row_matrix'], expected['global_to_body_row_matrix'], atol=1e-14)
    assert actual['requires_review'] and actual['norm_deviation_count'] == 10
    assert actual['norm_deviations'][0]['raw_wxyz'] == scaled[0].tolist()
    assert np.array_equal(scaled, before)


def test_nonunit_baseline_retains_dtw_and_reviews_only_affected_pair(full_subject):
    def change(table):
        values = table['T8 q0'].to_pylist()
        values[10] = .8
        return replace_column(table, 'T8 q0', values)
    rewrite_table(full_subject, ORIENTATION_SHEET, change)
    result = extract_subject(SubjectReader(full_subject), XsensConfig())
    assert result['computed_feature_values'] == 120 and len(result['pairs']) == 4
    for recording in result['results']:
        assert recording['feature_rows'][0]['feature_status']['dtw_hand'] == 'requires_review'
        assert all(row['feature_status']['dtw_hand'] == 'computed' for row in recording['feature_rows'][1:])


@pytest.mark.parametrize('status', ['skipped', 'failed'])
def test_unavailable_reference_reason_precedes_sampling_rate_check(status):
    a = pair_result('a', 'right', 'paretic')
    b = pair_result('b', 'left', status=status, sampling_rate_hz=None, feature_rows=[])
    assert not pair_hand_dtw([a, b], {'a': {1: np.zeros((12, 3))}})
    assert a['feature_rows'][0]['feature_evidence']['dtw_hand']['reason'] == 'dtw_reference_' + status


def test_bad_initial_orientation_excludes_only_the_affected_paired_repetition(full_subject):
    def change(table):
        values = table["T8 q0"].to_pylist()
        values[10] = None  # First baseline sample of repetition 1 only.
        return replace_column(table, "T8 q0", values)
    rewrite_table(full_subject, ORIENTATION_SHEET, change)
    result = extract_subject(SubjectReader(full_subject), XsensConfig())
    assert result["computed_feature_values"] == 118
    assert len(result["pairs"]) == 3
    for recording in result["results"]:
        assert recording["feature_rows"][0]["dtw_hand"] is None
        assert all(row["dtw_hand"] is not None for row in recording["feature_rows"][1:])
        assert all(row["computed_feature_count"] == (14 if row["repetition"] == 1 else 15) for row in recording["feature_rows"])


def test_missing_orientation_table_excludes_elbow_and_dtw_only(full_subject):
    recording = opened(full_subject)
    descriptor = json.loads(recording.descriptor_path.read_text())
    descriptor["tables"] = [t for t in descriptor["tables"] if t["source_sheet"] != ORIENTATION_SHEET]
    atomic_json(recording.descriptor_path, descriptor)
    report = extract_subject(SubjectReader(full_subject), XsensConfig())
    assert report["computed_feature_values"] == 108 and not report["pairs"]
    assert all(row["computed_feature_count"] == (13 if r["anatomical_side"] == "right" else 14) for r in report["results"] for row in r["feature_rows"])
    assert any(f["code"] == "dtw_orientation_unavailable" for r in report["results"] for f in r["channels"]["dtw_hand"]["findings"])
