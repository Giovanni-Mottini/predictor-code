from predictor_import.markers import MARKER_POLICY_ID, select_marker_boundaries
import pytest


@pytest.mark.parametrize("frames, applied, boundaries, valid", [
    ([10, 20, 30, 40, 50], False, [10, 20, 30, 40, 50], True),
    ([0, 10, 20, 30, 40, 50], False, [0, 10, 20, 30, 40, 50], False),
    ([0, 10, 20, 30, 40, 50, 60], False, [0, 10, 20, 30, 40, 50, 60], False),
    ([10, 20, 30, 40], False, [10, 20, 30, 40], False),
    ([0, 10, None, 30, 40, 50], False, [0, 10, None, 30, 40, 50], False),
    ([0, 10, 20, 20, 40, 50], False, [0, 10, 20, 20, 40, 50], False),
    ([0, 20, 10, 30, 40, 50], False, [0, 20, 10, 30, 40, 50], False),
    ([None, 10, 20, 30, 40, 50], False, [None, 10, 20, 30, 40, 50], False),
    ([], False, [], False),
])
def test_boundary_selection(frames, applied, boundaries, valid):
    original = list(frames)
    result = select_marker_boundaries(frames, expected_marker_count=5, expected_repetitions=4)
    assert frames == original
    assert result["policy_id"] == MARKER_POLICY_ID
    assert result["rule_applied"] == result["temporary"] == applied
    assert result["source_marker_count"] == len(frames)
    assert result["effective_marker_count"] == len(boundaries)
    assert result["boundary_frames"] == boundaries
    assert result["ignored_marker_indices"] == ([0] if applied else [])
    assert result["ignored_marker_frames"] == ([frames[0]] if applied else [])
    assert result["valid_for_expected_repetitions"] == valid


@pytest.mark.parametrize("expected_repetitions", [None, 4])
def test_no_rule_for_bb_or_individual_repetition(expected_repetitions):
    frames = list(range(6))
    result = select_marker_boundaries(frames, expected_marker_count=None,
                                     expected_repetitions=expected_repetitions)
    assert not result["rule_applied"]
    assert not result["valid_for_expected_repetitions"]
    assert result["boundary_frames"] == frames
