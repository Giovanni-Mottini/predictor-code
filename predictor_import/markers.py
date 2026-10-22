"""Explicit interpretation of source markers; never changes stored marker rows."""
from __future__ import annotations

# Versioned separately from table conversion so changing the selection rule
# requires only a descriptor refresh, not rebuilding verified Parquet tables.
MARKER_POLICY_ID = "2026-10-07-five-markers-strict-v1"


def select_marker_boundaries(
    frames: list[int | None], *, expected_marker_count: int | None,
    expected_repetitions: int | None,
) -> dict:
    """Require the original expected boundaries, with no marker-count repair.

    B&B and individual-repetition exports have no five-boundary expectation
    and are left unchanged. Invalid boundaries remain visible and unusable;
    no sorting, interpolation, or other marker-count repair is performed.
    """
    boundaries = list(frames)
    increasing = (all(frame is not None for frame in boundaries)
                  and all(a < b for a, b in zip(boundaries, boundaries[1:])))
    return {
        "policy_id": MARKER_POLICY_ID,
        "rule_applied": False,
        "temporary": False,
        "source_marker_count": len(frames),
        "effective_marker_count": len(boundaries),
        "boundary_frames": boundaries,
        "ignored_marker_indices": [],
        "ignored_marker_frames": [],
        "valid_for_expected_repetitions": (
            expected_repetitions is not None
            and expected_marker_count == expected_repetitions + 1
            and len(boundaries) == expected_marker_count and increasing
        ),
    }
