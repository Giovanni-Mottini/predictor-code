"""Contralateral hand DTW after initial trunk-heading alignment and reflection."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import numpy as np
from dtaidistance import dtw, dtw_ndim

from shared_code.reader import DataContractError
from shared_code.progress import ProgressReporter

DTW_BACKEND = "c" if dtw.dtw_cc is not None else "python"
DTW_COORDINATE_METHOD = "xsens-hand-initial-t8-heading-normalized-left-mirror-v2"
ORIENTATION_SHEET = "Segment Orientation - Quat"
ORIENTATION_COLUMNS = tuple(f"T8 q{i}" for i in range(4))
QUATERNION_NORM_TOLERANCE = 1e-3
HEADING_MIN_NORM = 1e-6
PAIR_FIELDS = ("participant_key", "session_id", "timepoint", "timepoint_kind", "protocol_day", "acquisition_date", "task")


def center_trajectory(trajectory: np.ndarray) -> np.ndarray:
    if not len(trajectory):
        return trajectory
    baseline = trajectory[0] if len(trajectory) < 10 else np.mean(trajectory[:10], axis=0)
    return np.ascontiguousarray(trajectory - baseline, dtype=np.float64)


def initial_heading_frame(quaternions: np.ndarray) -> dict:
    """Decode wxyz T8 orientations into a fixed, gravity-aligned body heading.

    Only the repetition's first ten samples (first one when shorter) are used.
    Export segment +X is forward; +Y is left and global +Z is up. Quaternion
    signs cancel in the rotation formula, so q and -q give the same heading.
    """
    quaternions = np.asarray(quaternions)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4 or not len(quaternions):
        raise DataContractError("dtw_orientation_invalid", "Expected nonempty wxyz T8 quaternion samples")
    baseline = quaternions[:10] if len(quaternions) >= 10 else quaternions[:1]
    try:
        baseline = np.asarray(baseline, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataContractError("dtw_orientation_invalid", "Initial trunk quaternion samples must be numeric") from exc
    norms = np.linalg.norm(baseline, axis=1)
    if not np.isfinite(baseline).all() or not np.isfinite(norms).all():
        raise DataContractError("dtw_orientation_nonfinite", "Initial trunk orientation contains null/nonfinite values")
    if np.any(norms <= 1e-12):
        raise DataContractError("dtw_orientation_zero_norm", "Initial trunk orientation has zero/near-zero norm")
    deviations = np.flatnonzero(np.abs(norms - 1) > QUATERNION_NORM_TOLERANCE)
    w, x, y, z = (baseline / norms[:, None]).T
    # First column of the body-to-global rotation: forward in global XY.
    forward = np.mean(np.column_stack((1 - 2 * (y*y + z*z), 2 * (x*y + w*z))), axis=0)
    length = float(np.linalg.norm(forward))
    if not math.isfinite(length) or length < HEADING_MIN_NORM:
        raise DataContractError("dtw_heading_undefined", "Initial trunk forward axis has no stable horizontal heading")
    c, s = forward / length
    matrix = [[float(c), float(-s), 0.0], [float(s), float(c), 0.0], [0.0, 0.0, 1.0]]
    return {"status": "computed", "method_id": DTW_COORDINATE_METHOD,
            "orientation_sheet": ORIENTATION_SHEET, "quaternion_columns": list(ORIENTATION_COLUMNS),
            "quaternion_order": "wxyz", "baseline_samples": len(baseline),
            "quaternion_norm_min": float(norms.min()), "quaternion_norm_max": float(norms.max()),
            "quaternions_normalized": True, "norm_deviation_count": len(deviations),
            "requires_review": bool(len(deviations)),
            "norm_deviations": [{"baseline_sample_index": int(i), "norm_before": float(norms[i]),
                                 "raw_wxyz": baseline[i].tolist()} for i in deviations],
            "heading_rad": math.atan2(float(s), float(c)),
            "global_to_body_row_matrix": matrix, "axis_order": ["forward", "left", "up"],
            "time_varying_trunk_motion_removed": False}


def align_and_mirror(trajectory: np.ndarray, frame: dict, side: str) -> np.ndarray:
    """Rotate row-vector global positions into fixed body axes; reflect left Y.

    Baseline translation is removed by hand_distance after this linear transform.
    No fitted rotation, scaling, time-varying frame, or double reflection is used.
    """
    if (side not in {"left", "right"} or frame.get("status") != "computed"
            or frame.get("method_id") != DTW_COORDINATE_METHOD):
        raise DataContractError("dtw_coordinate_frame_unavailable", "DTW requires an anatomical side and checked initial heading")
    matrix = np.asarray(frame.get("global_to_body_row_matrix"), dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-10, rtol=0)
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-10, rtol=0)
            or not math.isclose(float(np.linalg.det(matrix)), 1, abs_tol=1e-10)):
        raise DataContractError("dtw_coordinate_frame_invalid", "Heading transform must be a proper orthogonal rotation")
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3 or not len(trajectory) or not np.isfinite(trajectory).all():
        raise DataContractError("dtw_trajectory_invalid", "DTW requires finite nonempty xyz trajectories")
    aligned = trajectory @ matrix
    if side == "left":
        aligned[:, 1] *= -1
    return np.ascontiguousarray(aligned)


def hand_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Baseline-centered DTW for trajectories already in comparable coordinates."""
    if any(t.ndim != 2 or t.shape[1] != 3 or not len(t) or not np.isfinite(t).all() for t in (a, b)):
        raise DataContractError("dtw_trajectory_invalid", "DTW requires finite nonempty xyz trajectories")
    a, b = center_trajectory(a), center_trajectory(b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise DataContractError("dtw_centering_nonfinite", "Centering produced nonfinite values")
    value = float(dtw_ndim.distance(a, b, use_c=DTW_BACKEND == "c", use_pruning=False,
                                    inner_dist="squared euclidean", penalty=0, psi=0))
    if not math.isfinite(value):
        raise DataContractError("dtw_nonfinite_output", "DTW produced a nonfinite distance")
    return value


def pair_hand_dtw(results: list[dict], trajectories: dict[str, dict[int, np.ndarray]], *,
                  progress: ProgressReporter | None = None) -> list[dict]:
    """Set per-row distances; emit one provenance entry per unique source pair/rep.

    References come only from selected records. Simulation receives its own
    distance; it never overwrites the healthy bilateral reference value.
    """
    from .xsens_features import refresh_counts
    groups = defaultdict(list)
    for r in results:
        for row in r["feature_rows"]:
            row["dtw_hand"], row["feature_status"]["dtw_hand"], row["dtw_pair_id"] = None, "excluded", None
        groups[tuple(r.get(k) for k in PAIR_FIELDS)].append(r)
    pairs, cache = [], {}
    for key, group in groups.items():
        patient_group = any(r.get("limb_condition") == "paretic" for r in group)
        for result in group:
            side, condition = result.get("anatomical_side"), result.get("limb_condition")
            reason = None
            reference_condition = "paretic" if patient_group and condition == "non_paretic" else "non_paretic"
            same = [r for r in group if r.get("anatomical_side") == side and r.get("limb_condition") == condition]
            refs = [r for r in group if r.get("anatomical_side") in {"left", "right"} and r.get("anatomical_side") != side
                    and r.get("limb_condition") == reference_condition]
            if not result.get("participant_key") or not result.get("session_id") or side not in {"left", "right"} or condition not in {"non_paretic", "paretic", "simulated"}:
                reason = "dtw_pair_metadata_unresolved"
            elif len(same) != 1 or len(refs) > 1:
                reason = "dtw_pair_ambiguous"
            elif not refs:
                reason = "dtw_reference_missing"
            reference = refs[0] if reason is None else None
            if reference and reference.get("status") in {"skipped", "failed"}:
                reason = "dtw_reference_" + reference["status"]
            elif reference and (result.get("sampling_rate_hz") is None or reference.get("sampling_rate_hz") is None):
                reason = "dtw_sampling_rate_unavailable"
            elif reference and result.get("sampling_rate_hz") != reference.get("sampling_rate_hz"):
                reason = "dtw_pair_sampling_rate_mismatch"
            for row in result["feature_rows"]:
                rep = row["repetition"]
                a = trajectories.get(result["source_id"], {}).get(rep)
                b = trajectories.get(reference["source_id"], {}).get(rep) if reference else None
                issue = reason or ("dtw_trajectory_unavailable" if a is None or b is None else None)
                frame_a = result.get("dtw_coordinate_frames", {}).get(str(rep), {})
                frame_b = reference.get("dtw_coordinate_frames", {}).get(str(rep), {}) if reference else {}
                if not issue and any(frame.get("status") != "computed" for frame in (frame_a, frame_b)):
                    issue = "dtw_coordinate_frame_unavailable"
                if issue:
                    row["feature_evidence"]["dtw_hand"] = {"value": None, "reason": issue,
                        "reference_source_id": reference["source_id"] if reference else None}
                    result["flags"].append({"code": issue, "severity": "error", "repetition": rep})
                    continue
                pair_key = (*sorted((result["source_id"], reference["source_id"])), rep)
                if pair_key not in cache:
                    pair_id = "pair_" + hashlib.sha256(repr((DTW_COORDINATE_METHOD, key, pair_key)).encode()).hexdigest()[:24]
                    if progress:
                        progress.update(f"DTW pair {len(pairs) + 1}: {result['participant_key']} "
                                        f"{result['session_id']} task {result.get('task')} repetition {rep}; "
                                        f"{result['source_id']} vs {reference['source_id']} ({len(a)} x {len(b)} samples)")
                    try:
                        aligned_a = align_and_mirror(a, frame_a, result["anatomical_side"])
                        aligned_b = align_and_mirror(b, frame_b, reference["anatomical_side"])
                        value = hand_distance(aligned_a, aligned_b)
                        status = "requires_review" if len(a) < 10 or len(b) < 10 or any(
                            r["channels"]["dtw_hand"]["status"] == "requires_review" for r in (result, reference)) or any(
                            frame.get("requires_review") for frame in (frame_a, frame_b)) else "computed"
                        issue = None
                    except DataContractError as exc:
                        value, status, issue = None, "excluded", exc.code
                    evidence = {**dict(zip(PAIR_FIELDS, key)), "pair_id": pair_id, "repetition": rep,
                                "source_id_a": result["source_id"], "source_id_b": reference["source_id"],
                                "source_sha256_a": result["provenance"]["source_sha256"],
                                "source_sha256_b": reference["provenance"]["source_sha256"],
                                "window_a": {k: row[k] for k in ("start_frame", "end_frame_exclusive", "row_start", "row_end_exclusive", "n_samples")},
                                "window_b": {k: next(r for r in reference["feature_rows"] if r["repetition"] == rep)[k]
                                             for k in ("start_frame", "end_frame_exclusive", "row_start", "row_end_exclusive", "n_samples")},
                                "value": value, "status": status, "reason": issue, "backend": DTW_BACKEND,
                                "coordinate_method_id": DTW_COORDINATE_METHOD,
                                "coordinate_frame_a": frame_a, "coordinate_frame_b": frame_b,
                                "anatomical_side_a": result["anatomical_side"], "anatomical_side_b": reference["anatomical_side"],
                                "mirror_a": result["anatomical_side"] == "left", "mirror_b": reference["anatomical_side"] == "left",
                                "mirror_axis": "body_left_y", "mirror_matrix": [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
                                "centering_a": "first_sample" if len(a) < 10 else "first_10_mean",
                                "centering_b": "first_sample" if len(b) < 10 else "first_10_mean"}
                    cache[pair_key] = evidence
                    pairs.append(evidence)
                    if progress:
                        progress.update(f"DTW pair {len(pairs)} processed; status={status}")
                evidence = cache[pair_key]
                row["dtw_hand"], row["feature_status"]["dtw_hand"], row["dtw_pair_id"] = evidence["value"], evidence["status"], evidence["pair_id"]
                row["feature_evidence"]["dtw_hand"] = {"value": evidence["value"], "pair_id": evidence["pair_id"],
                                                       "reference_source_id": reference["source_id"],
                                                       "coordinate_method_id": DTW_COORDINATE_METHOD}
                if evidence["status"] != "computed":
                    result["flags"].append({"code": evidence["reason"] or "dtw_short_or_review_trajectory", "severity": "error" if evidence["value"] is None else "warning", "repetition": rep})
            refresh_counts(result)
    return pairs
