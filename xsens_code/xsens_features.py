"""Fifteen Xsens features using checked data and versioned processing."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from shared_code.reader import DataContractError
from .reader import XsensRecording
from .rom import ANGLE_POLICY, ROM_COLUMNS, ROMConfig, UNIT_EVIDENCE, extract_rom, filter_angle, unwrap_angle, range_evidence
from .smoothness import spectral_arc_length
from .xsens import SIGNAL_SHEETS, build_windows, frame_axis, numerical_screen
from .processing_span import PROCESSING_SPAN_POLICY, extract_on_finite_span
from .elbow_rom import ELBOW_FEATURE, SPEC as ELBOW_SPEC, extract_elbow_bending

METHOD_ID = "xsens-15-features-quaternion-elbow-finite-span-vector-speed-v3"
SEGMENTS = {"trunk": "T8", "shoulder": "Shoulder", "upper_arm": "Upper Arm", "forearm": "Forearm"}
COMP_COLUMNS = {"comp_trunk_lateral_bending": "Vertical_T8 Lateral Bending",
                "comp_trunk_flex_ext": "Vertical_T8 Flexion/Extension"}
FEATURE_UNITS = {**{f"vel_mean_{role}": "m/s" for role in SEGMENTS},
                 **{f"sparc_{role}": "dimensionless" for role in SEGMENTS},
                 **{f: "deg" for f in (*ROM_COLUMNS, ELBOW_FEATURE, *COMP_COLUMNS)}, "dtw_hand": "m (unnormalized accumulated distance)"}
FEATURE_NAMES = tuple(FEATURE_UNITS)
UNIT_POLICY_ID = "xsens-standard-kinematics-units-v1"
KINEMATIC_UNIT_EVIDENCE = {**UNIT_EVIDENCE,
    "statement": "Manufacturer segment kinematics use global xyz position in metres, velocity in metres/second and ZXY joint angles in degrees.",
    "location": "PDF page 264, printed Analyze manual page 95; ergonomic-angle definition on PDF page 263",
    "application": "explicit assumption for the named standard-export columns; descriptors remain unchanged"}


@dataclass(frozen=True)
class XsensConfig:
    cutoff_hz: float = 6.0
    sparc_cutoff_hz: float = 15.0

    def __post_init__(self):
        ROMConfig(self.cutoff_hz)
        ROMConfig(self.sparc_cutoff_hz)

    def method(self) -> dict:
        from .xsens_dtw import DTW_BACKEND, DTW_COORDINATE_METHOD, ORIENTATION_SHEET, ORIENTATION_COLUMNS, QUATERNION_NORM_TOLERANCE, HEADING_MIN_NORM
        return {"method_id": METHOD_ID, "feature_units": FEATURE_UNITS,
                "rom": ROMConfig(self.cutoff_hz).method(),
                "angle_preprocessing": ANGLE_POLICY,
                "angle_preprocessing_scope": "Three shoulder and two trunk Euler channels; no elbow unwrap",
                "elbow_bending": dict(ELBOW_SPEC),
                "feature_definitions": {ELBOW_FEATURE: "Unsigned 3D segment-bending max-minus-min ROM"},
                "segments": SEGMENTS, "compensation_columns": COMP_COLUMNS,
                "unit_policy_id": UNIT_POLICY_ID, "unit_evidence": KINEMATIC_UNIT_EVIDENCE,
                "speed": {"construction": "filter raw xyz separately, then sqrt(sum(filtered xyz squared))",
                          "mean_filter_cutoff_hz": self.cutoff_hz, "definition": "arithmetic mean of nonnegative vector magnitude"},
                "sparc": {"filter_cutoff_hz": self.sparc_cutoff_hz, "padlevel": 4, "fc_hz": 10.0, "amp_threshold": 0.05,
                          "selection": "inclusive span from first to last >= threshold bin within f <= fc",
                          "source": "https://github.com/siva82kb/SPARC/blob/7deff21add7e3b6403869c8932dff31bceacb472/scripts/smoothness.py",
                          "degenerate_spectrum": "null with explicit QC; no fabricated zero"},
                "filter": {**ROMConfig(self.cutoff_hz).method()["filter"],
                           "applies_to": "each velocity xyz axis before speed norm, each hand xyz axis, and each selected angle",
                           "processing_span_policy": PROCESSING_SPAN_POLICY},
                "dtw": {"backend": DTW_BACKEND, "library": "dtaidistance.dtw_ndim.distance",
                        "local_cost": "dependent 3D squared Euclidean", "result": "sqrt(accumulated minimum cost)",
                        "centering": "subtract mean of first 10 samples of each repetition; first sample when N < 10",
                        "position_filter_cutoff_hz": self.cutoff_hz, "window": None, "penalty": 0, "psi": 0,
                        "pruning": False, "mirroring": True, "resampling": False, "path_length_normalization": False,
                        "coordinate_method_id": DTW_COORDINATE_METHOD,
                        "coordinate_frame": "fixed initial T8 heading per repetition; global up retained",
                        "orientation_sheet": ORIENTATION_SHEET, "quaternion_columns": list(ORIENTATION_COLUMNS), "quaternion_order": "wxyz",
                        "heading_baseline": "mean horizontal forward vector of first 10 T8 quaternions; first sample when N < 10",
                        "quaternion_norm_tolerance": QUATERNION_NORM_TOLERANCE, "heading_min_horizontal_norm": HEADING_MIN_NORM,
                        "quaternion_normalization": "finite norms > 1e-12; deviations retained with review provenance",
                        "reflection": "left hand only: (forward,left,up) -> (forward,-left,up); right hand unchanged",
                        "trunk_motion": "fixed heading; no time-varying trunk subtraction or rotation",
                        "missing_orientation": "exclude DTW only; no global-axis fallback",
                        "coordinate_evidence": {"url": UNIT_EVIDENCE["url"], "location": "PDF pages 259-260 and 264: global/segment frames and wxyz segment orientation"},
                        "matching": "exact cohort-qualified subject/session/timepoint/day/date/task and repetition; opposite anatomical side",
                        "reference": "paretic vs non_paretic; otherwise bilateral non_paretic; simulated to opposite non_paretic only",
                        "ambiguity": "exclude rather than select among duplicate recordings", "sample_rate_mismatch": "exclude"},
                "qc": {"missing_nonfinite_or_unit_conflict": "exclude affected full channel only, before filtering",
                       "angle_jumps": "approved unwrap360 before filtering for joint ROM and trunk compensation; correction events retained",
                       "filtered_negative_speed": "nonnegative speed constructed after component filtering; no clipping",
                       "short_hand_window": "N < 10: retain first-sample centering with review",
                       "other_repair": False}, "scientifically_validated": False, "ready_for_features": False}

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.method(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass
class XsensExtraction:
    report: dict
    hand_trajectories: dict[int, np.ndarray]


def refresh_counts(result: dict) -> None:
    for row in result["feature_rows"]:
        row["computed_feature_count"] = sum(row[f] is not None for f in FEATURE_NAMES)
        row["qc_pass_feature_count"] = sum(row["feature_status"][f] == "computed" for f in FEATURE_NAMES)
        row["excluded_features"] = [f for f in FEATURE_NAMES if row["feature_status"][f] == "excluded"]
        row["review_features"] = [f for f in FEATURE_NAMES if row["feature_status"][f] == "requires_review"]
    result["computed_feature_values"] = sum(r["computed_feature_count"] for r in result["feature_rows"])
    result["qc_pass_feature_values"] = sum(r["qc_pass_feature_count"] for r in result["feature_rows"])
    count = result["computed_feature_values"]
    if result.get("status") not in {"failed", "skipped"}:
        result["status"] = "computed" if count == len(FEATURE_NAMES) * len(result["feature_rows"]) else "partial" if count else "excluded"


def selected_channel(recording, table, sheet, names, unit, frames, windows) -> tuple[dict, np.ndarray | None]:
    columns = {c["name"]: c for c in recording.table_descriptor(sheet)["columns"]}
    channel = {"source_sheet": sheet, "source_columns": names, "source_units": {n: columns.get(n, {}).get("unit") for n in names},
               "effective_unit": unit, "unit_policy_id": UNIT_POLICY_ID, "status": "computed", "findings": []}
    aliases = {"deg": {"deg", "degree", "degrees", "°"}, "m": {"m", "meter", "metre", "meters", "metres"},
               "m/s": {"m/s", "m s-1", "m/s^1"}}
    missing = [n for n in names if n not in table.column_names]
    conflicts = [n for n in names if columns.get(n, {}).get("unit") is not None
                 and str(columns[n]["unit"]).strip().casefold() not in aliases[unit]]
    if missing or conflicts:
        channel["status"] = "excluded"
        channel["findings"].append({"code": "feature_column_missing" if missing else "feature_unit_conflict", "severity": "error",
                                    "columns": missing or conflicts, "treatment": "affected channel excluded; no implicit conversion"})
        if conflicts:
            channel["effective_unit"] = None
        return channel, None
    if any(columns[n].get("unit") is None for n in names):
        channel["findings"].append({"code": "feature_unit_from_documentation", "severity": "info",
                                    "evidence_url": KINEMATIC_UNIT_EVIDENCE["url"], "unit_policy_id": UNIT_POLICY_ID})
    try:
        findings = numerical_screen(table.select(["Frame", *names]), windows, sheet)
    except DataContractError as exc:
        channel["status"] = "excluded"
        channel["findings"].append({"code": exc.code, "severity": "error", "message": str(exc),
                                    "treatment": "affected channel excluded"})
        return channel, None
    if findings:
        channel["status"] = "excluded"
        channel["findings"].extend({**f, "severity": "error", "treatment": "affected full channel excluded before filtering"} for f in findings)
        return channel, None
    values = np.column_stack([table[n].to_pylist() for n in names]).astype(np.float64)
    if len(frames) <= 12:
        channel["findings"].append({"code": "filter_padding_reduced", "severity": "warning", "sample_count": len(frames)})
    if any(f["severity"] == "warning" for f in channel["findings"]):
        channel["status"] = "requires_review"
    return channel, values


def checked_filter(values, channel, cutoff, fs):
    filtered = filter_angle(values, cutoff, fs)
    if not np.isfinite(filtered).all():
        channel["status"] = "excluded"
        channel["findings"].append({"code": "filter_nonfinite_output", "severity": "error", "cutoff_hz": cutoff})
        return None
    return filtered


def processing_channels(recording, *, include_elbow_quaternions=True):
    if recording.record.get("task") not in {1, 2, 3, 4} or recording.record.get("representation") != "full_recording":
        raise DataContractError("unsupported_activity", "Full-feature method covers full task1-task4 only")
    if recording.record.get("anatomical_side") not in {"left", "right"}:
        raise DataContractError("anatomical_side_unresolved", "Full-feature method requires an anatomical side")
    side = recording.record["anatomical_side"].title()
    fields = {
        "Joint Angles ZXY": [f"{side} {suffix}" for suffix in ROM_COLUMNS.values()],
        "Ergonomic Joint Angles ZXY": list(COMP_COLUMNS.values()),
        "Segment Velocity": [f"{seg if role == 'trunk' else side + ' ' + seg} {axis}"
                             for role, seg in SEGMENTS.items() for axis in "xyz"],
        "Segment Position": [f"{side} Hand {axis}" for axis in "xyz"],
        "Segment Orientation - Quat": [f"T8 q{i}" for i in range(4)],
    }
    if include_elbow_quaternions:
        fields["Segment Orientation - Quat"] += [f"{side} {seg} q{i}"
                                                 for seg in ("Upper Arm", "Forearm") for i in range(4)]
    return fields


def extract_xsens(recording: XsensRecording, config: XsensConfig = XsensConfig()) -> XsensExtraction:
    return extract_on_finite_span(recording, config, _extract_xsens, processing_channels(recording))


def _extract_xsens(recording: XsensRecording, config: XsensConfig) -> XsensExtraction:
    result = extract_rom(recording, ROMConfig(config.cutoff_hz))
    fs = result["sampling_rate_hz"]
    if config.sparc_cutoff_hz >= fs / 2 or 10.0 >= fs / 2:
        raise DataContractError("cutoff_above_nyquist", "SPARC preprocessing and spectral cutoff must be below source Nyquist")
    side = recording.record["anatomical_side"].title()
    frames = frame_axis(recording.read_table("Joint Angles ZXY", columns=["Frame"]), "Joint Angles ZXY")
    tables = {}
    for sheet in SIGNAL_SHEETS:
        if sheet == "Joint Angles ZXY":
            continue
        table = recording.read_table(sheet)
        if frame_axis(table, sheet) != frames:
            raise DataContractError("frame_axes_misaligned", f"{sheet} does not share the Joint Angles ZXY frame axis")
        tables[sheet] = table
    windows = build_windows(frames, result["marker_selection"]["boundary_frames"], fs)
    for row in result["feature_rows"]:
        row.pop("feature_unit", None)
        row.update({"method_id": METHOD_ID, "method_sha256": config.fingerprint(), "dtw_pair_id": None})
        for f in FEATURE_NAMES:
            if f not in row:
                row[f], row["feature_status"][f], row["feature_evidence"][f] = None, "excluded", {"value": None}
    for role, segment in SEGMENTS.items():
        base = segment if role == "trunk" else f"{side} {segment}"
        channel, xyz = selected_channel(recording, tables["Segment Velocity"], "Segment Velocity",
                                        [f"{base} {axis}" for axis in "xyz"], "m/s", frames, windows)
        with np.errstate(over="ignore", invalid="ignore"):
            speed = np.sqrt(np.sum(xyz ** 2, axis=1)) if xyz is not None else None
        if speed is not None and not np.isfinite(speed).all():
            channel["status"] = "excluded"
            channel["findings"].append({"code": "scalar_speed_nonfinite", "severity": "error"})
            speed = None
        for prefix, cutoff in (("vel_mean", config.cutoff_hz), ("sparc", config.sparc_cutoff_hz)):
            f = f"{prefix}_{role}"
            own = {**channel, "findings": list(channel["findings"]), "filter_cutoff_hz": cutoff}
            result["channels"][f] = own
            components = [checked_filter(xyz[:, axis], own, cutoff, fs) for axis in range(3)] if speed is not None else []
            filtered = np.linalg.norm(np.column_stack(components), axis=1) if components and all(x is not None for x in components) else None
            if filtered is not None and not np.isfinite(filtered).all():
                own["status"] = "excluded"
                own["findings"].append({"code": "scalar_speed_nonfinite", "severity": "error"})
                filtered = None
            if filtered is None:
                continue
            for row, window in zip(result["feature_rows"], windows):
                segment_values = filtered[window.row_start:window.row_end_exclusive]
                status, evidence = own["status"], {}
                try:
                    if prefix == "sparc":
                        value, evidence = spectral_arc_length(segment_values, fs)
                    else:
                        value = float(np.mean(segment_values))
                        evidence = {"filtered_min_m_s": float(segment_values.min()), "filtered_max_m_s": float(segment_values.max())}
                    if not math.isfinite(value):
                        raise DataContractError("feature_nonfinite_output", "Feature produced a nonfinite value")
                    negative = int(np.sum(segment_values < 0))
                    if negative:
                        status = "requires_review"
                        first = window.row_start + int(np.flatnonzero(segment_values < 0)[0])
                        own["findings"].append({"code": "filtered_negative_speed", "severity": "warning", "repetition": window.repetition,
                                                "count": negative, "minimum_m_s": float(segment_values.min()),
                                                "first_row_index": first, "first_source_excel_row": first + 2,
                                                "first_frame": frames[first],
                                                "treatment": "provisional inherited value retained; no clipping"})
                except DataContractError as exc:
                    value, status = None, "excluded"
                    own["findings"].append({"code": exc.code, "severity": "error", "repetition": window.repetition, "message": str(exc)})
                row[f], row["feature_status"][f], row["feature_evidence"][f] = value, status, {"value": value, **evidence}
    for f, name in COMP_COLUMNS.items():
        channel, raw = selected_channel(recording, tables["Ergonomic Joint Angles ZXY"], "Ergonomic Joint Angles ZXY",
                                        [name], "deg", frames, windows)
        result["channels"][f] = channel
        baseline = None
        if raw is not None:
            values, findings = unwrap_angle(raw[:, 0], frames, name, windows)
            channel["angle_preprocessing_policy_id"] = ANGLE_POLICY["policy_id"]
            channel["findings"].extend(findings)
            if any(issue["severity"] == "error" for issue in findings):
                channel["status"] = "excluded"
                continue
            if findings:
                candidate = filter_angle(raw[:, 0], config.cutoff_hz, fs)
                if np.isfinite(candidate).all():
                    baseline = candidate
            filtered = checked_filter(values, channel, config.cutoff_hz, fs)
        else:
            filtered = None
        if filtered is not None:
            for row, window in zip(result["feature_rows"], windows):
                evidence = range_evidence(filtered, window, frames)
                if baseline is not None and evidence["value"] is not None:
                    before = range_evidence(baseline, window, frames)
                    evidence.update(before_unwrap=before, unwrap_delta_deg=evidence["value"] - before["value"]
                                    if before["value"] is not None else None)
                value = evidence["value"]
                status = "excluded" if value is None else "requires_review" if window.n_samples == 1 else channel["status"]
                if value is None:
                    channel["findings"].append({"code": "feature_nonfinite_output", "severity": "error", "repetition": window.repetition})
                row[f], row["feature_status"][f], row["feature_evidence"][f] = value, status, evidence
    elbow_channel, elbow_rows = extract_elbow_bending(recording, frames, windows, config.cutoff_hz, fs)
    result["channels"][ELBOW_FEATURE] = elbow_channel
    for row, elbow in zip(result["feature_rows"], elbow_rows):
        row[ELBOW_FEATURE] = elbow["value"]
        row["feature_status"][ELBOW_FEATURE] = elbow["status"]
        row["feature_evidence"][ELBOW_FEATURE] = elbow["evidence"]
    hand_channel, hand = selected_channel(recording, tables["Segment Position"], "Segment Position",
                                          [f"{side} Hand {axis}" for axis in "xyz"], "m", frames, windows)
    result["channels"]["dtw_hand"] = hand_channel
    trajectories = {}
    if hand is not None:
        axes = [checked_filter(hand[:, axis], hand_channel, config.cutoff_hz, fs) for axis in range(3)]
        if all(axis is not None for axis in axes):
            filtered_hand = np.column_stack(axes)
            trajectories = {w.repetition: filtered_hand[w.row_start:w.row_end_exclusive] for w in windows}
    result["dtw_coordinate_frames"] = {}
    if trajectories:
        from .xsens_dtw import initial_heading_frame, ORIENTATION_SHEET, ORIENTATION_COLUMNS
        try:
            orientation = recording.read_table(ORIENTATION_SHEET, columns=["Frame", *ORIENTATION_COLUMNS])
            if frame_axis(orientation, ORIENTATION_SHEET) != frames:
                raise DataContractError("dtw_orientation_frames_misaligned", "Trunk orientation does not share the hand frame axis")
            descriptor = recording.table_descriptor(ORIENTATION_SHEET)
            for column in descriptor["columns"]:
                if column["name"] in ORIENTATION_COLUMNS and column.get("unit") not in {None, "1", "dimensionless", "unitless"}:
                    raise DataContractError("dtw_orientation_unit_conflict", "Trunk quaternion columns must be dimensionless")
            quaternions = np.column_stack([orientation[name].to_pylist() for name in ORIENTATION_COLUMNS])
            for window in windows:
                try:
                    frame = initial_heading_frame(quaternions[window.row_start:window.row_end_exclusive])
                    frame.update(baseline_row_start=window.row_start,
                                 baseline_row_end_exclusive=window.row_start + frame["baseline_samples"],
                                 baseline_first_frame=frames[window.row_start],
                                 baseline_last_frame=frames[window.row_start + frame["baseline_samples"] - 1])
                    if frame["requires_review"]:
                        hand_channel["findings"].append({
                            "code": "dtw_quaternion_norm_normalized_requires_review", "severity": "warning",
                            "repetition": window.repetition, "count": frame["norm_deviation_count"],
                            "events": [{"frame": frames[window.row_start + event["baseline_sample_index"]],
                                        "source_excel_row": window.row_start + event["baseline_sample_index"] + 2,
                                        **event} for event in frame["norm_deviations"]],
                            "treatment": "normalized in derived arrays; DTW retained with review",
                        })
                    result["dtw_coordinate_frames"][str(window.repetition)] = frame
                except DataContractError as exc:
                    result["dtw_coordinate_frames"][str(window.repetition)] = {"status": "excluded", "reason": exc.code}
                    hand_channel["findings"].append({"code": exc.code, "severity": "error", "repetition": window.repetition,
                                                     "message": str(exc), "treatment": "DTW repetition excluded; other features preserved"})
        except (DataContractError, OSError, ValueError, KeyError, TypeError) as exc:
            hand_channel["findings"].append({"code": "dtw_orientation_unavailable", "severity": "error",
                                             "reason": getattr(exc, "code", "orientation_read_failed"), "message": str(exc),
                                             "treatment": "DTW excluded; other features preserved; no global-axis fallback"})
    result["comparisons"] = []
    result["provenance"] = recording.provenance()
    refresh_counts(result)
    return XsensExtraction(result, trajectories)
