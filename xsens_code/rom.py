"""Shoulder ZXY excursions and shared angular filtering helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
from scipy.signal import butter, filtfilt

from shared_code.reader import DataContractError
from predictor_import.markers import MARKER_POLICY_ID
from .reader import XsensRecording
from .xsens import build_windows, frame_axis, numerical_screen, selected_boundaries, source_sampling_rate

ROM_METHOD_ID = "xsens-shoulder-rom-zxy-unwrap360-butterworth4-filtfilt-v3"
ROM_QC_POLICY_ID = "xsens-rom-channel-finite-unwrap360-v2"
ANGLE_POLICY = {"policy_id": "xsens-angle-unwrap360-before-filter-v1",
                "implementation": "numpy.unwrap(period=360.0, discont=180.0)",
                "period_deg": 360.0, "discontinuity_deg": 180.0,
                "scope": "selected finite degree channels; full recording before filtering and repetition slicing",
                "assumption": "adjacent physical component changes are at most 180 degrees; shortest periodic increment",
                "source_values": "preserved; corrections affect derived arrays only",
                "approval": "professor approval reported by user on 2026-10-06",
                "euler_singularity_correction": False}
UNIT_POLICY_ID = "xsens-standard-joint-zxy-degrees-v1"
UNIT_EVIDENCE = {
    "url": "https://www.movella.com/hubfs/User_Manual_Xsens%202025.pdf",
    "title": "Xsens MVN User Manual (2025)",
    "statement": "The manufacturer documents the ZXY joint-angle vector in degrees.",
    "verified_on": "2026-10-04",
    "application": "standard Xsens Joint Angles ZXY export; external documentation, not a source-cell declaration",
}
ROM_COLUMNS = {
    "rom_shoulder_flex_ext": "Shoulder Flexion/Extension",
    "rom_shoulder_abd_add": "Shoulder Abduction/Adduction",
    "rom_shoulder_int_ext_rot": "Shoulder Internal/External Rotation",
}


@dataclass(frozen=True)
class ROMConfig:
    cutoff_hz: float = 6.0
    comparison_cutoffs_hz: tuple[float, ...] = ()

    def __post_init__(self):
        values = (self.cutoff_hz, *self.comparison_cutoffs_hz)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
            raise DataContractError("cutoff_invalid", "Cutoffs must be finite positive numbers")
        if len(values) != len(set(values)):
            raise DataContractError("cutoff_duplicate", "Primary and comparison cutoffs must be distinct")

    def method(self) -> dict:
        return {"method_id": ROM_METHOD_ID, "qc_policy_id": ROM_QC_POLICY_ID,
                "marker_policy_id": MARKER_POLICY_ID,
                "source_sheet": "Joint Angles ZXY", "representation": "ZXY", "feature_unit": "deg",
                "unit_policy_id": UNIT_POLICY_ID, "unit_evidence": UNIT_EVIDENCE,
                "angle_preprocessing": ANGLE_POLICY,
                "channels": dict(ROM_COLUMNS),
                "definition": "maximum minus minimum of the filtered component within [start_frame, end_frame)",
                "filter": {"design": "Butterworth", "design_order": 4, "cutoff_hz": float(self.cutoff_hz),
                           "sampling_rate": "source General Information Frame Rate",
                           "implementation": "scipy.signal.butter(output='ba') then scipy.signal.filtfilt",
                           "scope": "full recording before repetition slicing", "padtype": "odd", "method": "pad",
                           "padlen": "min(12, recording_sample_count - 1)", "combined_order": 8},
                "qc": {"invalid_samples": "exclude the affected full channel and all four repetitions; preserve other channels",
                       "angle_jump": "abs(adjacent raw difference) > 180 deg; apply approved periodic correction and log every event",
                       "angle_jump_is_proof_of_wrapping": False, "unwrap": True, "imputation": False,
                       "single_sample_repetition": "compute legacy zero excursion and flag for review",
                       "unit_conflict": "exclude channel; no implicit conversion",
                       "short_recording": "at least two samples; reduced padding requires review"},
                "comparison_cutoffs_hz": [float(v) for v in self.comparison_cutoffs_hz],
                "scientifically_validated": False, "ready_for_features": False}

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.method(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def filter_angle(values: np.ndarray, cutoff_hz: float, fs: float) -> np.ndarray:
    """Preserve the legacy b/a implementation and padding, using source fs."""
    if not math.isfinite(cutoff_hz) or not 0 < cutoff_hz < fs / 2:
        raise DataContractError("cutoff_above_nyquist", f"Cutoff {cutoff_hz:g} must be below Nyquist {fs / 2:g} Hz")
    if len(values) < 2:
        raise DataContractError("recording_too_short", "Filtering requires at least two samples")
    b, a = butter(4, cutoff_hz / (0.5 * fs), btype="low")
    with np.errstate(over="ignore", invalid="ignore"):
        return filtfilt(b, a, values, padlen=min(12, len(values) - 1), padtype="odd", method="pad")


def jump_findings(values: np.ndarray, frames: list[int], column: str, windows) -> list[dict]:
    with np.errstate(over="ignore", invalid="ignore"):
        jumps = np.flatnonzero(np.abs(np.diff(values)) > 180.0) + 1
    if not len(jumps):
        return []
    first = int(jumps[0])
    with np.errstate(over="ignore", invalid="ignore"):
        jump_repr = repr(float(values[first] - values[first - 1]))
    return [{"code": "angle_jump_requires_review", "severity": "warning", "column": column,
             "count": len(jumps), "threshold_deg": 180.0, "comparison": "strictly greater than",
             "previous_frame": frames[first - 1], "first_frame": frames[first],
             "first_row_index": first, "first_source_excel_row": first + 2,
             "previous_value_deg": float(values[first - 1]), "value_deg": float(values[first]),
             "jump_value_repr": jump_repr,
             "repetition_counts": {str(w.repetition): int(np.sum((jumps >= w.row_start) & (jumps < w.row_end_exclusive))) for w in windows},
             "treatment": "preserved; provisional ROM for the full channel requires review; no unwrapping"}]


def unwrap_angle(values: np.ndarray, frames: list[int], column: str, windows) -> tuple[np.ndarray, list[dict]]:
    """Unwrap only screened finite degree traces; keep continuous traces exact.

    This changes the periodic representation, not the source or Euler convention.
    The first sample anchors the full trace, including samples outside markers.
    """
    findings = jump_findings(values, frames, column, windows)
    if not findings:
        return values, []
    with np.errstate(over="ignore", invalid="ignore"):
        corrected = np.unwrap(values, period=360.0, discont=180.0)
        offsets = corrected - values
    if not np.isfinite(corrected).all() or not np.isfinite(offsets).all():
        return corrected, [{"code": "angle_unwrap_nonfinite_output", "severity": "error", "column": column,
                            "treatment": "affected full channel excluded before filtering"}]
    jumps = np.flatnonzero(np.abs(np.diff(values)) > 180.0) + 1
    finding = findings[0]
    finding.update(code="angle_wrap_corrected", severity="info", policy_id=ANGLE_POLICY["policy_id"],
                   period_deg=360.0, corrected_sample_count=int(np.count_nonzero(offsets)),
                   correction_events=[{"row_index": int(i), "source_excel_row": int(i) + 2,
                                       "previous_frame": frames[i - 1], "frame": frames[i],
                                       "raw_previous_deg": float(values[i - 1]), "raw_deg": float(values[i]),
                                       "unwrapped_previous_deg": float(corrected[i - 1]), "unwrapped_deg": float(corrected[i]),
                                       "offset_deg": float(offsets[i]), "offset_change_deg": float(offsets[i] - offsets[i - 1])}
                                      for i in jumps],
                   treatment="approved 360-degree unwrapping before full-record filtering; raw values preserved")
    return corrected, findings


def range_evidence(filtered: np.ndarray, window, frames: list[int]) -> dict:
    segment = filtered[window.row_start:window.row_end_exclusive]
    low = window.row_start + int(np.argmin(segment))
    high = window.row_start + int(np.argmax(segment))
    with np.errstate(over="ignore", invalid="ignore"):
        excursion = float(filtered[high] - filtered[low])
    return {"value": excursion if math.isfinite(excursion) else None,
            "min_deg": float(filtered[low]), "max_deg": float(filtered[high]),
            "min_frame": frames[low], "max_frame": frames[high],
            "min_row_index": low, "max_row_index": high}


def extract_rom(recording: XsensRecording, config: ROMConfig = ROMConfig()) -> dict:
    record = recording.record
    if record.get("task") not in {1, 2, 3, 4} or record.get("representation") != "full_recording":
        raise DataContractError("unsupported_activity", "ROM method covers full task1-task4 only")
    side = record.get("anatomical_side")
    if side not in {"left", "right"}:
        raise DataContractError("anatomical_side_unresolved", "ROM needs an explicit canonical anatomical side")
    fs = source_sampling_rate(recording)
    for cutoff in (config.cutoff_hz, *config.comparison_cutoffs_hz):
        if cutoff >= fs / 2:
            raise DataContractError("cutoff_above_nyquist", f"Requested cutoff {cutoff:g} is not below Nyquist {fs / 2:g} Hz")
    selection = selected_boundaries(recording)
    table = recording.read_table("Joint Angles ZXY")
    frames = frame_axis(table, "Joint Angles ZXY")
    if len(frames) < 2:
        raise DataContractError("recording_too_short", "Filtering requires at least two samples")
    windows = build_windows(frames, selection["boundary_frames"], fs)
    flags = [{"code": code, "severity": "warning", "origin": "formatter"} for code in recording.descriptor.get("warnings", [])]
    channels = {}
    columns = {c["name"]: c for c in recording.table_descriptor("Joint Angles ZXY")["columns"]}
    filtered_by_feature, candidate_by_feature, wrapped_baselines = {}, {}, {}
    for feature, suffix in ROM_COLUMNS.items():
        column = f"{side.title()} {suffix}"
        source_unit = columns.get(column, {}).get("unit")
        channel = {"source_column": column, "source_unit": source_unit, "effective_unit": "deg",
                   "unit_basis": "source_declaration" if source_unit is not None else "manufacturer_documented_standard_export",
                   "unit_policy_id": UNIT_POLICY_ID, "status": "computed", "findings": []}
        channels[feature] = channel
        if column not in table.column_names:
            channel["status"] = "excluded"
            channel["findings"].append({"code": "rom_column_missing", "severity": "error", "column": column,
                                        "treatment": "all repetitions of this feature excluded"})
            continue
        if source_unit is not None and str(source_unit).strip().casefold() not in {"deg", "degree", "degrees", "°"}:
            channel["status"] = "excluded"
            channel["effective_unit"] = None
            channel["findings"].append({"code": "angle_unit_conflict", "severity": "error", "column": column,
                                        "source_unit": source_unit, "treatment": "excluded; no conversion"})
            continue
        if source_unit is None:
            channel["findings"].append({"code": "angle_unit_from_documentation", "severity": "info", "column": column,
                                        "unit_policy_id": UNIT_POLICY_ID, "evidence_url": UNIT_EVIDENCE["url"]})
        selected = table.select(["Frame", column])
        findings = numerical_screen(selected, windows, "Joint Angles ZXY")
        if findings:
            channel["status"] = "excluded"
            for finding in findings:
                finding.update({"severity": "error", "treatment": "affected full channel excluded before filtering; all four features null"})
            channel["findings"].extend(findings)
            continue
        values = np.asarray(table[column].to_pylist(), dtype=np.float64)
        raw = values
        values, findings = unwrap_angle(raw, frames, column, windows)
        channel["angle_preprocessing_policy_id"] = ANGLE_POLICY["policy_id"]
        channel["findings"].extend(findings)
        if any(f["severity"] == "error" for f in findings):
            channel["status"] = "excluded"
            continue
        if findings:
            baseline = filter_angle(raw, config.cutoff_hz, fs)
            if np.isfinite(baseline).all():
                wrapped_baselines[feature] = baseline
        if len(frames) <= 12:
            channel["status"] = "requires_review"
        if len(frames) <= 12:
            channel["findings"].append({"code": "filter_padding_reduced", "severity": "warning", "column": column,
                                        "sample_count": len(frames), "padlen": len(frames) - 1})
        filtered = filter_angle(values, config.cutoff_hz, fs)
        if not np.isfinite(filtered).all():
            channel["status"] = "excluded"
            channel["findings"].append({"code": "filter_nonfinite_output", "severity": "error", "column": column})
            continue
        filtered_by_feature[feature] = filtered
        candidate_by_feature[feature] = {}
        for cutoff in config.comparison_cutoffs_hz:
            candidate = filter_angle(values, cutoff, fs)
            if not np.isfinite(candidate).all():
                channel["findings"].append({"code": "comparison_filter_nonfinite_output", "severity": "error", "column": column,
                                            "cutoff_hz": cutoff, "treatment": "comparison excluded; reference feature retained"})
            candidate_by_feature[feature][float(cutoff)] = candidate
    rows, comparisons = [], []
    for window in windows:
        row = {**asdict(window), "sampling_rate_hz": fs, "feature_unit": "deg", "method_id": ROM_METHOD_ID,
               "method_sha256": config.fingerprint(), "marker_policy_id": selection["policy_id"],
               "temporary_marker_rule": selection["rule_applied"], "feature_evidence": {}, "feature_status": {}}
        for feature, channel in channels.items():
            evidence = range_evidence(filtered_by_feature[feature], window, frames) if feature in filtered_by_feature else {"value": None}
            if feature in wrapped_baselines and evidence["value"] is not None:
                baseline = range_evidence(wrapped_baselines[feature], window, frames)
                evidence.update(before_unwrap=baseline, unwrap_delta_deg=evidence["value"] - baseline["value"]
                                if baseline["value"] is not None else None)
            value = evidence["value"]
            status = channel["status"]
            if value is None:
                status = "excluded"
                if feature in filtered_by_feature:
                    channel["findings"].append({"code": "rom_nonfinite_range", "severity": "error", "column": channel["source_column"], "repetition": window.repetition})
            elif window.n_samples == 1:
                status = "requires_review"
            row[feature], row["feature_evidence"][feature], row["feature_status"][feature] = value, evidence, status
            for cutoff in config.comparison_cutoffs_hz:
                candidate = candidate_by_feature.get(feature, {}).get(float(cutoff))
                candidate_ok = candidate is not None and np.isfinite(candidate).all()
                alternate = range_evidence(candidate, window, frames)["value"] if candidate_ok else None
                delta = alternate - value if alternate is not None and value is not None else None
                percent = delta / value * 100 if delta is not None and value != 0 else None
                comparisons.append({"repetition": window.repetition, "feature": feature, "baseline_cutoff_hz": config.cutoff_hz,
                                    "comparison_cutoff_hz": cutoff, "baseline_deg": value, "comparison_deg": alternate,
                                    "delta_deg": delta, "absolute_delta_deg": abs(delta) if delta is not None else None,
                                    "relative_change_percent": percent if percent is not None and math.isfinite(percent) else None,
                                    "feature_status": status, "included_in_summary": status == "computed" and alternate is not None})
        if window.n_samples == 1:
            flags.append({"code": "repetition_single_sample", "severity": "warning", "repetition": window.repetition,
                          "treatment": "zero excursion retained for legacy comparison; all computed features require review"})
        row["computed_feature_count"] = sum(v is not None for v in (row[f] for f in ROM_COLUMNS))
        row["qc_pass_feature_count"] = sum(status == "computed" for status in row["feature_status"].values())
        row["excluded_features"] = [f for f, status in row["feature_status"].items() if status == "excluded"]
        row["review_features"] = [f for f, status in row["feature_status"].items() if status == "requires_review"]
        rows.append(row)
    computed = sum(r["computed_feature_count"] for r in rows)
    return {"status": "computed" if computed == len(ROM_COLUMNS) * len(rows) else "partial" if computed else "excluded",
            "sampling_rate_hz": fs, "frame_axis": {"first_frame": frames[0], "last_frame": frames[-1], "row_count": len(frames),
                                                   "frame_equals_row_index": all(frame == row for row, frame in enumerate(frames))},
            "marker_selection": selection, "channels": channels, "flags": flags,
            "feature_rows": rows, "comparisons": comparisons, "computed_feature_values": computed,
            "qc_pass_feature_values": sum(r["qc_pass_feature_count"] for r in rows),
            "ready_for_features": False, "provenance": recording.provenance()}
