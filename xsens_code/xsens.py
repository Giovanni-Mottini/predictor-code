"""Repetition segmentation and limited numerical screening of preserved signals."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import pyarrow as pa
import pyarrow.compute as pc

from predictor_import.markers import select_marker_boundaries
from . import SEGMENTATION_METHOD
from shared_code.reader import DataContractError
from .reader import XsensRecording

SIGNAL_SHEETS = ("Segment Velocity", "Segment Position", "Joint Angles ZXY",
                 "Ergonomic Joint Angles ZXY")


@dataclass(frozen=True)
class RepetitionWindow:
    repetition: int
    start_frame: int
    end_frame_exclusive: int
    row_start: int
    row_end_exclusive: int
    n_samples: int
    first_sample_frame: int
    last_sample_frame: int
    duration_s: float
    sample_span_s: float

    def slice(self, table: pa.Table) -> pa.Table:
        """Slice a table whose frame axis has been checked against these windows."""
        return table.slice(self.row_start, self.n_samples)


def frame_axis(table: pa.Table, sheet: str) -> list[int]:
    if "Frame" not in table.column_names or not pa.types.is_integer(table["Frame"].type):
        raise DataContractError("frame_type_invalid", f"{sheet}: Frame must be an integer column")
    frames = table["Frame"].to_pylist()
    if not frames or any(f is None for f in frames):
        raise DataContractError("frame_missing", f"{sheet}: empty or null frame axis")
    if frames[0] < 0:
        raise DataContractError("frame_negative", f"{sheet}: negative frame origin")
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise DataContractError("frame_order_invalid", f"{sheet}: duplicate or descending frames; no sorting applied")
    if any(b - a != 1 for a, b in zip(frames, frames[1:])):
        raise DataContractError("frame_gap", f"{sheet}: missing frames; no interpolation applied")
    return frames


def build_windows(frames: list[int], boundaries: list[int], fs: float) -> list[RepetitionWindow]:
    """Map frame labels to zero-based row positions and select [start, end).

    Every boundary must name an observed frame, except the final boundary may
    equal last_frame + 1, allowing inclusion of the final recorded sample.
    Callers must supply a checked, contiguous frame axis.
    """
    if (isinstance(fs, bool) or not isinstance(fs, (int, float)) or not math.isfinite(fs) or fs <= 0):
        raise DataContractError("sampling_rate_invalid", "A finite positive source sampling rate is required")
    if len(boundaries) != 5 or any(type(f) is not int for f in boundaries):
        raise DataContractError("boundary_count_or_type", "Four repetitions require five integer boundaries")
    if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise DataContractError("boundary_order_invalid", "Boundaries must be strictly increasing in source order")
    rows = {frame: row for row, frame in enumerate(frames)}
    positions = []
    for index, boundary in enumerate(boundaries):
        if boundary in rows:
            positions.append(rows[boundary])
        elif index == 4 and frames and boundary == frames[-1] + 1:
            positions.append(len(frames))
        else:
            raise DataContractError("boundary_out_of_bounds", f"Boundary {boundary} is not an observed frame or final one-past-end")
    windows = []
    for rep, (start, end) in enumerate(zip(positions, positions[1:]), 1):
        count = end - start
        windows.append(RepetitionWindow(rep, boundaries[rep - 1], boundaries[rep], start, end,
                                        count, frames[start], frames[end - 1], count / fs, (count - 1) / fs))
    return windows


def source_sampling_rate(recording: XsensRecording) -> float:
    info = recording.read_general_information()
    values = [row[1]["value"] for row in info["rows"] if len(row) >= 2
              and row[0].get("value") == "Frame Rate"]
    if len(values) != 1:
        raise DataContractError("sampling_rate_evidence_invalid", "Expected one explicit Frame Rate metadata row")
    fs = values[0]
    declared = recording.descriptor.get("sampling_rate_hz")
    for value in (fs, declared):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise DataContractError("sampling_rate_invalid", "No finite positive sampling rate; no 60 Hz fallback")
    if not math.isclose(fs, declared, rel_tol=0, abs_tol=1e-9):
        raise DataContractError("sampling_rate_mismatch", "General Information and descriptor rates disagree")
    return float(fs)


def selected_boundaries(recording: XsensRecording) -> dict:
    markers = recording.read_table("Markers")
    if "Frame" not in markers.column_names:
        raise DataContractError("marker_frame_missing", "Markers table lacks Frame")
    events = [row["Frame"] for row in markers.to_pylist() if any(v is not None for v in row.values())]
    desc = recording.descriptor
    if events != recording.table_descriptor("Markers").get("marker_frames") or len(events) != desc.get("observed_marker_count"):
        raise DataContractError("marker_source_mismatch", "Stored marker events disagree with descriptor")
    if desc.get("expected_marker_count") == 5 and len(events) != 5:
        raise DataContractError("marker_count_invalid", f"Full task requires exactly five source markers; found {len(events)}; recording skipped without dropping markers")
    selection = desc.get("marker_selection")
    if selection is None:
        raise DataContractError("marker_selection_missing", "Refresh descriptor with format-xsens before segmentation")
    expected = select_marker_boundaries(events, expected_marker_count=desc.get("expected_marker_count"),
                                        expected_repetitions=desc.get("expected_repetitions"))
    if selection != expected:
        raise DataContractError("marker_selection_mismatch", "Descriptor selection disagrees with supported marker policy")
    if not selection["valid_for_expected_repetitions"]:
        raise DataContractError("marker_boundaries_invalid", "Effective boundaries fail count/order/non-null checks")
    return selection


def numerical_screen(table: pa.Table, windows: list[RepetitionWindow], sheet: str) -> list[dict]:
    """Report null/nonfinite counts and exact first locations; never repair values.

    Counts cover every signal channel in the four declared sheets, including
    the contralateral limb. Future feature eligibility must be channel-specific.
    """
    findings = []
    for name in table.column_names:
        if name == "Frame":
            continue
        column = table[name]
        if not (pa.types.is_floating(column.type) or pa.types.is_integer(column.type)):
            raise DataContractError("signal_type_invalid", f"{sheet}/{name}: expected numerical signal")
        if column.null_count == 0 and pc.all(pc.is_finite(column)).as_py() is True:
            continue
        values = column.to_pylist()
        missing = [i for i, value in enumerate(values) if value is None]
        nonfinite = [i for i, value in enumerate(values) if value is not None and not math.isfinite(value)]
        for code, indices in (("signal_missing", missing), ("signal_nonfinite", nonfinite)):
            if not indices:
                continue
            first = indices[0]
            findings.append({"code": code, "severity": "warning", "sheet": sheet, "column": name,
                             "count": len(indices), "first_row_index": first,
                             "first_source_excel_row": first + 2,
                             "first_frame": table["Frame"][first].as_py(),
                             "raw_value_repr": repr(values[first]),
                             "repetition_counts": {str(w.repetition): sum(w.row_start <= i < w.row_end_exclusive for i in indices)
                                                   for w in windows},
                             "treatment": "preserved; channel requires explicit exclusion before filtering"})
    return findings


def inspect_recording(recording: XsensRecording) -> dict:
    record = recording.record
    if record.get("task") not in {1, 2, 3, 4} or record.get("representation") != "full_recording":
        raise DataContractError("unsupported_activity", "This method handles full task1-task4 only")
    fs = source_sampling_rate(recording)
    selection = selected_boundaries(recording)
    tables = {sheet: recording.read_table(sheet) for sheet in SIGNAL_SHEETS}
    axes = {sheet: frame_axis(table, sheet) for sheet, table in tables.items()}
    frames = axes[SIGNAL_SHEETS[0]]
    if any(axis != frames for axis in axes.values()):
        raise DataContractError("frame_axes_mismatch", "Signal sheets do not share the same frame labels in the same order")
    windows = build_windows(frames, selection["boundary_frames"], fs)
    findings = [finding for sheet, table in tables.items() for finding in numerical_screen(table, windows, sheet)]
    unknown = {sheet: [c["name"] for c in recording.table_descriptor(sheet)["columns"]
                       if c["name"] != "Frame" and c.get("unit") is None] for sheet in SIGNAL_SHEETS}
    flags = [{"code": code, "severity": "warning", "origin": "formatter"}
             for code in recording.descriptor.get("warnings", [])]
    if any(unknown.values()):
        flags.append({"code": "units_unresolved", "severity": "warning", "origin": "screening"})
    if record.get("anatomical_side") not in {"left", "right"}:
        flags.append({"code": "anatomical_side_unresolved", "severity": "warning", "origin": "screening"})
    return {"status": "segmented", "method_id": SEGMENTATION_METHOD,
            "sampling_rate_hz": fs, "sampling_rate_evidence": "General Information/Frame Rate",
            "frame_axis": {"first_frame": frames[0], "last_frame": frames[-1], "row_count": len(frames),
                           "contiguous": True, "aligned_across_sheets": True,
                           "frame_equals_row_index": all(frame == row for row, frame in enumerate(frames))},
            "marker_selection": selection, "windows": [asdict(w) for w in windows],
            "full_signal_values_finite": not findings,
            "repetition_values_finite": {str(w.repetition): not any(f["repetition_counts"][str(w.repetition)] for f in findings)
                                         for w in windows},
            "numerical_findings": findings, "unknown_unit_columns": unknown, "flags": flags,
            "ready_for_features": False, "provenance": recording.provenance()}
