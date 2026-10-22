"""Filter a common finite span while preserving original source coordinates."""
from copy import deepcopy

import numpy as np

from shared_code.reader import DataContractError
from .xsens import build_windows, frame_axis, selected_boundaries, source_sampling_rate

PROCESSING_SPAN_POLICY = {
    "policy_id": "xsens-contiguous-finite-edge-span-v1",
    "scope": "full-feature extraction only; common selected input channels",
    "allowed_gaps": "leading/trailing nonfinite samples outside every repetition",
    "minimum_margin_samples": 12,
    "interior_gaps": "no cropping; normal channel exclusion rules apply",
    "source_coordinates": "original frames, row indices and Excel rows retained",
    "imputation": False,
    "source_modified": False,
}


class _FiniteSpanRecording:
    def __init__(self, recording, start, end):
        self.recording, self.start, self.end = recording, start, end

    def __getattr__(self, name):
        return getattr(self.recording, name)

    def read_table(self, sheet, columns=None):
        table = self.recording.read_table(sheet, columns)
        return table if sheet == "Markers" else table.slice(self.start, self.end - self.start)

    def provenance(self):
        result = self.recording.provenance()
        result["finite_span_processing"] = {
            "policy_id": PROCESSING_SPAN_POLICY["policy_id"],
            "source_row_start": self.start, "source_row_end_exclusive": self.end,
            "no_imputation": True,
        }
        return result


def _restore_row_indices(node, offset):
    keys = {"first_row_index", "min_row_index", "max_row_index", "row_index",
            "source_excel_row", "first_source_excel_row", "baseline_row_start",
            "baseline_row_end_exclusive"}
    if isinstance(node, dict):
        for key, value in node.items():
            if key in keys and isinstance(value, int):
                node[key] += offset
            else:
                _restore_row_indices(value, offset)
    elif isinstance(node, list):
        for value in node:
            _restore_row_indices(value, offset)


def extract_on_finite_span(recording, config, extractor, fields):
    """Apply edge recovery once, including an outer quaternion-elbow extractor.

    An unavailable auxiliary channel never blocks normal feature extraction.
    No interior gap or gap inside a repetition is removed or interpolated.
    """
    if isinstance(recording, _FiniteSpanRecording):
        return extractor(recording, config)
    frames = frame_axis(recording.read_table("Joint Angles ZXY", columns=["Frame"]), "Joint Angles ZXY")
    boundaries = selected_boundaries(recording)["boundary_frames"]
    windows = build_windows(frames, boundaries, source_sampling_rate(recording))
    finite = np.ones(len(frames), dtype=bool)
    try:
        for sheet, columns in fields.items():
            table = recording.read_table(sheet, columns=["Frame", *columns])
            if frame_axis(table, sheet) != frames:
                return extractor(recording, config)
            values = np.column_stack([table[name].to_pylist() for name in columns]).astype(float)
            finite &= np.isfinite(values).all(axis=1)
    except (DataContractError, OSError, ValueError, KeyError, TypeError):
        return extractor(recording, config)
    valid = np.flatnonzero(finite)
    if not len(valid):
        return extractor(recording, config)
    start, end = int(valid[0]), int(valid[-1]) + 1
    margin = PROCESSING_SPAN_POLICY["minimum_margin_samples"]
    if ((start == 0 and end == len(frames)) or not finite[start:end].all()
            or any(w.row_start - start < margin or end - w.row_end_exclusive < margin for w in windows)):
        return extractor(recording, config)
    result = extractor(_FiniteSpanRecording(recording, start, end), config)
    report = result.report
    _restore_row_indices(report, start)
    for row in report["feature_rows"]:
        row["row_start"] += start
        row["row_end_exclusive"] += start
    report["processing_frame_axis"] = deepcopy(report["frame_axis"])
    report["frame_axis"] = {"first_frame": frames[0], "last_frame": frames[-1],
                            "row_count": len(frames), "frame_equals_row_index": frames == list(range(len(frames)))}
    report["flags"].append({
        "code": "leading_or_trailing_missing_samples_outside_windows", "severity": "info",
        "policy_id": PROCESSING_SPAN_POLICY["policy_id"],
        "leading_frames": frames[:start], "trailing_frames": frames[end:],
        "processing_start_frame": frames[start], "processing_end_frame_inclusive": frames[end - 1],
        "source_preserved": True,
        "treatment": "contiguous finite span filtered before original windows",
    })
    return result
