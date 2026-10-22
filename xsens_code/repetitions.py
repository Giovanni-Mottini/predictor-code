"""Inspect full-task repetition windows without extracting numerical features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from shared_code.io import atomic_json
from shared_code.reader import DataContractError, SubjectReader
from shared_code.progress import ProgressReporter
from shared_code.artifacts import CANONICAL_FIELDS, atomic_csv
from . import SEGMENTATION_METHOD
from .xsens import SIGNAL_SHEETS, inspect_recording

WINDOW_FIELDS = ("participant_key", *CANONICAL_FIELDS, "repetition", "start_frame", "end_frame_exclusive",
                 "row_start", "row_end_exclusive", "n_samples", "first_sample_frame", "last_sample_frame",
                 "duration_s", "sample_span_s", "sampling_rate_hz", "values_finite",
                 "temporary_marker_rule", "marker_policy_id")


def audit_subject(reader: SubjectReader, *, sessions=None, timepoints=None, tasks=None,
                  limb_conditions=None, anatomical_sides=None, progress: ProgressReporter | None = None) -> dict:
    records = reader.select(modality="xsens", sessions=sessions, timepoints=timepoints,
                            tasks=tasks if tasks is not None else [1, 2, 3, 4],
                            limb_conditions=limb_conditions, anatomical_sides=anatomical_sides)
    results = []
    if progress:
        progress.update(f"{reader.participant_key}: selected {len(records)} recording(s)")
    for index, record in enumerate(records, 1):
        if progress:
            progress.update(f"Inspecting {reader.participant_key}: {record.get('source_path') or record['source_id']}",
                            completed=index - 1, total=len(records))
        metadata = {"participant_key": reader.participant_key, **{k: record.get(k) for k in CANONICAL_FIELDS}}
        recording = None
        try:
            recording = reader.open_xsens(record)
            result = inspect_recording(recording)
            # Descriptor changes during inspection invalidate a mixed snapshot.
            if hashlib.sha256(recording.descriptor_path.read_bytes()).hexdigest() != recording.descriptor_sha256:
                raise DataContractError("metadata_changed_during_read", "Recording descriptor changed during inspection")
            results.append({**metadata, **result})
        except (DataContractError, OSError, ValueError, KeyError, TypeError) as exc:
            results.append({**metadata, "status": "failed", "windows": [], "ready_for_features": False,
                            "flags": [{"code": getattr(exc, "code", "record_read_failed"), "severity": "error",
                                       "message": str(exc)}],
                            "provenance": recording.provenance() if recording else None})
        if progress:
            progress.update(f"Recording inspected: {record.get('recording_id') or record['source_id']}; status={results[-1]['status']}",
                            completed=index, total=len(records))
    for name, expected in reader.metadata_sha256.items():
        if hashlib.sha256((reader.root / name).read_bytes()).hexdigest() != expected:
            raise DataContractError("metadata_changed_during_read", f"{name} changed during inspection")
    segmented = sum(r["status"] == "segmented" for r in results)
    return {"participant_key": reader.participant_key, "subject_path": str(reader.root),
            "download_completeness": reader.subject.get("download_completeness", "unknown"),
            "metadata_sha256": reader.metadata_sha256,
            "selected_recordings": len(records), "segmented_recordings": segmented,
            "failed_recordings": len(records) - segmented,
            "repetitions": sum(len(r["windows"]) for r in results),
            "mapped_xsens_outside_requested_scope": len(reader.select(modality="xsens")) - len(records),
            "results": results}


def write_windows(output: Path, result: dict) -> None:
    """Persist repetition windows and compact source/check metadata."""
    output.mkdir(parents=True, exist_ok=True)
    windows, flags = [], []
    for subject in result["subjects"]:
        for recording in subject["results"]:
            canonical = {"participant_key": recording["participant_key"], **{k: recording[k] for k in CANONICAL_FIELDS}}
            for window in recording["windows"]:
                windows.append({**canonical, **window, "sampling_rate_hz": recording["sampling_rate_hz"],
                    "values_finite": recording["repetition_values_finite"][str(window["repetition"])],
                    "temporary_marker_rule": recording["marker_selection"]["rule_applied"],
                    "marker_policy_id": recording["marker_selection"]["policy_id"]})
            for finding in recording["flags"] + recording.get("numerical_findings", []):
                flags.append({"participant_key": recording["participant_key"], "source_id": recording["source_id"],
                    "source_path": recording["source_path"], "code": finding["code"], "severity": finding["severity"]})
    atomic_csv(output / "repetitions.csv", windows, WINDOW_FIELDS)
    atomic_csv(output / "qc_flags.csv", flags, ("participant_key", "source_id", "source_path", "code", "severity"))
    atomic_json(output / "metadata.json", {k: v for k, v in result.items() if k != "subjects"} | {
        "subjects": [{k: v for k, v in subject.items() if k != "results"} for subject in result["subjects"]]})
