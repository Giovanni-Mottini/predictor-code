"""Complete Xsens extraction CLI, pairing and auditable derived artifacts."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import dtaidistance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy

from shared_code.provenance import implementation_hashes
from shared_code.io import atomic_json
from shared_code.artifacts import CANONICAL_FIELDS, atomic_csv, now
from shared_code.reader import DataContractError, SubjectReader
from shared_code.progress import ProgressReporter
from .xsens_features import FEATURE_NAMES, XsensConfig, extract_xsens
from .xsens_dtw import pair_hand_dtw

FEATURE_FIELDS = ("participant_key", *CANONICAL_FIELDS, "recorded_at_utc", "source_sha256", "repetition", "start_frame", "end_frame_exclusive",
                  "row_start", "row_end_exclusive", "n_samples", "first_sample_frame", "last_sample_frame", "duration_s", "sample_span_s",
                  "sampling_rate_hz", "method_id", "method_sha256", "marker_policy_id", "temporary_marker_rule", "dtw_pair_id",
                  *FEATURE_NAMES, "computed_feature_count", "qc_pass_feature_count", "excluded_features", "review_features")


def extract_subject(reader: SubjectReader, config: XsensConfig, *, sessions=None, timepoints=None, tasks=None,
                    limb_conditions=None, anatomical_sides=None, progress: ProgressReporter | None = None) -> dict:
    records = reader.select(modality="xsens", sessions=sessions, timepoints=timepoints,
                            tasks=tasks if tasks is not None else [1, 2, 3, 4],
                            limb_conditions=limb_conditions, anatomical_sides=anatomical_sides)
    results, trajectories = [], {}
    if progress:
        progress.update(f"{reader.participant_key}: selected {len(records)} recording(s)")
    for index, record in enumerate(records, 1):
        if progress:
            progress.update(f"Reading/extracting {reader.participant_key}: {record.get('source_path') or record['source_id']}",
                            completed=index - 1, total=len(records))
        metadata = {"participant_key": reader.participant_key, **{k: record.get(k) for k in CANONICAL_FIELDS}}
        metadata["recorded_at_utc"] = None
        recording = None
        try:
            recording = reader.open_xsens(record)
            metadata["recorded_at_utc"] = recording.descriptor.get("recorded_at_utc")
            extraction = extract_xsens(recording, config)
            if hashlib.sha256(recording.descriptor_path.read_bytes()).hexdigest() != recording.descriptor_sha256:
                raise DataContractError("metadata_changed_during_read", "Recording descriptor changed during extraction")
            results.append({**metadata, **extraction.report})
            trajectories[record["source_id"]] = extraction.hand_trajectories
        except (DataContractError, OSError, ValueError, KeyError, TypeError) as exc:
            skipped = isinstance(exc, DataContractError) and exc.code == "marker_count_invalid"
            results.append({**metadata, "status": "skipped" if skipped else "failed", "feature_rows": [], "channels": {},
                            "computed_feature_values": 0, "qc_pass_feature_values": 0, "ready_for_features": False,
                            "flags": [{"code": getattr(exc, "code", "record_read_failed"), "severity": "error", "message": str(exc)}],
                            "provenance": recording.provenance() if recording else None})
        if progress:
            progress.update(f"Scalar recording processed: {record.get('recording_id') or record['source_id']}; status={results[-1]['status']}",
                            completed=index, total=len(records))
    if progress:
        progress.update(f"{reader.participant_key}: pairing hand trajectories for DTW")
    pairs = pair_hand_dtw(results, trajectories, progress=progress)
    if progress:
        progress.update(f"{reader.participant_key}: DTW finished; {len(pairs)} unique pair(s); verifying metadata")
    for name, expected in reader.metadata_sha256.items():
        if hashlib.sha256((reader.root / name).read_bytes()).hexdigest() != expected:
            raise DataContractError("metadata_changed_during_read", f"{name} changed during extraction")
    return {"participant_key": reader.participant_key, "subject_path": str(reader.root),
            "download_completeness": reader.subject.get("download_completeness", "unknown"),
            "metadata_sha256": reader.metadata_sha256, "selected_recordings": len(records),
            "recording_status_counts": dict(Counter(r["status"] for r in results)),
            "feature_rows": sum(len(r["feature_rows"]) for r in results),
            "computed_feature_values": sum(r["computed_feature_values"] for r in results),
            "qc_pass_feature_values": sum(r["qc_pass_feature_values"] for r in results), "pairs": pairs, "results": results}


def atomic_parquet(path: Path, rows: list[dict]) -> None:
    integers = {"protocol_day", "task", "repetition", "start_frame", "end_frame_exclusive", "row_start", "row_end_exclusive",
                "n_samples", "first_sample_frame", "last_sample_frame", "computed_feature_count", "qc_pass_feature_count"}
    floats = {"duration_s", "sample_span_s", "sampling_rate_hz", *FEATURE_NAMES}
    fields = [pa.field(name, pa.int64() if name in integers else pa.float64() if name in floats
                       else pa.bool_() if name == "temporary_marker_rule" else pa.string()) for name in FEATURE_FIELDS]
    table = pa.Table.from_pylist(rows, schema=pa.schema(fields))
    fd, temporary = tempfile.mkstemp(prefix=".xsens-", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(table, temporary, compression="zstd")
        if not pq.read_table(temporary).equals(table):
            raise DataContractError("feature_output_roundtrip_failed", "Feature Parquet output did not round-trip")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_features(output: Path, report: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows, flags, pairs = [], [], []
    for subject in report["subjects"]:
        pairs.extend(subject["pairs"])
        for result in subject["results"]:
            metadata = {"participant_key": result["participant_key"], **{k: result[k] for k in CANONICAL_FIELDS},
                        "recorded_at_utc": result.get("recorded_at_utc")}
            for row in result["feature_rows"]:
                full = {**metadata, **row, "source_sha256": result["provenance"]["source_sha256"]}
                full["excluded_features"], full["review_features"] = ";".join(row["excluded_features"]), ";".join(row["review_features"])
                rows.append({k: full[k] for k in FEATURE_FIELDS})
            findings = [(None, f) for f in result["flags"]] + [(feature, f) for feature, channel in result["channels"].items() for f in channel["findings"]]
            for feature, finding in findings:
                flags.append({"participant_key": result["participant_key"], "source_id": result["source_id"], "source_path": result["source_path"],
                              "feature": feature, "code": finding["code"], "severity": finding["severity"], "details_json": json.dumps(finding, allow_nan=False)})
    atomic_json(output / "method.json", report["method"])
    atomic_csv(output / "features.csv", rows, FEATURE_FIELDS)
    atomic_parquet(output / "features.parquet", rows)
    atomic_json(output / "dtw_pairs.json", pairs)
    atomic_csv(output / "qc_flags.csv", flags, ("participant_key", "source_id", "source_path", "feature", "code", "severity", "details_json"))
    report["artifacts"] = {name: {"sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(), "size_bytes": (output / name).stat().st_size}
                           for name in ("method.json", "features.csv", "features.parquet", "dtw_pairs.json", "qc_flags.csv")}
    metadata = {k: v for k, v in report.items() if k != "subjects"}
    metadata["subjects"] = []
    for subject in report["subjects"]:
        summary = {k: v for k, v in subject.items() if k not in {"results", "pairs"}}
        summary["results"] = [{
            **{k: recording.get(k) for k in ("source_id", "source_path", "recording_id", "session_id", "timepoint",
                "task", "anatomical_side", "limb_condition", "status", "flags")},
            "feature_rows": len(recording["feature_rows"]),
            "source_sha256": (recording.get("provenance") or {}).get("source_sha256"),
            "processing_span": (recording.get("provenance") or {}).get("finite_span_processing"),
        } for recording in subject["results"]]
        metadata["subjects"].append(summary)
    atomic_json(output / "metadata.json", metadata)


def method_document(config: XsensConfig) -> dict:
    method = config.method()
    method["method_sha256"] = config.fingerprint()
    method["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                             "pyarrow": pa.__version__, "dtaidistance": dtaidistance.__version__}
    method["implementation_sha256"] = implementation_hashes(('xsens_code/xsens_features.py', 'xsens_code/elbow_rom.py', 'xsens_code/processing_span.py', 'xsens_code/xsens_dtw.py', 'xsens_code/smoothness.py', 'xsens_code/rom.py', 'xsens_code/reader.py', 'xsens_code/xsens.py', 'xsens_code/xsens_cli.py', 'xsens_code/__main__.py', 'predictor_import/markers.py', 'shared_code/reader.py', 'shared_code/artifacts.py', 'shared_code/io.py', 'shared_code/provenance.py', 'shared_code/progress.py'))
    return method


def extraction_status(subjects: list[dict]) -> int:
    if any(s["selected_recordings"] == 0 for s in subjects):
        return 2
    return 1 if any(r["status"] in {"failed", "skipped"} or r["qc_pass_feature_values"] != 15 * len(r["feature_rows"])
        or any(f["severity"] == "error" for c in r["channels"].values() for f in c["findings"])
        for s in subjects for r in s["results"]) else 0


def run_xsens(args, readers: list[SubjectReader], output: Path, *, progress: ProgressReporter | None = None) -> int:
    config = XsensConfig(args.cutoff_hz, args.sparc_cutoff_hz)
    method = method_document(config)
    started = now()
    subjects = []
    for index, reader in enumerate(readers, 1):
        if progress:
            progress.update(f"Starting participant {reader.participant_key}", completed=index - 1, total=len(readers))
        subjects.append(extract_subject(reader, config, sessions=args.session, timepoints=args.timepoint, tasks=args.task,
                                        limb_conditions=args.limb_condition, anatomical_sides=args.anatomical_side,
                                        progress=progress))
        if progress:
            progress.update(f"Participant processed: {reader.participant_key}", completed=index, total=len(readers))
    report = {"report_version": 1, "started_at": started, "finished_at": now(), "method": method,
              "filters": {"sessions": args.session, "timepoints": args.timepoint, "tasks": args.task,
                          "limb_conditions": args.limb_condition, "anatomical_sides": args.anatomical_side},
              "ready_for_features": False, "subjects": subjects}
    if progress:
        progress.update(f"Writing feature tables and metadata to {output}")
    write_features(output, report)
    print(json.dumps({"metadata": str(output / "metadata.json"), "subjects": [{k: s[k] for k in
          ("participant_key", "selected_recordings", "recording_status_counts", "feature_rows", "computed_feature_values", "qc_pass_feature_values")} for s in subjects]}, indent=2))
    if any(s["selected_recordings"] == 0 for s in subjects):
        if progress:
            progress.update(f"Finished; exit status 2 (empty recording selection); report: {output / 'metadata.json'}")
        return 2
    status = extraction_status(subjects)
    if progress:
        progress.update(f"Finished; exit status {status}; report: {output / 'metadata.json'}")
    return status
