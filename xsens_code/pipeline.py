"""Coordinate local Dati PREDICTOR preparation, feature extraction and patient XLSX export."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from predictor_import.batch import run_batch
from shared_code.artifacts import now, validate_output
from shared_code.io import atomic_json
from shared_code.progress import ProgressReporter
from shared_code.reader import DataContractError, SubjectReader
from .export import export_workbooks, patient_destination
from .xsens_cli import extract_subject, method_document, write_features, extraction_status
from .xsens_features import XsensConfig


def run_pipeline(args) -> int:
    source, prepared, output = args.source.resolve(), args.standardized.resolve(), args.output.resolve()
    for target in (prepared, output):
        if target == source or target.is_relative_to(source) or source.is_relative_to(target):
            raise DataContractError("unsafe_output", "Prepared data and OUTPUT must be separate from raw sources")
    if prepared == output or prepared.is_relative_to(output) or output.is_relative_to(prepared):
        raise DataContractError("unsafe_output", "Prepared data and OUTPUT must be separate trees")
    for key in args.subject:
        if patient_destination(output, key).exists():
            raise DataContractError("output_exists", "Use a fresh OUTPUT directory; existing workbooks are preserved")
    if len(set(args.subject)) != len(args.subject):
        raise DataContractError("duplicate_subject", "Select each participant once")
    config = XsensConfig(args.cutoff_hz, args.sparc_cutoff_hz)
    run = args.work_directory.resolve() if args.work_directory else Path("derived/xsens") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    run = run.resolve()
    for protected in (source, prepared, output):
        if run == protected or run.is_relative_to(protected) or protected.is_relative_to(run):
            raise DataContractError("unsafe_output", "Run artifacts must be separate from raw, prepared and OUTPUT trees")
    if run.exists():
        raise DataContractError("output_exists", "Use a fresh work directory")
    progress = ProgressReporter("xsens-pipeline", enabled=not args.no_progress)
    progress.update("Preparing selected Xsens sources and clinical metadata")
    preparation = run_batch(source, prepared, run / "preparation", modalities=["xsens"], subjects=args.subject,
        progress=lambda event: progress.update(f"Preparation: {event.get('participant_key', '')} {event.get('source_path', '')}"))
    if preparation["state"] != "completed":
        print(json.dumps({"state": "preparation_incomplete", "run_directory": str(run), "preparation": preparation["summary"]}, indent=2))
        return 1
    readers = [SubjectReader(prepared / key) for key in args.subject]
    validate_output(output, readers)
    method = method_document(config)
    directories, summaries, statuses = [], [], []
    for reader in readers:
        result = extract_subject(reader, config, sessions=args.session, timepoints=args.timepoint, tasks=args.task,
                                 progress=progress)
        destination = run / "runs" / reader.participant_key / "xsens"
        write_features(destination, {"metadata_version": 1, "started_at": now(), "finished_at": now(),
            "method": method, "filters": {"sessions": args.session, "timepoints": args.timepoint, "tasks": args.task},
            "ready_for_features": False, "subjects": [result]})
        directories.append(destination)
        statuses.append(extraction_status([result]))
        summaries.append({k: result[k] for k in ("participant_key", "selected_recordings", "recording_status_counts", "feature_rows")})
    if 2 in statuses or any(summary["feature_rows"] == 0 for summary in summaries):
        print(json.dumps({"state": "no_export_for_empty_selection", "run_directory": str(run), "subjects": summaries}, indent=2))
        return 2
    progress.update("Exporting and verifying patient workbooks")
    workbooks = export_workbooks(directories, output)
    status = 1 if 1 in statuses else 0
    receipt = {"state": "completed_with_findings" if status else "completed", "method_id": method["method_id"],
        "raw_source": str(source), "standardized": str(prepared), "output": str(output),
        "run_directory": str(run), "subjects": summaries, "workbooks": workbooks}
    atomic_json(run / "run.json", receipt)
    print(json.dumps(receipt, indent=2))
    progress.update(f"Finished; exit status {status}")
    return status
