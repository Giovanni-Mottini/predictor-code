"""Xsens quaternion-default features, repetition inspection and local raw-to-XLSX pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared_code.artifacts import now, validate_output
from shared_code.progress import ProgressReporter
from shared_code.reader import DataContractError, SubjectReader
from . import SEGMENTATION_METHOD
from .repetitions import audit_subject, write_windows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    features = sub.add_parser("xsens-features", help="Extract fifteen features with quaternion elbow bending")
    repetitions = sub.add_parser("xsens-repetitions", help="Inspect task windows and export structured metadata")
    pipeline = sub.add_parser("pipeline", help="Prepare Dati PREDICTOR, extract features and create patient workbooks")
    pipeline.add_argument("source", type=Path, help="Local Dati PREDICTOR directory")
    pipeline.add_argument("--subject", action="append", required=True, help="PAZIENTI/<subject>; repeatable")
    pipeline.add_argument("--standardized", type=Path, default=Path("standardized"))
    pipeline.add_argument("--output", type=Path, default=Path("OUTPUT"))
    pipeline.add_argument("--work-directory", type=Path, help="Fresh run directory; default: derived/xsens/<unique-run>")
    for command in (features, repetitions):
        command.add_argument("subjects", nargs="+", type=Path, help="Explicit standardized participant directories")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--limb-condition", action="append", choices=["non_paretic", "paretic", "simulated"])
        command.add_argument("--anatomical-side", action="append", choices=["left", "right"])
    for command in (features, repetitions, pipeline):
        command.add_argument("--session", action="append")
        command.add_argument("--timepoint", action="append")
        command.add_argument("--task", type=int, action="append", choices=[1, 2, 3, 4])
        command.add_argument("--no-progress", action="store_true")
    for command in (features, pipeline):
        command.add_argument("--cutoff-hz", type=float, default=6.0)
        command.add_argument("--sparc-cutoff-hz", type=float, default=15.0)
    args = parser.parse_args(argv)
    progress = ProgressReporter(args.command, enabled=not args.no_progress)
    try:
        if args.command == "pipeline":
            from .pipeline import run_pipeline
            return run_pipeline(args)
        readers = [SubjectReader(path) for path in args.subjects]
        if len({reader.participant_key for reader in readers}) != len(readers):
            raise DataContractError("duplicate_subject", "Select each cohort-qualified participant once")
        output = validate_output(args.output, readers)
        if args.command == "xsens-features":
            from .xsens_cli import run_xsens
            return run_xsens(args, readers, output, progress=progress)
        subjects = [audit_subject(reader, sessions=args.session, timepoints=args.timepoint, tasks=args.task,
            limb_conditions=args.limb_condition, anatomical_sides=args.anatomical_side, progress=progress) for reader in readers]
        write_windows(output, {"metadata_version": 1, "method_id": SEGMENTATION_METHOD, "finished_at": now(), "ready_for_features": False, "subjects": subjects})
        status = 2 if any(s["selected_recordings"] == 0 for s in subjects) else 1 if any(s["failed_recordings"] for s in subjects) else 0
        print(json.dumps({"metadata": str(output / "metadata.json"), "subjects": [
            {k: s[k] for k in ("participant_key", "selected_recordings", "segmented_recordings", "failed_recordings", "repetitions")}
            for s in subjects]}, indent=2))
        progress.update(f"Finished; exit status {status}")
        return status
    except (DataContractError, OSError, ValueError, KeyError, TypeError) as exc:
        progress.update(f"Stopped: {getattr(exc, 'code', 'read_failed')}: {exc}")
        parser.exit(2, f"{getattr(exc, 'code', 'read_failed')}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
