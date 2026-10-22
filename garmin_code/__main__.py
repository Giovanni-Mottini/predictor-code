"""Bounded Garmin preparation and gap-aware window features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared_code.reader import DataContractError, SubjectReader
from shared_code.progress import ProgressReporter


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    template_command = sub.add_parser("garmin-period-template", help="Write canonical timepoints with unset monitoring boundaries")
    template_command.add_argument("subjects", nargs="+", type=Path)
    template_command.add_argument("--output", type=Path, required=True, help="New period-definition JSON file")
    align_command = sub.add_parser("garmin-align", help="Align verified window artifacts to explicit monitoring periods")
    align_command.add_argument("subjects", nargs="+", type=Path)
    align_command.add_argument("--windows", nargs="+", type=Path, required=True, help="Existing garmin-windows output directories")
    align_command.add_argument("--periods", type=Path, required=True, help="Explicit cohort-qualified monitoring periods JSON")
    align_command.add_argument("--output", type=Path, required=True)
    clinical_command = sub.add_parser("garmin-clinical", help="Extract and align T0–T10 on exact recorded patient clinical days")
    clinical_command.add_argument("subjects", nargs="+", type=Path)
    clinical_command.add_argument("--output", type=Path, required=True)
    clinical_command.add_argument("--max-clinical-days", type=int, default=64, help="Maximum explicitly dated patient days in this batch")
    assessment_command = sub.add_parser("garmin-assessments", help="T1-T10 Xsens-centered five-day means and verified patient XLSX workbooks")
    assessment_command.add_argument("subjects", nargs="+", type=Path)
    assessment_command.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    assessment_command.add_argument("--anchor-overrides", type=Path, help="Explicit source-backed assessment date mapping JSON")
    garmin_command = sub.add_parser("garmin-windows", help="Bounded Garmin preparation and gap-aware window features")
    garmin_command.add_argument("subjects", nargs="+", type=Path)
    garmin_command.add_argument("--output", type=Path, required=True)
    garmin_command.add_argument("--ranges", type=Path, help="JSON mapping exact participant_key to start/end_exclusive dates")
    garmin_command.add_argument("--start", help="Local start date, inclusive; used for every explicit subject")
    garmin_command.add_argument("--end-exclusive", help="Local end date, exclusive")
    for command in (garmin_command, clinical_command, assessment_command):
        command.add_argument("--window-minutes", type=int, default=30)
        command.add_argument("--calorie-rate", type=float, default=25)
        command.add_argument("--max-pair-gap-seconds", type=float, default=5)
        command.add_argument("--max-selected-rows", type=int, default=2_000_000)
        command.add_argument("--hrv-entropy-beats", type=int, default=1000, help="Contiguous beats per entropy epoch (1000–10000)")
        command.add_argument("--hrv-clock-tolerance-ms", type=float, default=50, help="Max |UTC step - following BBI| for consecutive beats; no timestamp correction")
        command.add_argument("--hrv-method", choices=("notebook", "segments"), default="notebook", help="notebook: first-wave block method; segments: gap-free epochs")
        command.add_argument("--bbi-screens", action="store_true", help="Use only beats in 230-1550 ms, in worn intervals, with MET<3")
        command.add_argument("--exclude-second-stream", action="store_true", help="Drop beats, stress and minute HR in 5-min bins holding more beat time than clock time")
        command.add_argument("--bbi-local-median-pct", type=float, help="Drop beats deviating more than this percent from the 31-beat median (e.g. 20)")
    assessment_command.set_defaults(window_minutes=360)
    for command in (garmin_command, clinical_command, align_command, template_command, assessment_command):
        command.add_argument("--no-progress", action="store_true", help="Suppress execution progress on stderr")
    args = parser.parse_args(argv)
    progress = ProgressReporter(args.command, enabled=not args.no_progress)
    try:
        progress.update(f"Loading metadata for {len(args.subjects)} participant(s)")
        if args.command == "garmin-period-template":
            from .alignment import period_template
            from shared_code.artifacts import validate_output
            from shared_code.io import atomic_json
            output = validate_output(args.output, [SubjectReader(p) for p in args.subjects])
            if output.exists():
                raise DataContractError("output_exists", "Template output already exists; choose a new file")
            atomic_json(output, period_template(args.subjects))
            print(json.dumps({"template": str(output), "boundaries": "unset"}, indent=2))
            progress.update(f"Finished; exit status 0; template: {output}")
            return 0
        if args.command == "garmin-align":
            from .alignment import align_periods
            periods, values, report = align_periods(args.subjects, args.windows, args.periods, args.output, progress=progress)
            print(json.dumps({"report": str(Path(args.output) / "report.md"), "periods": len(periods),
                              "feature_rows": len(values), "unassigned_windows": report["unassigned_windows"],
                              "ready_for_features": False}, indent=2))
            status = 1 if report["review_required"] else 0
            progress.update(f"Finished; exit status {status}; report: {Path(args.output) / 'report.md'}")
            return status
        from .garmin import GarminConfig
        from .garmin_workflow import runs_from_ranges, run_windows
        config = GarminConfig(window_minutes=args.window_minutes, calorie_rate_kcal_min=args.calorie_rate,
                              max_pair_gap_s=args.max_pair_gap_seconds,max_selected_rows=args.max_selected_rows,
                              hrv_entropy_beats=args.hrv_entropy_beats,hrv_clock_tolerance_ms=args.hrv_clock_tolerance_ms,
                              hrv_method=args.hrv_method, bbi_screens=args.bbi_screens,
                              exclude_second_stream=args.exclude_second_stream, bbi_local_median_pct=args.bbi_local_median_pct)
        if args.command == "garmin-assessments":
            from .assessments import run_assessments
            result = run_assessments(args.subjects, args.output, override_path=args.anchor_overrides, config=config, progress=progress)
            print(json.dumps({"state": result["state"], "workbooks": [v["workbook"] for v in result["subjects"].values()], "metadata": str(args.output/"metadata.json")}, indent=2))
            return 0
        if args.command == "garmin-clinical":
            from .alignment import run_clinical_alignment
            periods, values, report = run_clinical_alignment(args.subjects, args.output, config, max_days=args.max_clinical_days,
                                                            progress=progress)
            print(json.dumps({"report": str(Path(args.output) / "report.md"), "clinical_days": len(periods),
                              "feature_rows": len(values), "ready_for_features": False}, indent=2))
            status = 1 if report["review_required"] else 0
            progress.update(f"Finished; exit status {status}; report: {Path(args.output) / 'report.md'}")
            return status
        if args.ranges:
            if args.start or args.end_exclusive:
                raise DataContractError("conflicting_dates", "Use --ranges or --start/--end-exclusive")
            ranges = json.loads(args.ranges.read_text())
        else:
            if not args.start or not args.end_exclusive:
                raise DataContractError("dates_required", "Supply --ranges or explicit --start and --end-exclusive")
            ranges = {SubjectReader(p).participant_key: {"start": args.start, "end_exclusive": args.end_exclusive} for p in args.subjects}
        progress.update("Resolving Garmin exports for selected date ranges")
        runs = runs_from_ranges(args.subjects,ranges,config)
        frame, _, report = run_windows(runs,args.output,progress=progress)
        print(json.dumps({"report":str(Path(args.output)/"report.md"),"windows":len(frame),
                          "participants":[r.participant_key for r in runs],"ready_for_features":False},indent=2))
        status = 1 if any(f["severity"] == "review" for s in report["subjects"] for p in s["streams"].values() for f in p["findings"]) else 0
        progress.update(f"Finished; exit status {status}; report: {Path(args.output) / 'report.md'}")
        return status
    except (DataContractError, OSError, ValueError, KeyError, TypeError) as exc:
        progress.update(f"Stopped: {getattr(exc, 'code', 'read_failed')}: {exc}")
        parser.exit(2, f"{getattr(exc, 'code', 'read_failed')}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
