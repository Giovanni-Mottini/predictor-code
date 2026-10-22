"""Run with python -m predictor_import; discovery and adaptation are separate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapter import atomic_json, make_plan, write_plan
from .inventory import scan
from .inventory_output import write_inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PREDICTOR inventory and metadata adapter")
    commands = parser.add_subparsers(dest="command", required=True)
    inventory_parser = commands.add_parser("inventory", help="Read-only source discovery; write inventory reports")
    inventory_parser.add_argument("source", type=Path)
    inventory_parser.add_argument("--output", type=Path, required=True)
    adapt_parser = commands.add_parser("adapt", help="Build a canonical JSON metadata plan; no numerical conversion")
    adapt_parser.add_argument("inventory", type=Path, help="inventory.json from the discovery command")
    adapt_parser.add_argument("--output", type=Path, required=True)
    adapt_parser.add_argument("--subject", help="Exact cohort-qualified key, e.g. PAZIENTI/FG91")
    adapt_parser.add_argument("--write", action="store_true", help="Stage subject.json and manifest.json; otherwise report only")
    adapt_parser.add_argument("--hash-sources", action="store_true", help="SHA-256 raw files when staging; requires --write")
    format_parser = commands.add_parser("format-xsens", help="Convert selected Xsens sources to verified Parquet and JSON")
    format_parser.add_argument("subject", type=Path, help="Standardized subject folder containing manifest.json")
    format_parser.add_argument("--source-id", action="append", help="Only convert these source IDs (repeatable)")
    format_parser.add_argument("--limit", type=int, help="Maximum number of recordings in this run")
    layout_parser = commands.add_parser("migrate-xsens-layout", help="Rename derived Xsens folders/tables without rebuilding data")
    layout_parser.add_argument("output", type=Path, help="Standardized root or participant directory")
    layout_parser.add_argument("--dry-run", action="store_true", help="Verify and preview changes without modifying standardized data")
    layout_parser.add_argument("--report", type=Path, help="Write source IDs, old/new paths and unchanged table checksums")
    garmin_layout_parser = commands.add_parser("migrate-garmin-layout", help="Rename derived Garmin folders/tables without rebuilding data")
    garmin_layout_parser.add_argument("output", type=Path, help="Standardized root or participant directory")
    garmin_layout_parser.add_argument("--dry-run", action="store_true", help="Verify and preview changes without modifying standardized data")
    garmin_layout_parser.add_argument("--report", type=Path, help="Write source IDs, old/new paths and unchanged table checksums")
    garmin_parser = commands.add_parser("format-garmin", help="Convert selected Garmin CSV/ZIP tables to verified Parquet")
    garmin_parser.add_argument("subject", type=Path, help="Standardized subject folder containing manifest.json")
    garmin_parser.add_argument("--source-id", action="append", help="Only convert these source IDs (repeatable)")
    garmin_parser.add_argument("--limit", type=int, help="Maximum number of source exports in this run")
    garmin_parser.add_argument("--table", action="append", help="Only these selected CSV paths or basenames; repeatable, resumable")
    batch_parser = commands.add_parser("standardize", help="Refresh inventory, stage metadata and resume batch formatting")
    batch_parser.add_argument("source", type=Path)
    batch_parser.add_argument("--output", type=Path, required=True)
    batch_parser.add_argument("--reports", type=Path, required=True)
    batch_parser.add_argument("--modality", choices=["xsens", "garmin"], action="append")
    batch_parser.add_argument("--subject", action="append", help="Exact cohort-qualified participant keys; repeatable")
    batch_parser.add_argument("--max-records", type=int, help="Process at most this many sources; rerun to continue")
    batch_parser.add_argument("--min-free-gib", type=float, default=2.0)
    batch_parser.add_argument("--plan-only", action="store_true")
    drive_parser = commands.add_parser("refresh-drive", help="Refresh standardized data by Drive source hash; temporary raw downloads only")
    drive_parser.add_argument("remote", help="Configured Drive-root rclone remote, e.g. predictor:")
    drive_parser.add_argument("--output", type=Path, required=True)
    drive_parser.add_argument("--reports", type=Path, required=True)
    drive_parser.add_argument("--subject", action="append", help="Exact cohort-qualified key; repeatable")
    drive_parser.add_argument("--local-source", type=Path, help="Optional existing raw tree for first-run hash seeding; read-only")
    drive_parser.add_argument("--staging-dir", type=Path, help="Temporary download parent; default: output parent")
    drive_parser.add_argument("--min-free-gib", type=float, default=2.0)
    drive_parser.add_argument("--tpslimit", type=float, default=2.0)
    drive_parser.add_argument("--dry-run", action="store_true", help="List metadata and verify existing outputs; download/modify no data")
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            inventory = scan(args.source)
            write_inventory(inventory, args.output)
            print(json.dumps(inventory["summary"], indent=2))
        elif args.command in {"migrate-xsens-layout", "migrate-garmin-layout"}:
            from .layout import migrate_layout
            modality = "xsens" if args.command == "migrate-xsens-layout" else "garmin"
            report = migrate_layout(args.output, dry_run=args.dry_run, modality=modality)
            if args.report:
                atomic_json(args.report, report)
            print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
        elif args.command == "refresh-drive":
            from .drive_refresh import RcloneDrive, run_refresh
            report = run_refresh(args.remote, args.output, args.reports, subjects=args.subject,
                local_source=args.local_source, staging_dir=args.staging_dir, min_free_gib=args.min_free_gib,
                dry_run=args.dry_run, drive=RcloneDrive(args.remote, tpslimit=args.tpslimit),
                progress=lambda row: print(json.dumps(row), file=sys.stderr, flush=True))
            print(json.dumps({k: report[k] for k in ("state", "summary", "blocked_participants")}, indent=2))
            return 1 if report["state"].endswith("with_findings") else 0
        elif args.command == "standardize":
            from .batch import run_batch
            report = run_batch(args.source, args.output, args.reports, modalities=args.modality, subjects=args.subject,
                max_records=args.max_records, min_free_gib=args.min_free_gib, plan_only=args.plan_only,
                progress=lambda row: print(json.dumps(row), file=sys.stderr, flush=True))
            print(json.dumps({k: report[k] for k in ("state", "summary", "stop_reason") if k in report}, indent=2))
            return 1 if report.get("state") in {"partial", "completed_with_failures"} else 0
        elif args.command in {"format-xsens", "format-garmin"}:
            if args.limit is not None and args.limit < 1:
                parser.error("--limit must be positive")
            if args.command == "format-xsens":
                from .formatter import format_subject
                report = format_subject(args.subject, source_ids=args.source_id, limit=args.limit)
            else:
                from .garmin_formatter import format_subject
                report = format_subject(args.subject, source_ids=args.source_id, limit=args.limit, table_names=args.table,
                    progress=lambda name, rows: print(f"Verified {name}: {rows} rows", file=sys.stderr, flush=True))
            print(json.dumps(report, indent=2))
            return 1 if report["failed"] else 0
        else:
            if args.hash_sources and not args.write:
                parser.error("--hash-sources requires --write")
            inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
            plan = make_plan(inventory, args.subject)
            root, output = Path(inventory["source_root"]).resolve(), args.output.resolve()
            if output == root or output.is_relative_to(root) or root.is_relative_to(output):
                raise ValueError("Adapter output must be separate from raw sources")
            if args.write:
                summary = write_plan(plan, output, hash_sources=args.hash_sources)
            else:
                atomic_json(output / "adapter_plan.json", plan)
                summary = {"mode": "dry_run", **plan["summary"], "blocked_participant_keys": plan["blocked_participant_keys"]}
            print(json.dumps(summary, indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
