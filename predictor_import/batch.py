"""Refresh, stage and resume conversion across cohorts with a content ledger."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import shutil

from . import SCHEMA_VERSION
from .adapter import atomic_json, make_plan, write_plan
from .inventory import now, scan
from .inventory_output import write_csv, write_inventory

KINDS = {"xsens_workbook", "garmin_archive", "garmin_table"}


def load_registry(output: Path) -> dict:
    path = output / "source_registry.json"
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
        if registry.get("registry_version") != 1:
            raise ValueError("Unsupported source registry version")
        return registry
    registry = {"registry_version": 1, "sources": {}}
    # Preserve the pre-batch checksum baseline, including removed ZIP sources.
    for manifest_path in output.glob("*/*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["records"]:
            if record["artifact_kind"] not in KINDS:
                continue
            sha = record.get("sha256")
            registry["sources"][record["source_id"]] = {
                "source_path": record["source_path"], "participant_key": manifest["participant_key"],
                "modality": record["modality"], "artifact_kind": record["artifact_kind"],
                "sha256": sha, "present": True, "verified_this_run": False,
                "versions": [{"sha256": sha, "observed_at": None, "evidence": "preexisting_manifest"}] if sha else [],
            }
    return registry


def update_content(entry: dict, sha: str, timestamp: str, *, evidence="verified_formatter_source_sha256") -> str:
    previous = entry.get("sha256")
    state = "unchanged" if previous == sha else "modified" if previous else "unknown_previous_content" if entry.get("seen_before") else "new"
    if previous != sha:
        entry["versions"].append({"sha256": sha, "observed_at": timestamp, "evidence": evidence})
    entry.update({"sha256": sha, "verified_this_run": True, "content_verified_at": timestamp, "last_content_state": state})
    return state


def write_batch_report(output: Path, report_dir: Path, report: dict, registry: dict) -> None:
    report["summary"] = {
        "processed_sources": len(report["results"]),
        "conversion_statuses": dict(Counter(r["status"] for r in report["results"])),
        "content_states": dict(Counter(r["content_state"] for r in report["results"])),
        "tables_rebuilt": sum(r.get("tables_rebuilt", 0) for r in report["results"]),
        "tables_reused": sum(r.get("tables_reused", 0) for r in report["results"]),
        "missing_previously_seen_sources": sum(not e["present"] for e in registry["sources"].values()),
    }
    coverage = []
    for path in sorted(output.glob("*/*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        selected = [r for r in manifest["records"] if r["disposition"] == "mapped" and r["artifact_kind"] in KINDS]
        coverage.append({"participant_key": manifest["participant_key"], "selected_sources": len(selected),
            "converted_sources": sum(r["formatter_status"] == "converted" for r in selected),
            "partial_sources": sum(bool(r.get("formatter_report")) and r["formatter_report"].get("complete_selected_tables") is False for r in selected),
            "failed_sources": sum(r["formatter_status"] == "failed" for r in selected),
            "not_converted_sources": sum(r["formatter_status"] == "not_converted" for r in selected),
            "ignored_sources": sum(r["disposition"] == "excluded" for r in manifest["records"]),
            "deferred_sources": sum(r["disposition"] == "deferred" for r in manifest["records"])})
    report["coverage"] = coverage
    registry["updated_at"] = now()
    atomic_json(output / "source_registry.json", registry)
    atomic_json(report_dir / "batch_report.json", report)
    write_csv(report_dir / "coverage.csv", coverage, ["participant_key", "selected_sources", "converted_sources", "partial_sources", "failed_sources", "not_converted_sources", "ignored_sources", "deferred_sources"])
    write_csv(report_dir / "results.csv", report["results"], ["participant_key", "modality", "source_id", "source_path", "status", "content_state", "previous_sha256", "sha256", "tables_rebuilt", "tables_reused", "error"])



def run_batch(source: Path, output: Path, report_dir: Path, *, modalities=None, subjects=None,
              max_records=None, min_free_gib=2.0, plan_only=False, progress=None) -> dict:
    source, output, report_dir = source.resolve(), output.resolve(), report_dir.resolve()
    for path in (output, report_dir):
        if path == source or path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError("Batch output/reports must be separate from raw sources")
    if max_records is not None and max_records < 1:
        raise ValueError("Maximum records must be positive")
    if min_free_gib < 0:
        raise ValueError("Minimum free space must be nonnegative")
    modalities = set(modalities or ("xsens", "garmin"))
    if not modalities <= {"xsens", "garmin"}:
        raise ValueError("Supported batch modalities are xsens and garmin")
    inventory = scan(source)
    write_inventory(inventory, report_dir / "inventory")
    plan = make_plan(inventory)
    if subjects:
        keys = {p["subject"]["participant_key"] for p in plan["subjects"]}
        if set(subjects) - keys:
            raise ValueError("Unknown or blocked participant key")
        plan["subjects"] = [p for p in plan["subjects"] if p["subject"]["participant_key"] in subjects]
        plan["summary"] = {"subjects": len(plan["subjects"]),
                           "records": sum(len(p["manifest"]["records"]) for p in plan["subjects"]),
                           "dispositions": dict(Counter(r["disposition"] for p in plan["subjects"] for r in p["manifest"]["records"]))}
    atomic_json(report_dir / "adapter_plan.json", plan)
    if plan_only:
        return {"state": "plan_only", "subjects": len(plan["subjects"]), "inventory_summary": inventory["summary"]}
    output.mkdir(parents=True, exist_ok=True)
    registry = load_registry(output)
    sources = registry["sources"]
    previous_shas = {key: entry.get("sha256") for key, entry in sources.items()}
    for entry in sources.values():
        entry["present"], entry["verified_this_run"], entry["seen_before"] = False, False, True
    for file in inventory["files"]:
        if file["artifact_kind"] not in KINDS:
            continue
        entry = sources.setdefault(file["source_id"], {"sha256": None, "versions": [], "seen_before": False,
                                                     "verified_this_run": False})
        entry.update({"source_path": file["source_path"], "participant_key": f"{file['cohort']}/{file['subject_id']}",
                      "modality": file["modality"], "artifact_kind": file["artifact_kind"], "present": True,
                      "last_seen_at": inventory["scan_finished_at"], "disposition": file["disposition"]})
    staging = write_plan(copy.deepcopy(plan), output)
    queue = []
    for planned in plan["subjects"]:
        key = planned["subject"]["participant_key"]
        path = output / planned["subject"]["cohort"] / planned["subject"]["subject_id"]
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        queue.extend((key, path, record) for record in manifest["records"] if record["disposition"] == "mapped"
                     and record["artifact_kind"] in KINDS and record["modality"] in modalities)
    # A bounded rerun must advance to unfinished or changed sources before
    # revisiting outputs that staging has already fully verified.
    queue.sort(key=lambda item: item[2]["formatter_status"] == "converted")
    report = {"schema_version": SCHEMA_VERSION, "started_at": now(), "state": "running",
              "source_root": str(source), "output_root": str(output), "selected_modalities": sorted(modalities),
              "selected_sources_this_run": len(queue), "staging": staging, "results": [], "ready_for_features": False}
    write_batch_report(output, report_dir, report, registry)
    for key, path, record in queue:
        if max_records is not None and len(report["results"]) >= max_records:
            report.update({"state": "partial", "stop_reason": "requested_record_limit"})
            break
        # Leave room for a complete temporary replacement plus the next source.
        required = int(min_free_gib * 2**30) + max(64 * 2**20, 2 * record["size_bytes"])
        if shutil.disk_usage(output).free < required:
            report.update({"state": "partial", "stop_reason": "insufficient_free_space_for_next_source"})
            break
        if progress:
            progress({"participant_key": key, "source_path": record["source_path"], "status": "starting"})
        if record["modality"] == "xsens":
            from .formatter import format_subject
            kwargs = {}
        else:
            from .garmin_formatter import format_subject
            kwargs = {"progress": lambda name, rows: progress({"participant_key": key,
                "source_path": record["source_path"], "source_table": name,
                "row_count": rows, "status": "table_verified"})} if progress else {}
        try:
            result = format_subject(path, source_ids=[record["source_id"]], **kwargs)["results"][0]
        except KeyboardInterrupt:
            report.update({"state": "partial", "stop_reason": "interrupted",
                           "interrupted_source": record["source_path"]})
            break
        current = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        updated = next(r for r in current["records"] if r["source_id"] == record["source_id"])
        sha = updated["sha256"] if result["status"] != "failed" else None
        state = update_content(sources[record["source_id"]], sha, now()) if sha else "unverified_due_to_failure"
        row = {"participant_key": key, "modality": record["modality"], "source_path": record["source_path"],
               **result, "content_state": state, "previous_sha256": previous_shas.get(record["source_id"]), "sha256": sha}
        report["results"].append(row)
        write_batch_report(output, report_dir, report, registry)
        if progress:
            progress(row)
    else:
        report["state"] = "completed_with_failures" if any(r["status"] == "failed" for r in report["results"]) else "completed"
    report["finished_at"] = now()
    write_batch_report(output, report_dir, report, registry)
    return report
