"""Adapt discovery into canonical JSON manifests without copying acquisitions."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from shared_code.io import atomic_json

from . import SCHEMA_VERSION
from .inventory import now
from .policy import POLICY
from .storage_names import garmin_export_name, garmin_targets, xsens_recording_name, xsens_targets


def clinical_metadata(subject: dict, files: list[dict]) -> dict:
    evaluations = [f for f in files if f["artifact_kind"] == "evaluation_workbook"]
    result = {
        "status": "not_collected_by_protocol" if subject["cohort"] == "SANI" else "not_available_in_snapshot",
        "source_ids": [f["source_id"] for f in evaluations], "lesion_hemisphere": None,
        "paretic_side": None, "non_paretic_side": None, "dominant_hand": None,
        "timeline_dates": {}, "side_derivation_rule": None,
        "hemisphere_evidence": [],
    }
    if not evaluations:
        return result
    # A second workbook is accepted only if all extracted fields agree.
    payloads = [f.get("clinical_metadata", {}) for f in evaluations]
    usable = all(f["disposition"] == "mapped" and m.get("status") == "available" for f, m in zip(evaluations, payloads))
    if not usable or any(m != payloads[0] for m in payloads[1:]):
        result["status"] = "needs_review"
        return result
    metadata = payloads[0]
    result.update({"status": "available", "lesion_hemisphere": metadata.get("lesion_hemisphere"),
                   "dominant_hand": metadata.get("dominant_hand"), "timeline_dates": metadata.get("timeline_dates", {})})
    result["hemisphere_evidence"] = [{"source_id": file["source_id"], **evidence}
        for file, payload in zip(evaluations, payloads) for evidence in payload.get("hemisphere_evidence", [])]
    if result["lesion_hemisphere"]:
        result["non_paretic_side"] = result["lesion_hemisphere"]
        result["paretic_side"] = "left" if result["lesion_hemisphere"] == "right" else "right"
        result["side_derivation_rule"] = "paretic_arm_contralateral_to_lesion_hemisphere"
    return result


def canonical_record(file: dict, clinical: dict) -> dict:
    parsed = file["parsed_name"] or {}
    condition = None
    side = parsed.get("anatomical_side")
    side_evidence = "source_filename" if side else None
    code = parsed.get("original_limb_code")
    identity_ok = "subject_identity_conflict" not in file["issues"]
    if identity_ok and code in POLICY["numeric_limb_codes"]:
        condition = POLICY["numeric_limb_codes"][code]
        side = clinical[f"{condition}_side"] if file["cohort"] == "PAZIENTI" else None
        if side:
            side_evidence = "clinical_workbook_and_contralateral_rule"
    elif code == "sim":
        condition = "simulated"
    elif file["cohort"] == "SANI" and code in {"dx", "sx"}:
        condition = "non_paretic"
    timepoint = parsed.get("timepoint")
    day = parsed.get("protocol_day")
    timeline_key = f"{timepoint}_D{day}" if timepoint and day is not None else timepoint
    # Exact matches only. Do not replace a week label by its first day's date.
    date = clinical["timeline_dates"].get(timeline_key) if identity_ok and file["cohort"] == "PAZIENTI" and "timepoint_folder_conflict" not in file["issues"] else None
    record = {
        "recording_id": "rec_" + file["source_id"][4:], "source_id": file["source_id"],
        "source_path": file["source_path"], "size_bytes": file["size_bytes"], "mtime_ns": file["mtime_ns"],
        "sha256": file["sha256"], "modality": file["modality"], "artifact_kind": file["artifact_kind"],
        "disposition": file["disposition"], "issues": sorted(set(file["issues"])),
        "activity": parsed.get("activity"), "task": parsed.get("task"), "timepoint": timepoint,
        "timepoint_kind": ("protocol_timepoint" if file["cohort"] == "PAZIENTI" else "session_label" if file["cohort"] == "SANI" else "source_label") if timepoint else None,
        "protocol_day": day, "session_id": (f"{timepoint}_d{day}" if day is not None and timepoint else timepoint),
        "acquisition_date": date, "anatomical_side": side, "side_evidence": side_evidence,
        "limb_condition": condition, "original_limb_code": code, "repetition": parsed.get("repetition"),
        "representation": parsed.get("representation"), "sheets": file["sheets"], "columns": file["columns"],
        "archive_members": file["archive_members"], "normalized_target": None,
        "formatter_status": "not_converted", "ready_for_features": False,
        "formatter_report": None,
        "selection_reason": file["selection_reason"], "selection_assumption": file["selection_assumption"],
        "duplicate_of_source_id": file["duplicate_of_source_id"], "selected_archive_members": file["selected_archive_members"],
        "expected_repetitions": POLICY["expected_task_repetitions"] if parsed.get("task") else None,
        "expected_marker_count": POLICY["expected_task_marker_count"] if parsed.get("task") and parsed.get("representation") == "full_recording" else None,
        "simulation_types_by_repetition": None,
    }
    if not record["session_id"] and file["cohort"] == "SANI" and file["artifact_kind"] == "xsens_workbook":
        record["session_id"] = "unlabelled_sani_acquisition"
    if file["disposition"] != "mapped":
        return record
    if file["artifact_kind"] == "xsens_workbook" and record["session_id"]:
        record["normalized_target"] = f"xsens/{record['session_id']}/{xsens_recording_name(record)}"
    elif file["artifact_kind"] in {"garmin_archive", "garmin_table"}:
        record["normalized_target"] = f"garmin/{garmin_export_name(record)}"
    elif file["modality"] == "motore":
        # The initial adapter preserves each export/session's source identity.
        # Session timestamps and grouping for MOTORE will be added by its decoder.
        record["normalized_target"] = f"{file['modality']}/{file['source_id']}"
    return record


def make_plan(inventory: dict, subject_filter: str | None = None) -> dict:
    if inventory.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported inventory schema version; expected {SCHEMA_VERSION}")
    subjects = inventory["subjects"]
    if subject_filter:
        subjects = [s for s in subjects if s["participant_key"] == subject_filter]
        if not subjects:
            raise ValueError("Unknown participant key; use e.g. PAZIENTI/FG91")
    collisions = {key for key, count in Counter(s["participant_key"] for s in subjects).items() if count > 1}
    planned = []
    for subject in subjects:
        if subject["participant_key"] in collisions:
            continue
        files = [f for f in inventory["files"] if f["cohort"] == subject["cohort"] and
                 f["source_subject_folder"] == subject["source_subject_folder"] and f["artifact_kind"] not in {"housekeeping", "symlink"}]
        clinical = clinical_metadata(subject, files)
        subject_document = {
            "schema_version": SCHEMA_VERSION, "cohort": subject["cohort"], "subject_id": subject["subject_id"],
            "participant_key": subject["participant_key"], "source_subject_folder": subject["source_subject_folder"],
            "clinical_metadata": clinical, "available_modalities": subject["modalities"],
            "snapshot_file_count": len(files), "download_completeness": "unknown",
        }
        manifest = {
            "schema_version": SCHEMA_VERSION, "participant_key": subject["participant_key"],
            "source_root": inventory["source_root"], "inventory_started_at": inventory["scan_started_at"],
            "inventory_finished_at": inventory["scan_finished_at"], "adapter_stage": "metadata_only",
            "import_policy": inventory["import_policy"], "records": [canonical_record(f, clinical) for f in files],
        }
        targets = {**xsens_targets(manifest["records"]), **garmin_targets(manifest["records"])}
        for record in manifest["records"]:
            if record["source_id"] in targets:
                record["normalized_target"] = targets[record["source_id"]]
        planned.append({"subject": subject_document, "manifest": manifest})
    return {"schema_version": SCHEMA_VERSION, "created_at": now(), "stage": "metadata_only",
            "subjects": planned, "blocked_participant_keys": sorted(collisions),
            "summary": {"subjects": len(planned), "records": sum(len(p["manifest"]["records"]) for p in planned),
                "dispositions": dict(sorted(Counter(r["disposition"] for p in planned for r in p["manifest"]["records"]).items()))}}


def write_plan(plan: dict, output: Path, *, hash_sources: bool = False) -> dict:
    """Stage JSON metadata only. Changed sources are deferred, never silently read."""
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("stage") != "metadata_only":
        raise ValueError("Unsupported adapter plan version or stage")
    output = output.resolve()
    source_roots = {Path(p["manifest"]["source_root"]).resolve() for p in plan["subjects"]}
    for root in source_roots:
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Standardized output must be separate from the source tree")
    written, deferred, retained = [], [], []
    for planned in plan["subjects"]:
        document, manifest = planned["subject"], planned["manifest"]
        root = Path(manifest["source_root"]).resolve()
        cohort, subject_id = document["cohort"], document["subject_id"]
        if cohort not in {"SANI", "PAZIENTI", "CONTROLLO PAZIENTI"} or Path(subject_id).name != subject_id or subject_id in {".", ".."}:
            raise ValueError("Unsafe participant destination")
        folder = output / cohort / subject_id
        if not folder.resolve().is_relative_to(output):
            raise ValueError("Participant destination escapes output root")
        if (folder / "manifest.json").is_file():
            from .layout import migrate_subject
            migrate_subject(folder, targets=xsens_targets(manifest["records"]))
            migrate_subject(folder, targets=garmin_targets(manifest["records"]), modality="garmin")
        try:
            previous = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            old_records = {r["source_id"]: r for r in previous["records"]} if previous.get("schema_version") == SCHEMA_VERSION else {}
        except (OSError, ValueError, KeyError, TypeError):
            previous, old_records = {}, {}
        for record in manifest["records"]:
            if record["disposition"] in {"deferred", "excluded"}:
                continue
            path = root / record["source_path"]
            try:
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    raise ValueError("Source escapes source root")
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != (record["size_bytes"], record["mtime_ns"]):
                    raise ValueError("Source fingerprint changed")
                old = old_records.get(record["source_id"], {})
                previous_tables = old.get("formatter_status") == "converted" or bool(old.get("formatter_report", {}).get("descriptor_path") if old.get("formatter_report") else False)
                if hash_sources or previous_tables:
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    after = path.stat()
                    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError("Source changed while hashing")
                    record["sha256"] = digest.hexdigest()
                if (previous_tables and record["disposition"] == "mapped" and record["sha256"] == old.get("sha256")
                    and record["sha256"] is not None and record["normalized_target"] == old.get("normalized_target")
                    and record["selected_archive_members"] == old.get("selected_archive_members")
                    and previous.get("import_policy") == manifest["import_policy"]):
                    # Preserve only flags whose source AND output checksums have
                    # just been verified. Size/mtime alone never justify reuse.
                    target = folder / record["normalized_target"]
                    if not target.resolve().is_relative_to(folder.resolve()):
                        raise ValueError("Conversion destination escapes participant")
                    if record["artifact_kind"] == "xsens_workbook":
                        from .formatter import cached_descriptor
                        descriptor = cached_descriptor(target, record["sha256"], manifest["import_policy"]["policy_id"])
                    else:
                        from .garmin_formatter import FORMATTER_VERSION, cached_tables
                        try:
                            descriptor = json.loads((target / "export.json").read_text(encoding="utf-8"))
                            valid = cached_tables(target, record["sha256"], manifest["import_policy"]["policy_id"], descriptor["selected_source_tables"], record)
                            if descriptor["formatter_version"] != FORMATTER_VERSION or len(valid) != len(descriptor["tables"]):
                                descriptor = None
                        except (OSError, ValueError, KeyError, TypeError):
                            descriptor = None
                    if descriptor:
                        record["formatter_status"] = old["formatter_status"]
                        record["formatter_report"] = old["formatter_report"]
                        manifest["adapter_stage"] = "metadata_and_tables"
                        retained.append(record["source_id"])
            except (OSError, ValueError):
                record["disposition"] = "deferred"
                record["normalized_target"] = None
                record["issues"] = sorted(set(record["issues"] + ["source_changed_since_inventory"]))
                deferred.append(record["source_path"])
        # Stale clinical evidence cannot be used to derive anatomical sides.
        clinical_sources = set(document["clinical_metadata"]["source_ids"])
        if any(r["source_id"] in clinical_sources and r["disposition"] == "deferred" for r in manifest["records"]):
            document["clinical_metadata"].update({"status": "needs_review", "lesion_hemisphere": None,
                "paretic_side": None, "non_paretic_side": None, "dominant_hand": None,
                "timeline_dates": {}, "side_derivation_rule": None})
            document["clinical_metadata"]["hemisphere_evidence"] = []
            for record in manifest["records"]:
                if record["side_evidence"] == "clinical_workbook_and_contralateral_rule":
                    record["anatomical_side"] = None
                    record["side_evidence"] = None
                record["acquisition_date"] = None
        atomic_json(folder / "subject.json", document)
        atomic_json(folder / "manifest.json", manifest)
        written.append(document["participant_key"])
    report = {"schema_version": SCHEMA_VERSION, "stage": "metadata_only", "written_participants": written,
              "deferred_sources": deferred, "blocked_participant_keys": plan["blocked_participant_keys"],
              "numeric_tables_converted": 0, "ready_for_feature_extraction": False}
    report["retained_verified_formatter_sources"] = retained
    atomic_json(output / "import_report.json", report)
    return report
