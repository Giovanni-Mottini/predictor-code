"""Snapshot inventory with explicit file-level findings and source fingerprints."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PurePosixPath
import re

from . import SCHEMA_VERSION
from .inspection import XSENS_SHEETS, inspect_file, read_patient_metadata
from .naming import COHORTS, filename_subject, parse_xsens_name
from .policy import POLICY, apply_selection


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_id(relative_path: str) -> str:
    # This identifies a source LOCATION. A content checksum is a separate field.
    return "src_" + hashlib.sha256(relative_path.encode()).hexdigest()[:24]


def issue(code: str, path: str, message: str, severity: str = "warning") -> dict:
    return {"code": code, "source_path": path, "severity": severity, "message": message}


def excluded_name(name: str) -> bool:
    return name == ".DS_Store" or name.startswith(("._", "~$"))


def pending_name(name: str) -> bool:
    return name.lower().endswith((".partial", ".part", ".tmp"))


def inspect_source(root: Path, path: Path, stat=None, *, logical_path: str | None = None) -> dict:
    """Inspect one real source, using the same interpretation as local discovery."""
    stat = stat or path.lstat()
    name = path.name
    rel = logical_path or path.relative_to(root).as_posix()
    parts = PurePosixPath(rel).parts
    cohort = parts[0]
    subject_folder = parts[1] if len(parts) >= 3 else None
    record = {
        "source_id": source_id(rel), "source_path": rel, "cohort": cohort,
        "subject_id": subject_folder.upper() if subject_folder else None,
        "source_subject_folder": subject_folder, "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "sha256": None, "extension": path.suffix.lower(),
        "modality": "unknown", "artifact_kind": "unknown", "parsed_name": None,
        "inspection_status": "not_inspected", "sheets": [], "columns": [], "archive_members": [],
        "issues": [], "disposition": "preserve_only",
    }
    if excluded_name(name) or path.is_symlink():
        record["disposition"] = "excluded"
        record["artifact_kind"] = "housekeeping" if not path.is_symlink() else "symlink"
    elif pending_name(name):
        record["disposition"] = "deferred"
        record["artifact_kind"] = "transfer_temporary"
        record["issues"].append("incomplete_transfer")
    else:
        if stat.st_size == 0:
            record["issues"].append("empty_file")
        modality_folder = parts[2].casefold() if len(parts) >= 4 else ""
        folder_modality = {"xsens": "xsens", "garmin": "garmin", "motore": "motore"}.get(modality_folder, "unknown")
        record["modality"] = folder_modality
        record.update(inspect_file(path))
        if record["inspection_status"] == "inspection_error":
            record["issues"].append("inspection_error")
        if path.suffix.lower() == ".xlsx" and set(XSENS_SHEETS).issubset(record["sheets"]):
            record["modality"] = "xsens"
            record["artifact_kind"] = "xsens_workbook"
            record["parsed_name"] = parse_xsens_name(name)
            if folder_modality != "xsens":
                record["issues"].append("modality_folder_conflict")
            parsed = record["parsed_name"]
            if parsed is None:
                record["issues"].append("unrecognized_xsens_name")
            else:
                expected = filename_subject(subject_folder or "", cohort)
                if parsed["filename_subject_id"] != expected:
                    record["issues"].append("subject_identity_conflict")
                folder_times = {f"T{int(p[1:])}" for p in parts[2:-1] if re.fullmatch(r"t\d+", p, re.IGNORECASE)}
                if folder_times and (len(folder_times) > 1 or parsed["timepoint"] not in folder_times):
                    record["issues"].append("timepoint_folder_conflict")
                if parsed["timepoint"] is None:
                    record["issues"].append("timepoint_unresolved")
                if parsed["original_limb_code"] == "sim":
                    record["issues"].append("simulated_side_unresolved")
        elif path.suffix.lower() == ".xlsx" and "TIMELINE_01" in record["sheets"] and len(parts) == 3 and cohort == "PAZIENTI":
            record["modality"] = "clinical"
            record["artifact_kind"] = "evaluation_workbook"
            record["clinical_metadata"] = read_patient_metadata(path, filename_subject(subject_folder, cohort))
            if record["clinical_metadata"]["status"] != "available":
                record["issues"].append("clinical_metadata_unresolved")
        elif folder_modality == "garmin" and path.suffix.lower() in {".csv", ".zip"}:
            record["artifact_kind"] = "garmin_archive" if path.suffix.lower() == ".zip" else "garmin_table"
            for member in record["archive_members"]:
                if not member["safe_path"] or not member["unique_path"]:
                    record["issues"].append("unsafe_or_duplicate_archive_member")
                if member["inspection_status"] == "inspection_error":
                    record["issues"].append("archive_member_header_error")
        elif folder_modality == "xsens" and path.suffix.lower() == ".mvn":
            record["artifact_kind"] = "xsens_native"
        elif folder_modality == "motore":
            record["artifact_kind"] = "motore_" + (path.suffix.lower().lstrip(".") or "other")
        elif folder_modality == "xsens" and path.suffix.lower() == ".xlsx":
            record["issues"].append("xsens_sheet_profile_unresolved")
        try:
            after = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                record["issues"].append("source_changed_during_scan")
        except OSError:
            record["issues"].append("source_changed_during_scan")
        if record["artifact_kind"] in {"xsens_workbook", "evaluation_workbook", "garmin_archive", "garmin_table"} or folder_modality == "motore":
            record["disposition"] = "mapped"
        if record["issues"]:
            record["disposition"] = "needs_review"
    record["issues"] = sorted(set(record["issues"]))
    return record


def scan(root: Path) -> dict:
    root = root.resolve()
    if not root.is_dir() or not any((root / cohort).is_dir() for cohort in COHORTS):
        raise ValueError("Source root must contain at least one of the three cohort folders")
    started = now()
    files, subjects, issues = [], [], []
    for cohort in COHORTS:
        folder = root / cohort
        if not folder.is_dir():
            issues.append(issue("cohort_not_available", cohort, "Cohort directory is not present in this snapshot", "info"))
            continue
        for subject in sorted(folder.iterdir(), key=lambda p: p.name):
            if not subject.is_dir() or subject.is_symlink():
                continue
            subjects.append({"cohort": cohort, "subject_id": subject.name.upper(), "source_subject_folder": subject.name,
                             "participant_key": cohort + "/" + subject.name.upper()})
        # os.walk does not follow linked directories and preserves original spelling.
        for directory, dirs, names in os.walk(folder, followlinks=False):
            for linked in [d for d in dirs if (Path(directory) / d).is_symlink()]:
                rel = (Path(directory) / linked).relative_to(root).as_posix()
                issues.append(issue("symlink_not_followed", rel, "Linked directory excluded from source discovery", "info"))
                dirs.remove(linked)
            dirs.sort()
            for name in sorted(names):
                path = Path(directory) / name
                rel = path.relative_to(root).as_posix()
                try:
                    stat = path.lstat()
                except OSError:
                    issues.append(issue("source_disappeared", rel, "Source disappeared during discovery"))
                    continue
                files.append(inspect_source(root, path, stat))
    return finish_inventory(root, started, files, subjects, issues)


def finish_inventory(root: Path, started: str, files: list[dict], subjects: list[dict],
                     issues: list[dict], *, scope="local_download_snapshot") -> dict:
    """Apply cross-file selection to inspected local or remotely cached metadata."""
    # Case-only collisions must not overwrite one another in canonical output.
    case_paths = defaultdict(list)
    logical_recordings = defaultdict(list)
    for file in files:
        case_paths[file["source_path"].casefold()].append(file)
        parsed = file["parsed_name"]
        if parsed:
            key = (file["cohort"], file["subject_id"], parsed["timepoint"], parsed["activity"], parsed["original_limb_code"])
            logical_recordings[key].append(file)
    for group in case_paths.values():
        if len(group) > 1:
            for file in group:
                file["issues"].append("case_normalization_collision")
                file["disposition"] = "needs_review"
    for group in logical_recordings.values():
        representations = {f["parsed_name"]["representation"] for f in group}
        if len(group) > 1:
            code = "overlapping_recording_representations" if len(representations) > 1 else "multiple_recordings_same_key"
            # Individual repetitions with distinct indices are not duplicates.
            reps = [f["parsed_name"]["repetition"] for f in group]
            if representations == {"individual_repetition"} and len(set(reps)) == len(reps):
                continue
            for file in group:
                file["issues"].append(code)
                file["disposition"] = "needs_review"
    # Matching table names indicate a candidate overlap, not verified equality.
    for archive in [f for f in files if f["artifact_kind"] == "garmin_archive"]:
        member_names = {Path(m["path"]).name.casefold() for m in archive["archive_members"]}
        loose = [f for f in files if f["artifact_kind"] == "garmin_table" and
                 (f["cohort"], f["subject_id"]) == (archive["cohort"], archive["subject_id"]) and
                 Path(f["source_path"]).name.casefold() in member_names]
        if loose:
            for file in [archive, *loose]:
                file["issues"].append("archive_and_loose_table_overlap_candidate")
                file["disposition"] = "needs_review"
    apply_selection(files)
    for file in files:
        for code in sorted(set(file["issues"])):
            severity = "error" if code in {"subject_identity_conflict", "modality_folder_conflict", "timepoint_folder_conflict",
                "inspection_error", "empty_file", "source_changed_during_scan", "unsafe_or_duplicate_archive_member"} else "warning"
            issues.append(issue(code, file["source_path"], code.replace("_", " "), severity))
    for subject in subjects:
        visible = [f for f in files if f["cohort"] == subject["cohort"] and f["subject_id"] == subject["subject_id"] and f["artifact_kind"] not in {"housekeeping", "symlink"}]
        subject["file_count"] = len(visible)
        subject["size_bytes"] = sum(f["size_bytes"] for f in visible)
        subject["modalities"] = sorted({f["modality"] for f in visible if f["modality"] != "unknown"})
        subject["evaluation_sources"] = [f["source_path"] for f in visible if f["artifact_kind"] == "evaluation_workbook"]
        subject["evaluation_status"] = ("available" if subject["evaluation_sources"] else
            "not_collected_by_protocol" if subject["cohort"] == "SANI" else "not_available_in_snapshot")
        if not visible:
            issues.append(issue("no_subject_files_available", subject["participant_key"], "No non-housekeeping files currently available; download completeness is unknown", "info"))
    counts = Counter(s["participant_key"] for s in subjects)
    for key, count in counts.items():
        if count > 1:
            issues.append(issue("subject_normalization_collision", key, "Multiple source folders normalize to the same participant", "error"))
    return {"schema_version": SCHEMA_VERSION, "import_policy": POLICY, "source_root": str(root), "scan_started_at": started,
            "scan_finished_at": now(), "scan_scope": scope, "files": files,
            "subjects": subjects, "issues": issues, "summary": summarize(files, subjects, issues)}


def summarize(files: list[dict], subjects: list[dict], issues: list[dict]) -> dict:
    data = [f for f in files if f["artifact_kind"] not in {"housekeeping", "symlink"}]
    return {
        "data_file_count": len(data), "excluded_file_count": len(files) - len(data),
        "size_bytes": sum(f["size_bytes"] for f in data), "subject_count": len(subjects),
        "empty_subject_count": sum(s["file_count"] == 0 for s in subjects),
        "cohorts": {c: {"subjects": sum(s["cohort"] == c for s in subjects),
            "subjects_with_files": sum(s["cohort"] == c and s["file_count"] > 0 for s in subjects),
            "files": sum(f["cohort"] == c for f in data),
            "size_bytes": sum(f["size_bytes"] for f in data if f["cohort"] == c)} for c in COHORTS},
        "dispositions": dict(sorted(Counter(f["disposition"] for f in files).items())),
        "artifact_kinds": dict(sorted(Counter(f["artifact_kind"] for f in data).items())),
        "issue_counts": dict(sorted(Counter(i["code"] for i in issues).items())),
        "archive_member_count": sum(len(f["archive_members"]) for f in data),
        "ignored_data_file_count": sum(f["disposition"] == "excluded" for f in data),
        "selected_file_count": sum(f["disposition"] == "mapped" for f in data),
        "selection_assumptions_pending_verification": sum(f["selection_assumption"] is not None for f in data),
    }
