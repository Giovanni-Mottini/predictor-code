"""Reviewed import choices; original references remain in the inventory."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path, PurePosixPath

POLICY = json.loads(Path(__file__).with_name("policy.json").read_text(encoding="utf-8"))

# Missing interpretation fields need not prevent a lossless format conversion.
NONBLOCKING = {
    "timepoint_unresolved", "simulated_side_unresolved", "overlapping_recording_representations",
    "multiple_recordings_same_key", "archive_and_loose_table_overlap_candidate",
}


def apply_selection(files: list[dict]) -> None:
    """Select one provisional representation, recording the alternatives and reason."""
    for file in files:
        file.update({"selection_reason": None, "selection_assumption": None,
                     "duplicate_of_source_id": None, "selected_archive_members": None})
        file.pop("replaces_source_ids", None)
        if file["disposition"] in {"excluded", "deferred"}:
            continue
        parsed = file["parsed_name"] or {}
        parts = PurePosixPath(file["source_path"]).parts
        in_motore = len(parts) > 2 and parts[2].casefold() == "motore"
        if in_motore or file["modality"] == "motore" or parsed.get("activity") in POLICY["ignored_xsens_activities"]:
            file["disposition"] = "excluded"
            file["selection_reason"] = "motore_out_of_scope"
            continue
        if set(file["issues"]) - NONBLOCKING:
            file["disposition"] = "excluded"
            file["selection_reason"] = "flagged_conflict_ignored"
            continue
        if file["disposition"] == "needs_review":
            file["disposition"] = "mapped"
        if file["disposition"] == "mapped":
            file["selection_reason"] = "recognized_source"

    groups = defaultdict(list)
    for file in files:
        parsed = file["parsed_name"]
        if parsed and file["disposition"] == "mapped":
            # SANI is a single experimental acquisition in the supplied protocol.
            # This grouping assumption does NOT invent a source timepoint/date.
            session = "sani_acquisition" if file["cohort"] == "SANI" else parsed["timepoint"]
            key = (file["cohort"], file["subject_id"], session, parsed.get("protocol_day"), parsed["activity"], parsed["original_limb_code"])
            groups[key].append(file)
    for group in groups.values():
        # reexports.json: in a declared visit the re-export replaces the original recording.
        reexports = sorted((f for f in group if f.get("export_role") == "declared_reexport"), key=lambda f: f["source_path"])
        if reexports and len(group) > 1:
            chosen = reexports[0]
            chosen["selection_reason"] = "declared_reexport_replaces_original"
            chosen["replaces_source_ids"] = sorted(f["source_id"] for f in group if f is not chosen)
            for other in group:
                if other is not chosen:
                    other.update({"disposition": "excluded", "selection_reason": "superseded_by_declared_reexport",
                                  "duplicate_of_source_id": chosen["source_id"]})
            continue
        full = sorted((f for f in group if f["parsed_name"]["representation"] == "full_recording"), key=lambda f: f["source_path"])
        if full:
            chosen = full[0]
            if len(group) > 1:
                chosen["selection_reason"] = "preferred_full_recording"
                chosen["selection_assumption"] = POLICY["overlap_assumption"]
                for other in group:
                    if other is not chosen:
                        other.update({"disposition": "excluded", "selection_reason": "assumed_duplicate_full_recording",
                                      "selection_assumption": POLICY["overlap_assumption"], "duplicate_of_source_id": chosen["source_id"]})
        else:
            reps = defaultdict(list)
            for file in group:
                rep = file["parsed_name"]["repetition"]
                if rep is not None and not 1 <= rep <= POLICY["expected_task_repetitions"]:
                    file["issues"].append("unexpected_repetition_index")
                    file.update({"disposition": "excluded", "selection_reason": "flagged_conflict_ignored"})
                else:
                    reps[rep].append(file)
            for alternatives in reps.values():
                if len(alternatives) > 1:
                    chosen = min(alternatives, key=lambda f: f["source_path"])
                    chosen["selection_assumption"] = POLICY["overlap_assumption"]
                    for other in alternatives:
                        if other is not chosen:
                            other.update({"disposition": "excluded", "selection_reason": "assumed_duplicate_repetition",
                                          "selection_assumption": POLICY["overlap_assumption"], "duplicate_of_source_id": chosen["source_id"]})

    # Prefer archive tables provisionally. A loose table not present in a usable
    # archive remains selected. Partial ZIPs do not hide readable loose exports.
    tables = defaultdict(list)
    for file in files:
        if file["disposition"] != "mapped" or file["modality"] != "garmin":
            continue
        participant = (file["cohort"], file["subject_id"])
        if file["artifact_kind"] == "garmin_archive":
            file["selected_archive_members"] = []
            for member in file["archive_members"]:
                if member["safe_path"] and member["unique_path"] and member["path"].lower().endswith(".csv"):
                    tables[(*participant, PurePosixPath(member["path"]).name.casefold())].append((file, member["path"]))
        elif file["artifact_kind"] == "garmin_table":
            tables[(*participant, PurePosixPath(file["source_path"]).name.casefold())].append((file, None))
    for alternatives in tables.values():
        chosen, member = min(alternatives, key=lambda pair: (pair[1] is None, pair[0]["source_path"], pair[1] or ""))
        if member is not None:
            chosen["selected_archive_members"].append(member)
        if len(alternatives) > 1:
            chosen["selection_reason"] = "preferred_archive_representation" if member else "preferred_loose_representation"
            chosen["selection_assumption"] = POLICY["overlap_assumption"]
            for other, other_member in alternatives:
                if other is chosen and other_member == member:
                    continue
                other["selection_assumption"] = POLICY["overlap_assumption"]
                if other_member is None:
                    other.update({"disposition": "excluded", "selection_reason": "assumed_duplicate_archive_table",
                                  "duplicate_of_source_id": chosen["source_id"]})
    for file in files:
        if file["artifact_kind"] == "garmin_archive" and file["disposition"] == "mapped" and not file["selected_archive_members"]:
            file.update({"disposition": "excluded", "selection_reason": "archive_has_no_selected_tables"})


def exclude_hd_without_reexport(files: list[dict], root: Path) -> None:
    """Normal only in declared visits: an original left without a replacement must not be HD.

    Only originals of declared visits (reexports.json) are read, so other files cost nothing.
    An unreadable quality is flagged and the recording kept, as for any other source.
    """
    from .xlsx_repair import processing_quality

    for file in files:
        if file.get("export_role") != "original" or file["disposition"] != "mapped" or file["artifact_kind"] != "xsens_workbook":
            continue
        file["processing_quality"] = processing_quality(Path(root) / file["source_path"])
        if file["processing_quality"] == "HD":
            file.update({"disposition": "excluded", "selection_reason": "hd_without_normal_reexport"})
        elif file["processing_quality"] is None:
            file["issues"].append("processing_quality_unreadable")
