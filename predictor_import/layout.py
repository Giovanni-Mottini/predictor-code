"""Migrate derived Xsens/Garmin names without rebuilding tables or reading raw files."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile

from . import SCHEMA_VERSION
from .storage_names import garmin_table_names, garmin_targets, xsens_table_names, xsens_targets


def contained(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe derived path: {relative}")
    target = root / path
    if target == root or not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError(f"Derived path escapes its root or is a symlink: {relative}")
    return target


def migrate_subject(subject: Path, *, targets: dict[str, str] | None = None,
                    dry_run: bool = False, modality: str = "xsens") -> dict:
    """Publish renamed directories before the manifest; preserve originals until commit.

    Hard links avoid duplicating tables on the same filesystem. A copy fallback
    supports other filesystems. Every table is checksum-verified before and after
    staging. Failed publication restores the original layout and manifest.
    """
    from .adapter import atomic_json
    from .formatter import cached_descriptor, checksum

    if modality not in {"xsens", "garmin"}:
        raise ValueError("Layout migration supports Xsens or Garmin")
    label = "Xsens" if modality == "xsens" else "Garmin"
    descriptor_file = "recording.json" if modality == "xsens" else "export.json"
    source_field = "source_sheet" if modality == "xsens" else "source_table"
    kinds = {"xsens_workbook"} if modality == "xsens" else {"garmin_archive", "garmin_table"}
    target_names = xsens_targets if modality == "xsens" else garmin_targets
    table_names = xsens_table_names if modality == "xsens" else garmin_table_names
    migrated_field = "recordings_migrated" if modality == "xsens" else "exports_migrated"

    def verified_descriptor(directory, descriptor, record):
        if modality == "xsens":
            return cached_descriptor(directory, descriptor["source_sha256"], manifest["import_policy"]["policy_id"])
        from .garmin_formatter import FORMATTER_VERSION, cached_tables
        candidate = json.loads((directory / descriptor_file).read_text(encoding="utf-8"))
        valid = cached_tables(directory, descriptor["source_sha256"], manifest["import_policy"]["policy_id"],
                              descriptor["selected_source_tables"], record)
        if (candidate.get("formatter_version") != FORMATTER_VERSION or not candidate.get("tables")
                or len(valid) != len(candidate["tables"])
                or any(t.get("roundtrip_verified") is not True for t in candidate["tables"])):
            return None
        return candidate

    subject = subject.resolve()
    manifest_path = subject / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema for {label} layout migration")
    raw = Path(manifest["source_root"]).resolve()
    if subject == raw or subject.is_relative_to(raw) or raw.is_relative_to(subject):
        raise ValueError("Layout migration must stay outside raw sources")
    targets = target_names(manifest["records"]) if targets is None else targets
    updated = deepcopy(manifest)
    changes, jobs = [], []
    for record in updated["records"]:
        destination = targets.get(record["source_id"])
        old_relative = record.get("normalized_target")
        if not destination or not old_relative or record.get("artifact_kind") not in kinds:
            continue
        old, new = contained(subject, old_relative), contained(subject, destination)
        descriptor, renamed = None, []
        if old.exists():
            try:
                descriptor = json.loads((old / descriptor_file).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                if old == new:
                    # Current-layout cache damage is handled by the formatter.
                    continue
                raise
            # Garmin partial exports use the whole selected set so filenames
            # stay stable when the remaining tables are converted later.
            titles = [t[source_field] for t in descriptor["tables"]] if modality == "xsens" else descriptor["selected_source_tables"]
            names = table_names(titles)
            renamed = [{"old": t["file"], "new": names[t[source_field]], "sha256": t["sha256"]}
                       for t in descriptor["tables"]]
        elif old != new and record["formatter_status"] == "converted":
            raise ValueError(f"Converted {label} directory is missing: {old_relative}")
        if old_relative == destination and all(t["old"] == t["new"] for t in renamed):
            continue
        if descriptor is not None:
            if (descriptor["source_id"] != record["source_id"]
                    or descriptor["source_path"] != record["source_path"]
                    or (record.get("sha256") and descriptor["source_sha256"] != record["sha256"])):
                raise ValueError(f"{label} descriptor identity/revision mismatch: {old_relative}")
            verified = verified_descriptor(old, descriptor, record)
            if verified is None:
                raise ValueError(f"Cannot migrate unverified {label} output: {old_relative}")
            expected = {descriptor_file, *(t["old"] for t in renamed)}
            if {p.name for p in old.iterdir()} != expected:
                raise ValueError(f"Unexpected files in {label} directory: {old_relative}")
            for table in renamed:
                contained(old, table["old"])
                contained(new, table["new"])
            descriptor = deepcopy(descriptor)
            for table in descriptor["tables"]:
                table["file"] = names[table[source_field]]
        # Refuse occupied destinations. A fully verified staged destination from
        # an interrupted migration can be safely resumed without overwriting it.
        reuse = False
        if new != old and new.exists():
            candidate = verified_descriptor(new, descriptor, record) if descriptor else None
            if candidate != descriptor or candidate is None:
                raise ValueError(f"{label} destination already exists: {destination}")
            if {p.name for p in new.iterdir()} != {descriptor_file, *(t["new"] for t in renamed)}:
                raise ValueError(f"Unexpected files in {label} destination: {destination}")
            reuse = True
        record["normalized_target"] = destination
        if (record.get("formatter_report") or {}).get("descriptor_path"):
            record["formatter_report"]["descriptor_path"] = destination + "/" + descriptor_file
        changes.append({"source_id": record["source_id"], "old": old_relative,
                        "new": destination, "tables": renamed})
        if descriptor is not None:
            jobs.append({"old": old, "new": new, "descriptor": descriptor,
                         "tables": renamed, "reuse": reuse})
    # Reject collisions before writing any directories or metadata.
    destinations = [r["normalized_target"] for r in updated["records"] if r.get("normalized_target")]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Layout migration would create duplicate destinations")
    result = {"participant_key": manifest["participant_key"], "dry_run": dry_run,
              "records_changed": len(changes), migrated_field: len(jobs),
              "tables_rebuilt": 0, "changes": changes}
    if dry_run or not changes:
        return result
    staged, published, committed = [], [], False
    try:
        for job in jobs:
            if job["reuse"]:
                continue
            job["new"].parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".layout-", dir=job["new"].parent))
            job["temporary"] = temporary
            staged.append(job)
            for table in job["tables"]:
                source, target = job["old"] / table["old"], temporary / table["new"]
                try:
                    target.hardlink_to(source)
                except OSError:
                    shutil.copy2(source, target)
                if checksum(target) != table["sha256"]:
                    raise ValueError(f"Table changed while migrating: {source}")
            atomic_json(temporary / descriptor_file, job["descriptor"])
        for job in staged:
            old, new = job["old"], job["new"]
            if old == new:
                backup = Path(tempfile.mkdtemp(prefix=".layout-previous-", dir=old.parent))
                backup.rmdir()
                old.rename(backup)
                job["backup"] = backup
            elif new.exists():
                raise ValueError(f"{label} destination appeared during migration: {new}")
            published.append(job)
            job["temporary"].rename(new)
            job["published"] = True
        atomic_json(manifest_path, updated)
        committed = True
    finally:
        if not committed:
            for job in reversed(published):
                if job.get("published"):
                    shutil.rmtree(job["new"])
                if job.get("backup"):
                    job["backup"].rename(job["old"])
        for job in staged:
            if job["temporary"].exists():
                shutil.rmtree(job["temporary"])
    # All active references now point to verified new files.
    for job in jobs:
        obsolete = job.get("backup") if job["old"] == job["new"] else job["old"]
        if obsolete and obsolete.exists():
            shutil.rmtree(obsolete)
    return result


def migrate_layout(output: Path, *, dry_run: bool = False, modality: str = "xsens") -> dict:
    """Accept a standardized root or one participant directory."""
    output = output.resolve()
    subjects = [output] if (output / "manifest.json").is_file() else sorted(p.parent for p in output.glob("*/*/manifest.json"))
    if not subjects:
        raise ValueError("No standardized participant manifests found")
    # Validate the whole selection before starting mutations.
    previews = [migrate_subject(subject, dry_run=True, modality=modality) for subject in subjects]
    results = previews if dry_run else [migrate_subject(subject, modality=modality) for subject in subjects]
    migrated_field = "recordings_migrated" if modality == "xsens" else "exports_migrated"
    return {"dry_run": dry_run, "subjects_checked": len(subjects),
            "records_changed": sum(r["records_changed"] for r in results),
            migrated_field: sum(r[migrated_field] for r in results),
            "tables_rebuilt": 0, "results": results}
