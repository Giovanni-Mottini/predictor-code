"""Manifest-based selection and checksum-verified reads; no filename parsing."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from . import SCHEMA_VERSION
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xsens_code.reader import XsensRecording


class DataContractError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def contained_path(root: Path, relative: str) -> Path:
    """Reject absolute/traversing paths and symlinks escaping the data root."""
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise DataContractError("unsafe_path", f"Not a contained relative path: {relative}")
    result = (root / rel).resolve()
    if not result.is_relative_to(root.resolve()):
        raise DataContractError("unsafe_path", f"Path escapes {root}: {relative}")
    return result


def document(path: Path) -> tuple[dict, str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise DataContractError("invalid_document", f"Expected JSON object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


@dataclass
class SubjectReader:
    root: Path
    subject: dict = field(init=False)
    manifest: dict = field(init=False)
    metadata_sha256: dict = field(init=False)

    def __post_init__(self):
        self.root = Path(self.root).resolve()
        self.subject, subject_hash = document(self.root / "subject.json")
        self.manifest, manifest_hash = document(self.root / "manifest.json")
        for doc in (self.subject, self.manifest):
            if doc.get("schema_version") != SCHEMA_VERSION:
                raise DataContractError("unsupported_schema", "Reader requires schema 0.2.0")
        key = f"{self.subject.get('cohort')}/{self.subject.get('subject_id')}"
        if (self.subject.get("cohort") not in {"SANI", "PAZIENTI", "CONTROLLO PAZIENTI"}
                or self.subject.get("participant_key") != key
                or self.manifest.get("participant_key") != key):
            raise DataContractError("identity_mismatch", "Subject and manifest identities disagree")
        records = self.manifest.get("records")
        if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
            raise DataContractError("invalid_manifest", "Manifest records must be objects")
        ids = [r.get("source_id") for r in records]
        if None in ids or len(ids) != len(set(ids)):
            raise DataContractError("duplicate_source_id", "Manifest source IDs must be unique")
        self.metadata_sha256 = {"subject.json": subject_hash, "manifest.json": manifest_hash}

    @property
    def participant_key(self) -> str:
        return self.subject["participant_key"]

    def select(self, *, modality: str, sessions=None, timepoints=None, tasks=None,
               limb_conditions=None, anatomical_sides=None) -> list[dict]:
        """Select only mapped sources by canonical metadata, preserving source order.

        Conversion state is checked when opening a recording so pending sources
        are reported as errors rather than silently disappearing from coverage.
        ``None`` means no filter; an empty collection selects nothing.
        """
        filters = {"session_id": sessions, "timepoint": timepoints, "task": tasks,
                   "limb_condition": limb_conditions, "anatomical_side": anatomical_sides}
        return [r for r in self.manifest["records"]
                if r.get("modality") == modality and r.get("disposition") == "mapped"
                and all(values is None or r.get(key) in values for key, values in filters.items())]

    def open_xsens(self, record: dict) -> "XsensRecording":
        """Open one selected recording through the separate Xsens reader."""
        from xsens_code.reader import open_recording
        return open_recording(self, record)
