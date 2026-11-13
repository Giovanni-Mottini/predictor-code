"""Dataset-owner decisions: a re-export folder replaces the first export of named visits.

The table lives beside this module as reexports.json (git-ignored, like side_overrides.json,
because it names real participants). Without the file nothing changes. Shape:

    {"PAZIENTI/FG91": {
        "folder": "XSens_new",            # first folder under the participant
        "timepoints": ["T1", "T2"],       # visits whose recordings it replaces
        "decided_on": "...", "decided_by": "...", "reason": "...",
        "markers_from_original": {        # optional, per re-export source path
            "<re-export source_path>": {
                "reexport_sha256": "...", "original_source_path": "...", "original_sha256": "...",
                "frames": [5 frames], "verified_on": "...", "check": "..."}}}}

Rules applied by the importer:
- in a declared visit, a recording found in the declared folder replaces the recording of
  the same activity and arm in the regular Xsens folder;
- an original in a declared visit that has no replacement is left out when its export is
  processed in HD (Normal only); other originals stay;
- files in the declared folder outside the declared visits stay flagged and ignored;
- a marker substitution is used only for the exact re-export and original named, after the
  formatter re-checks the original's checksum, markers, recording time and frame rate.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

TABLE = Path(__file__).with_name("reexports.json")
REEXPORTS: dict = json.loads(TABLE.read_text(encoding="utf-8")) if TABLE.exists() else {}


def participant_key(source_path: str) -> str | None:
    parts = PurePosixPath(source_path).parts
    return f"{parts[0]}/{parts[1].upper()}" if len(parts) >= 3 else None


def declaration(source_path: str) -> dict | None:
    return REEXPORTS.get(participant_key(source_path) or "")


def in_declared_folder(source_path: str) -> bool:
    entry, parts = declaration(source_path), PurePosixPath(source_path).parts
    return bool(entry) and len(parts) >= 4 and parts[2].casefold() == entry["folder"].casefold()


def declared_visit(source_path: str, timepoint: str | None) -> bool:
    entry = declaration(source_path)
    return bool(entry) and timepoint is not None and timepoint in entry["timepoints"]


def marker_source(source_path: str) -> dict | None:
    entry = declaration(source_path) or {}
    found = entry.get("markers_from_original", {}).get(source_path)
    return dict(found, decided_on=entry.get("decided_on")) if found else None
