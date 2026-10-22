"""Shared metadata fields and protected, atomic derived-artifact writes."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile

from .reader import DataContractError, SubjectReader

CANONICAL_FIELDS = ("recording_id", "source_id", "source_path", "session_id", "timepoint",
                    "timepoint_kind", "protocol_day", "acquisition_date", "task", "anatomical_side",
                    "side_evidence", "limb_condition", "selection_reason", "selection_assumption")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_output(output: Path, readers: list[SubjectReader]) -> Path:
    output = output.resolve()
    for reader in readers:
        for protected in (reader.root.parents[1], Path(reader.manifest["source_root"]).resolve()):
            if output.is_relative_to(protected):
                raise DataContractError("unsafe_output", f"Artifact output must stay outside {protected}")
    return output


def atomic_csv(path: Path, rows: list[dict], fields: tuple[str, ...]):
    fd, temporary = tempfile.mkstemp(prefix=".features-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path: Path, value: str):
    fd, temporary = tempfile.mkstemp(prefix=".features-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
