"""Structured inventory and source-selection metadata; presentation stays local."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .adapter import atomic_json


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            values = {k: json.dumps(row.get(k), ensure_ascii=False) if isinstance(row.get(k), (list, dict)) else row.get(k) for k in fields}
            writer.writerow(values)




def write_inventory(inventory: dict, output: Path) -> None:
    root = Path(inventory["source_root"]).resolve()
    output = output.resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Inventory reports must be outside the raw source tree")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "inventory.json", inventory)
    atomic_json(output / "summary.json", inventory["summary"])
    write_csv(output / "files.csv", inventory["files"], ["source_id", "source_path", "cohort", "subject_id", "modality", "artifact_kind", "size_bytes", "mtime_ns", "sha256", "disposition", "selection_reason", "selection_assumption", "duplicate_of_source_id", "selected_archive_members", "inspection_status", "parsed_name", "sheets", "columns", "issues"])
    write_csv(output / "subjects.csv", inventory["subjects"], ["participant_key", "source_subject_folder", "file_count", "size_bytes", "modalities", "evaluation_status", "evaluation_sources"])
    write_csv(output / "issues.csv", inventory["issues"], ["severity", "code", "source_path", "message"])
