"""Verified derived Garmin tables keyed by raw CSV content, not container name."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .adapter import atomic_json
from .formatter import checksum


def stream_checksum(open_stream) -> str:
    digest = hashlib.sha256()
    with open_stream() as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_folder(root: Path, version: str, raw_sha: str) -> Path:
    if not raw_sha or len(raw_sha) != 64 or any(c not in "0123456789abcdef" for c in raw_sha):
        raise ValueError("Invalid content checksum")
    folder = root / version / raw_sha
    if not folder.resolve().is_relative_to(root.resolve()):
        raise ValueError("Content cache escapes its root")
    return folder


def load_table(root: Path, version: str, raw_sha: str, headers: list[str]) -> tuple[dict, Path] | None:
    folder = cache_folder(root, version, raw_sha)
    try:
        metadata = json.loads((folder / "table.json").read_text(encoding="utf-8"))
        table = metadata["table"]
        file = folder / "table.parquet"
        if metadata["formatter_version"] != version or metadata["source_csv_sha256"] != raw_sha or table["source_headers"] != headers:
            return None
        if not table["roundtrip_verified"] or file.is_symlink() or checksum(file) != table["sha256"]:
            return None
        return table, file
    except (OSError, ValueError, KeyError, TypeError):
        return None


def link_table(source: Path, target: Path) -> None:
    # Link derived artifacts only. Raw inputs are never linked or modified.
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def save_table(root: Path, version: str, raw_sha: str, table: dict, file: Path) -> None:
    if load_table(root, version, raw_sha, table["source_headers"]):
        return
    folder = cache_folder(root, version, raw_sha)
    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".cache-", dir=folder.parent))
    try:
        link_table(file, temporary / "table.parquet")
        atomic_json(temporary / "table.json", {"formatter_version": version, "source_csv_sha256": raw_sha, "table": table})
        backup = folder.with_name(folder.name + ".previous")
        if backup.exists():
            raise ValueError("Unresolved content cache backup")
        if folder.exists():
            folder.rename(backup)
        try:
            temporary.rename(folder)
        except OSError:
            if backup.exists():
                backup.rename(folder)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
