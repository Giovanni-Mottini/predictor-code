"""Atomic JSON output for importers and derived-artifact consumers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A same-directory replacement leaves either the old or the complete new file.
    fd, temporary = tempfile.mkstemp(prefix=".import-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

