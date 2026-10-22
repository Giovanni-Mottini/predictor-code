"""Fingerprint the current implementation using paths relative to the workspace."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .reader import contained_path


def implementation_hashes(paths: tuple[str, ...]) -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    return {relative: hashlib.sha256(contained_path(root, relative).read_bytes()).hexdigest()
            for relative in paths}
