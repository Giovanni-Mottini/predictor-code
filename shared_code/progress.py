"""Line-based execution progress, flushed to stderr for terminals and logs."""
from __future__ import annotations

import sys
import time
from typing import TextIO


class ProgressReporter:
    """Report phase-local work counts without changing stdout or artifacts."""

    def __init__(self, label: str, *, enabled: bool = True, stream: TextIO | None = None):
        self.label = label
        self.enabled = enabled
        self.stream = stream
        self.started = time.monotonic()

    def update(self, message: str, *, completed: int | None = None, total: int | None = None) -> None:
        if not self.enabled:
            return
        elapsed = max(0, int(time.monotonic() - self.started))
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        count = ""
        if completed is not None and total is not None:
            count = f" | {completed}/{total}"
            if total:
                count += f" ({100 * completed / total:.0f}%)"
        print(f"[{self.label}] {message}{count} | elapsed {duration}",
              file=self.stream if self.stream is not None else sys.stderr, flush=True)
