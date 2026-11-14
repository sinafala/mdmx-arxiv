"""A tiny cross-process file lock (used when several chapters render at once).

Writes to shared files — ``results/manifest.json``, ``results/numbers.json``
— are wrapped in :func:`locked` so that two processes never interleave a
read-modify-write.  Works on macOS/Linux (``fcntl``) and Windows (``msvcrt``).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

if os.name == "nt":  # pragma: no cover - Windows
    import msvcrt

    def _lock(fh):
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock(fh):
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(fh):
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)

    def _unlock(fh):
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextmanager
def locked(path: Path):
    """Hold an exclusive lock on ``<path>.lock`` for the duration of the block."""
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as fh:
        _lock(fh)
        try:
            yield
        finally:
            _unlock(fh)
