"""Local weight reuse. This module never downloads."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def inspect_local_weight(
    *,
    name: str,
    path: str,
    expected_sha256: str | None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    """Reuse a weight only when the file exists and its sha256 matches.

    A missing hash, a missing file, or a mismatch is not reusable. Nothing is
    fetched to fill the gap.
    """

    file_path = Path(path)
    if not file_path.is_file():
        return {"name": name, "reusable": False, "reason": "missing", "bytes": None, "sha256": None}
    size = file_path.stat().st_size
    if expected_bytes is not None and size != int(expected_bytes):
        return {"name": name, "reusable": False, "reason": "size_mismatch", "bytes": size, "sha256": None}
    if not expected_sha256:
        return {"name": name, "reusable": False, "reason": "sha256_required", "bytes": size, "sha256": None}
    digest = _sha256(file_path)
    if digest != expected_sha256.lower():
        return {"name": name, "reusable": False, "reason": "sha256_mismatch", "bytes": size, "sha256": digest}
    return {"name": name, "reusable": True, "reason": "match", "bytes": size, "sha256": digest}


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
