"""Resumable transfer of one official file. Nothing calls this unless enable is set."""

from __future__ import annotations

import hashlib
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

CHUNK = 1024 * 1024


def transfer_file(
    *,
    url: str,
    dest: Path,
    expected_bytes: int,
    expected_sha256: str,
    deadline: float,
) -> dict[str, Any]:
    """Download one file or reuse a verified copy. One attempt, no retry loop."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == expected_bytes and _sha256(dest) == expected_sha256:
        return {"ok": True, "action": "reused", "path": str(dest), "bytes": expected_bytes, "sha256": expected_sha256, "started_transfer": False}
    partial = dest.with_name(dest.name + ".partial")
    existing = partial.stat().st_size if partial.is_file() else 0
    if existing > expected_bytes:
        partial.unlink()
        existing = 0
    if time.monotonic() >= deadline:
        return {"ok": False, "action": "not_started", "status": "DEADLINE", "started_transfer": False, "bytes": existing}
    try:
        status, chunks = _open(url, existing, expected_bytes)
    except Exception as exc:
        return {"ok": False, "action": "failed", "status": "HTTP_ERROR", "error_type": type(exc).__name__, "message": str(exc)[:300], "started_transfer": False, "bytes": existing}
    if status == "RANGE_IGNORED":
        return {"ok": False, "action": "failed", "status": "RANGE_IGNORED", "started_transfer": False, "bytes": existing}
    started = time.monotonic()
    written = existing
    try:
        with partial.open("ab" if existing else "wb") as handle:
            for chunk in chunks:
                if time.monotonic() >= deadline:
                    handle.flush()
                    os.fsync(handle.fileno())
                    return {"ok": False, "action": "cancelled", "status": "DEADLINE", "started_transfer": True, "bytes": written, "path": str(partial)}
                elapsed = time.monotonic() - started
                if elapsed >= 2 and written < expected_bytes:
                    rate = (written - existing) / max(elapsed, 0.001)
                    remaining = deadline - time.monotonic()
                    if rate * remaining < (expected_bytes - written):
                        handle.flush()
                        os.fsync(handle.fileno())
                        return {"ok": False, "action": "cancelled", "status": "TOO_SLOW", "started_transfer": True, "bytes": written, "bytes_per_second": rate}
                if not chunk:
                    break
                room = expected_bytes - written
                if room <= 0:
                    break
                handle.write(chunk[:room])
                written += min(len(chunk), room)
                if written > expected_bytes:
                    break
            handle.flush()
            os.fsync(handle.fileno())
    except Exception as exc:
        return {"ok": False, "action": "failed", "status": "HTTP_ERROR", "error_type": type(exc).__name__, "message": str(exc)[:300], "started_transfer": True, "bytes": written}
    if written != expected_bytes:
        return {"ok": False, "action": "failed", "status": "SHORT_FILE", "started_transfer": True, "bytes": written, "expected_bytes": expected_bytes}
    digest = _sha256(partial)
    if digest != expected_sha256:
        partial.unlink(missing_ok=True)
        return {"ok": False, "action": "failed", "status": "HASH_MISMATCH", "started_transfer": True, "sha256": digest}
    os.replace(partial, dest)
    return {"ok": True, "action": "transferred", "path": str(dest), "bytes": written, "sha256": digest, "started_transfer": True}


def _open(url: str, existing: int, expected_bytes: int) -> tuple[str, Any]:
    current = url
    headers = {"User-Agent": "h3-transfer"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    for _ in range(6):
        request = urllib.request.Request(current, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as exc:
            response = exc
            if exc.code not in (301, 302, 303, 307, 308):
                raise
        status = getattr(response, "status", None) or getattr(response, "code", None)
        response_headers = {key.lower(): value for key, value in response.headers.items()}
        if status in (301, 302, 303, 307, 308):
            location = response_headers.get("location")
            response.close()
            if not location:
                raise RuntimeError("redirect without location")
            current = urljoin(current, location)
            continue
        if existing and status != 206:
            response.close()
            return "RANGE_IGNORED", ()
        if not existing and status not in (200, 206):
            response.close()
            raise RuntimeError(f"unexpected status {status}")
        length = response_headers.get("content-length")
        if length is not None:
            coming = int(length)
            if existing and status == 206 and coming != expected_bytes - existing:
                response.close()
                raise RuntimeError(f"content-length {coming} does not match the remaining bytes")
            if not existing and coming != expected_bytes:
                response.close()
                raise RuntimeError(f"content-length {coming} does not match expected {expected_bytes}")

        def chunks():
            while True:
                block = response.read(CHUNK)
                if not block:
                    response.close()
                    return
                yield block

        return "OK", chunks()
    raise RuntimeError("too many redirects")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
