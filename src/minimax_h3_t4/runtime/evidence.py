"""Incremental evidence bundle for startup failures.

Every event is appended and fsynced before the next stage runs. ``quick_save``
rewrites a small zip plus a sha256 sidecar immediately. It does not wait for a
notebook pin, a model download, or a clean shutdown.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CURRENT: "EvidenceStore | None" = None
_MAX_TEXT = 8000
_MAX_PROCESSES = 40
_MAX_CMD = 160


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _bounded(value: Any, limit: int = _MAX_TEXT) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, sort_keys=True)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


class EvidenceStore:
    def __init__(self, run_dir: Path, base: Path | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.base = Path(base) if base is not None else self.run_dir.parent
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.base.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.status_path = self.run_dir / "status.json"
        self._status: dict[str, Any] = {"stage": "created", "updated_at": _now(), "error": None}
        self._write_json(self.status_path, self._status)

    @classmethod
    def open_default(cls) -> "EvidenceStore":
        base = Path(os.environ.get("H3_T4_EVIDENCE_DIR", str(Path(tempfile.gettempdir()) / "h3-t4-evidence")))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        store = cls(base / f"run-{stamp}", base=base)
        activate(store)
        store.event("evidence_open", run_dir=store.run_dir.name)
        return store

    def event(self, kind: str, **fields: Any) -> None:
        record = {"ts": _now(), "kind": kind, **fields}
        self._append(record)
        self._status = {
            "stage": kind,
            "updated_at": record["ts"],
            "error": self._status.get("error"),
            "last": {key: _bounded(value, 500) for key, value in fields.items()},
        }
        self._write_json(self.status_path, self._status)

    def record_error(self, exc: BaseException) -> None:
        text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._write_text(self.run_dir / "error.txt", _bounded(text, 20000))
        self._status["error"] = {"type": type(exc).__name__, "message": _bounded(str(exc), 2000)}
        self.event("error", error_type=type(exc).__name__, message=_bounded(str(exc), 2000))

    def record_history(self, payload: dict[str, Any], prompt_id: str | None = None) -> None:
        status = payload.get("status_str", payload.get("status"))
        messages = payload.get("messages", payload.get("error", payload.get("exception_message")))
        body = {
            "prompt_id": prompt_id or payload.get("prompt_id"),
            "status": status,
            "error": _bounded(messages),
        }
        self._write_json(self.run_dir / "history.json", body)
        self.event("history", status=_bounded(status, 200))

    def record_model_metadata(self, metadata: dict[str, Any]) -> None:
        safe = {
            "name": str(metadata.get("name", "")),
            "bytes": metadata.get("bytes"),
            "sha256": metadata.get("sha256"),
            "reusable": bool(metadata.get("reusable", False)),
            "reason": str(metadata.get("reason", "")),
        }
        self._write_json(self.run_dir / "model_metadata.json", safe)
        self.event("model_metadata", name=safe["name"], reusable=safe["reusable"], reason=safe["reason"])

    def write_payload(self, name: str, payload: dict[str, Any]) -> None:
        self._write_json(self.run_dir / name, payload)

    def snapshot_processes(self, *, limit: int = _MAX_PROCESSES) -> None:
        rows = _collect_processes(limit=limit)
        payload = {"count": len(rows), "limit": limit, "processes": rows}
        self._write_json(self.run_dir / "processes.json", payload)
        self.event("process_snapshot", count=len(rows))

    def quick_save(self) -> dict[str, Any]:
        """Rewrite the downloadable zip and sha256 sidecar from the current files."""

        zip_path = self.base / "evidence.zip"
        tmp_path = self.base / f".evidence-{os.getpid()}.zip"
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.run_dir.rglob("*")):
                if path.is_file() and path.suffix != ".zip":
                    archive.write(path, arcname=str(path.relative_to(self.run_dir)))
        os.replace(tmp_path, zip_path)
        digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        (self.base / "evidence.zip.sha256").write_text(digest + "  evidence.zip\n", encoding="utf-8")
        descriptor = {"zip": str(zip_path), "sha256": digest, "bytes": zip_path.stat().st_size}
        self._write_json(self.run_dir / "bundle.json", descriptor)
        return descriptor

    def _append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str, sort_keys=True) + "\n"
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        self._write_text(path, json.dumps(payload, default=str, indent=2, sort_keys=True))


def activate(store: EvidenceStore | None) -> None:
    global _CURRENT
    _CURRENT = store


def current() -> EvidenceStore | None:
    return _CURRENT


def note_failure(exc: BaseException) -> None:
    store = _CURRENT
    if store is None:
        return
    try:
        store.record_error(exc)
        store.snapshot_processes()
    except Exception:
        return


def finalize_active() -> dict[str, Any] | None:
    store = _CURRENT
    if store is None:
        return None
    try:
        return store.quick_save()
    except Exception:
        return None


def _collect_processes(*, limit: int) -> list[dict[str, Any]]:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return [{"pid": os.getpid(), "comm": "self", "cmd": "process-snapshot-unavailable"}]
    rows: list[dict[str, Any]] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        row = _process_row(entry)
        if row is not None:
            rows.append(row)
        if len(rows) >= 2000:
            break
    interesting = [
        row
        for row in rows
        if any(token in f"{row['comm']} {row['cmd']}" for token in ("H3T4", "ray", "python", "comfy"))
    ]
    chosen: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in interesting + rows:
        pid = int(row["pid"])
        if pid in seen:
            continue
        seen.add(pid)
        chosen.append(row)
        if len(chosen) >= limit:
            break
    return chosen


def _process_row(entry: Path) -> dict[str, Any] | None:
    try:
        comm = (entry / "comm").read_text(encoding="utf-8", errors="replace").strip()
        cmd = (entry / "cmdline").read_text(encoding="utf-8", errors="replace").replace("\x00", " ").strip()
    except OSError:
        return None
    return {"pid": int(entry.name), "comm": comm[:64], "cmd": cmd[:_MAX_CMD]}
