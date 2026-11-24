"""Run the real two-rank initializer through ready and NCCL join, then stop.

No checkpoint is opened. A passing subprocess probe is not this admission.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .evidence import EvidenceStore


def summarize_actor_join(results: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(results, key=lambda item: int(item.get("rank", -1)))
    visibles = [str(item.get("cuda_visible_devices")) for item in ordered]
    uuids = [str(item.get("device_uuid") or "") for item in ordered]
    sums = [item.get("sum") for item in ordered]
    distinct = len(ordered) == 2 and visibles == ["0", "1"] and all(uuids) and len(set(uuids)) == 2
    collective_ok = distinct and sums == [5, 5] and all(item.get("weights_loaded") is False for item in ordered)
    return {
        "kind": "ray-h3t4-actor-admission",
        "collective_ok": collective_ok,
        "distinct_gpu_mapping": distinct,
        "gpu_mapping": {"distinct": distinct, "ranks": ordered},
        "sum": sums[0] if sums else None,
        "weights_loaded": False,
        "used_mock": False,
        "video_inference_claimed": False,
        "inference_memory_proven": False,
        "model_stage_allowed": False,
    }


def require_actor_admission_file(path: str) -> None:
    file_path = Path(path) if path else None
    if file_path is None or not file_path.is_file():
        raise RuntimeError("model stage blocked: Ray actor admission file is missing")
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != "ray-h3t4-actor-admission" or payload.get("collective_ok") is not True:
        raise RuntimeError("model stage blocked: Ray actor admission did not pass")
    if payload.get("weights_loaded") is not False or payload.get("inference_memory_proven") is not False:
        raise RuntimeError("model stage blocked: actor admission must not claim weights or a memory proof")


def run_actor_admission(*, evidence_dir: Path, initializer: Any | None = None) -> dict[str, Any]:
    """Create the real initializer's actors, join, record the result, and close them.

    ``initializer`` is only for tests. Production omits it and therefore uses
    ``H3T4Worker`` rather than a second worker implementation.
    """

    evidence_dir.mkdir(parents=True, exist_ok=True)
    os.environ["H3_T4_EVIDENCE_DIR"] = str(evidence_dir)
    if initializer is None:
        from ..loader_nodes import H3T4Initializer

        initializer = H3T4Initializer()
    group = None
    report: dict[str, Any]
    try:
        group = initializer.initialize(load_after=object(), manage_parent_memory=False, force_on_failure=True)[0]
        report = summarize_actor_join(list(getattr(initializer, "last_join", []) or []))
        report["placement"] = getattr(initializer, "last_placement", None)
    except BaseException as exc:
        report = summarize_actor_join(list(getattr(initializer, "last_join", []) or []))
        report["collective_ok"] = False
        report["error_type"] = type(exc).__name__
        report["message"] = str(exc)
        _write(evidence_dir / "actor_admission.json", report)
        _mirror_evidence(initializer, evidence_dir)
        raise
    else:
        _write(evidence_dir / "actor_admission.json", report)
        _mirror_evidence(initializer, evidence_dir)
        if not report["collective_ok"]:
            raise RuntimeError("Ray actor admission did not prove two independent GPUs and an allreduce sum of 5")
        return report
    finally:
        if group is not None and getattr(group, "alive", False):
            group.close(timeout_seconds=float(os.environ.get("H3_T4_CLEANUP_TIMEOUT_S", "20")), force=True)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _mirror_evidence(initializer: Any, evidence_dir: Path) -> None:
    store = getattr(initializer, "evidence", None)
    if isinstance(store, EvidenceStore):
        try:
            store.quick_save()
        except Exception:
            return
