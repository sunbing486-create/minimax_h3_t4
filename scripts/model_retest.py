#!/usr/bin/env python3
"""Staged MiniMax-H3 retest. The default stage is preflight and never downloads.

actor-admission uses H3T4Initializer and H3T4Worker, then closes the actors
before any checkpoint is opened. model-load stays off unless
H3_T4_ALLOW_MODEL_LOAD=1, every official file is reusable, and the actor
admission in this process passed. A subprocess NCCL probe is not that proof.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimax_h3_t4.runtime.actor_admission import run_actor_admission
from minimax_h3_t4.model_catalog import (
    OFFICIAL_TOTAL_BYTES,
    disk_report,
    inventory_local_models,
    memory_budget,
    official_download_list,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight, Ray actor admission, or an explicit model-load gate")
    parser.add_argument("--stage", default="preflight", choices=("preflight", "actor-admission", "model-load"))
    parser.add_argument("--search", action="append", default=[])
    parser.add_argument("--evidence-dir", default=os.environ.get("H3_T4_EVIDENCE_DIR", ""))
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)
    roots = [Path(item) for item in args.search] or _default_roots()
    evidence = Path(args.evidence_dir) if args.evidence_dir else Path.cwd() / "h3-model-retest"
    evidence.mkdir(parents=True, exist_ok=True)
    report = _preflight(roots)
    if args.stage == "preflight":
        return _emit(report, evidence, args.report)
    if args.stage == "actor-admission":
        return _emit(_admission(evidence, report), evidence, args.report)
    return _emit(_model_gate(evidence, report), evidence, args.report)


def _default_roots() -> list[Path]:
    raw = os.environ.get("H3_T4_MODEL_ROOTS", "")
    if raw:
        return [Path(item) for item in raw.split(os.pathsep) if item]
    return [Path("/kaggle/input"), Path("/kaggle/working")]


def _preflight(roots: list[Path]) -> dict:
    inventory = inventory_local_models(roots)
    disks = [disk_report(path) for path in roots]
    device_bytes = _device_bytes()
    logical = _logical_memory_hint()
    budget = memory_budget(logical_memory_bytes=logical, device_memory_bytes=device_bytes)
    return {
        "stage": "preflight",
        "inventory": inventory,
        "disks": disks,
        "memory_budget": budget,
        "download_list_if_missing": official_download_list() if inventory["status"] == "MISSING" else [],
        "official_total_bytes": OFFICIAL_TOTAL_BYTES,
        "inference_allowed": False,
        "download": False,
        "video_inference_claimed": False,
    }


def _admission(evidence: Path, preflight: dict) -> dict:
    devices = _cuda_count()
    if devices != 2:
        preflight.update(
            {
                "stage": "actor-admission",
                "status": "NOT_RUN",
                "reason": f"need 2 CUDA devices before starting H3T4Worker, found {devices}",
                "inference_allowed": False,
            }
        )
        return preflight
    try:
        admission = run_actor_admission(evidence_dir=evidence)
    except Exception as exc:
        preflight.update(
            {
                "stage": "actor-admission",
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "inference_allowed": False,
                "video_inference_claimed": False,
            }
        )
        return preflight
    preflight.update(
        {
            "stage": "actor-admission",
            "status": "ADMITTED" if admission.get("collective_ok") else "FAILED",
            "actor_admission": admission,
            "inference_allowed": False,
            "inference_memory_proven": False,
            "video_inference_claimed": False,
        }
    )
    return preflight


def _model_gate(evidence: Path, preflight: dict) -> dict:
    """Do not load weights unless this process repeats actor admission and files match."""

    preflight["stage"] = "model-load"
    preflight["inference_allowed"] = False
    preflight["video_inference_claimed"] = False
    if os.environ.get("H3_T4_ALLOW_MODEL_LOAD") != "1":
        preflight["status"] = "NOT_RUN"
        preflight["reason"] = "H3_T4_ALLOW_MODEL_LOAD is not 1"
        return preflight
    if preflight["inventory"]["status"] != "REUSABLE":
        preflight["status"] = "MISSING"
        preflight["reason"] = "official files are not all present with matching sha256"
        return preflight
    if _cuda_count() != 2:
        preflight["status"] = "NOT_RUN"
        preflight["reason"] = "model load was not started because two CUDA devices are not visible"
        return preflight
    try:
        admission = run_actor_admission(evidence_dir=evidence)
    except Exception as exc:
        preflight["status"] = "FAILED"
        preflight["error_type"] = type(exc).__name__
        preflight["message"] = str(exc)
        return preflight
    preflight["actor_admission"] = admission
    if not admission.get("collective_ok"):
        preflight["status"] = "FAILED"
        preflight["reason"] = "actor admission failed before load"
        return preflight
    preflight["status"] = "NOT_RUN"
    preflight["reason"] = (
        "actor admission passed and files match, but this command does not launch T2V/I2V. "
        "Memory headroom is not proven. Start ComfyUI only in a separate explicit step."
    )
    return preflight


def _device_bytes() -> int:
    try:
        import torch
    except Exception:
        return 0
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return 0
        return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return 0


def _cuda_count() -> int:
    try:
        import torch
    except Exception:
        return 0
    try:
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _logical_memory_hint() -> int:
    try:
        page = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return 0
    for line in page.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def _emit(report: dict, evidence: Path, report_path: str) -> int:
    target = Path(report_path) if report_path else evidence / "preflight.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(target), "stage": report.get("stage"), "inventory": report.get("inventory", {}).get("status"), "inference_allowed": False}, sort_keys=True))
    if report.get("stage") == "actor-admission" and report.get("status") not in {"ADMITTED", "NOT_RUN"}:
        return 2
    if report.get("stage") == "actor-admission" and report.get("status") == "NOT_RUN":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
