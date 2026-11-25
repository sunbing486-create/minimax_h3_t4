#!/usr/bin/env python3
"""One GPU session: install, optionally transfer the UNet, load it, stop.

The default does not transfer and does not load. H3_T4_ENABLE_TRANSFER=1
is the explicit switch. The whole command has one deadline and exits by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from minimax_h3_t4.model_catalog import inventory_local_models
from minimax_h3_t4.runtime.checkpoint_budget import host_available_bytes
from minimax_h3_t4.runtime.unet_load import UNET_SPEC, choose_weight_dir, required_files, run_unet_load_only
from scripts.prepare_model_stage import audit_imports, clone_comfy, install_model_deps
from scripts.run_actor_session import seal_evidence, show_session_evidence

SESSION_BUDGET_S = 1080
LOAD_RESERVE_S = 240


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded UNet load-only session")
    parser.add_argument("--evidence-dir", default="h3-unet-load")
    parser.add_argument("--budget-s", type=float, default=SESSION_BUDGET_S)
    parser.add_argument("--enable-transfer", action="store_true")
    parser.add_argument("--comfy-dir", default=os.environ.get("H3_T4_COMFY_DIR", "/tmp/ComfyUI-h3"))
    args = parser.parse_args(argv)
    evidence = Path(args.evidence_dir)
    code = run_session(evidence, budget_s=args.budget_s, enable_transfer=args.enable_transfer, comfy_dir=Path(args.comfy_dir))
    show_session_evidence(evidence)
    return code


def run_session(evidence: Path, *, budget_s: float, enable_transfer: bool, comfy_dir: Path) -> int:
    evidence.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + budget_s
    state: dict[str, object] = {
        "stage": "unet-load-session",
        "status": "RUNNING",
        "download": False,
        "started_transfer": False,
        "unet_loaded": False,
        "video_inference_claimed": False,
        "inference_memory_proven": False,
        "actor_admission_passed": False,
        "model_stage_ready": False,
        "required_files": required_files(),
        "budget_s": budget_s,
        "snapshot": "pre-seal",
        "bundle_status": "NOT_SEALED",
    }
    exit_code = 2
    try:
        if time.monotonic() >= deadline:
            state["status"] = "NOT_RUN_BUDGET"
            return exit_code
        state["clone"] = clone_comfy(comfy_dir)
        state["install"] = install_model_deps()
        state["audit"] = audit_imports(comfy_dir)
        _write(evidence / "session-status.json", state)
        destination = choose_weight_dir([Path("/tmp"), Path("/kaggle/tmp")], int(UNET_SPEC["bytes"]), shutil.disk_usage)
        state["weight_dir"] = None if destination is None else str(destination)
        if enable_transfer and destination is None:
            state["status"] = "STOP_NO_SPACE"
            state["reason"] = "no writable disk can hold the UNet plus 1GiB; /kaggle/working is smaller than the file"
            return exit_code
        if enable_transfer and destination is not None:
            if deadline - time.monotonic() < LOAD_RESERVE_S:
                state["status"] = "STOP_BUDGET_BEFORE_DOWNLOAD"
                return exit_code
            from minimax_h3_t4.runtime.official_transfer import transfer_file

            state["transfer"] = transfer_file(
                url=f"https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/4b53a35191437de05370da548e59aeabe2a5a09a/{UNET_SPEC['path']}",
                dest=destination / Path(str(UNET_SPEC["path"])).name,
                expected_bytes=int(UNET_SPEC["bytes"]),
                expected_sha256=str(UNET_SPEC["sha256"]),
                deadline=deadline - LOAD_RESERVE_S,
            )
            state["started_transfer"] = state["transfer"].get("started_transfer") is True
            state["download"] = state["started_transfer"]
            if state["transfer"].get("ok") is not True:
                state["status"] = "STOP_TRANSFER"
                return exit_code
        else:
            state["transfer"] = {"ok": False, "started_transfer": False, "status": "NOT_ENABLED"}
        roots = [Path("/tmp"), Path("/kaggle/working"), Path("/kaggle/input")]
        inventory = inventory_local_models(roots)
        unet = next(row for row in inventory["files"] if row["name"] == "unet")
        state["load"] = run_unet_load_only(
            unet_path=unet.get("local_path"),
            cuda_count=_cuda_count(),
            allow=enable_transfer and unet.get("reusable") is True,
            available_bytes=host_available_bytes(),
        )
        state["unet_loaded"] = state["load"].get("unet_loaded") is True
        state["status"] = "UNET_LOADED" if state["unet_loaded"] else "UNET_NOT_LOADED"
        exit_code = 0 if state["unet_loaded"] else 2
        return exit_code
    except Exception as exc:
        state["status"] = "FAILED"
        state["error_type"] = type(exc).__name__
        state["message"] = str(exc)[:500]
        return 2
    finally:
        state["elapsed_s"] = round(budget_s - (deadline - time.monotonic()), 3)
        state["snapshot"] = "pre-seal"
        _write(evidence / "session-status.json", state)
        try:
            state["bundle"] = seal_evidence(evidence)
            state["bundle_status"] = "SEALED"
        except Exception as exc:
            state["bundle_status"] = "FAILED"
            state["bundle_error"] = {"error_type": type(exc).__name__, "message": str(exc)[:300]}
        state["snapshot"] = "final"
        _write(evidence / "session-status.json", state)
        _write(evidence.parent / f"{evidence.name}-manifest.json", state)


def _cuda_count() -> int:
    try:
        import torch
    except Exception:
        return 0
    try:
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


if __name__ == "__main__":
    raise SystemExit(main())
