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
import signal
import subprocess
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
from minimax_h3_t4.loader_nodes import build_unet_runtime_env, probe_worker_imports
from minimax_h3_t4.runtime.unet_load import UNET_SPEC, choose_weight_dir, required_files, run_unet_load_only, runtime_gate, worker_probe_blockers
from scripts.bootstrap_actor_admission import apply_plan, import_check
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
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--sleep-child", action="store_true")
    args = parser.parse_args(argv)
    evidence = Path(args.evidence_dir)
    if args.sleep_child:
        _sleep_child(evidence)
        return 0
    if args.child:
        code = run_session(evidence, budget_s=args.budget_s, enable_transfer=args.enable_transfer, comfy_dir=Path(args.comfy_dir))
        show_session_evidence(evidence)
        return code
    code = run_watchdog(evidence, budget_s=args.budget_s, enable_transfer=args.enable_transfer, comfy_dir=Path(args.comfy_dir), sleep_child=False)
    show_session_evidence(evidence)
    return code


def run_watchdog(evidence: Path, *, budget_s: float, enable_transfer: bool, comfy_dir: Path, sleep_child: bool = False, memory_reader=host_available_bytes) -> int:
    """Kill the session process group when the deadline or low memory hits.

    The child finally block does not run after SIGKILL. This parent writes the
    zip and manifest after the kill.
    """

    evidence.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--evidence-dir", str(evidence), "--budget-s", str(budget_s), "--comfy-dir", str(comfy_dir)]
    if enable_transfer:
        command.append("--enable-transfer")
    command.append("--sleep-child" if sleep_child else "--child")
    process = subprocess.Popen(command, start_new_session=True)
    deadline = time.monotonic() + budget_s
    reason = None
    while process.poll() is None:
        available = memory_reader()
        if time.monotonic() >= deadline:
            reason = "DEADLINE"
            break
        if available < 2 * 1024**3:
            reason = "STOP_LOW_MEMORY"
            break
        time.sleep(0.2)
    if reason is not None:
        _kill_group(process.pid)
        try:
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ChildProcessError):
            process.kill()
        state = _read_state(evidence)
        state.update({
            "status": reason,
            "watchdog_killed": True,
            "child_pid": process.pid,
            "unet_loaded": False,
            "weight_alignment": "unknown",
            "video_inference_claimed": False,
            "remaining_processes": _matching_processes(evidence),
        })
        _seal(evidence, state)
        return 2
    state = _read_state(evidence)
    if state.get("bundle_status") != "SEALED":
        _seal(evidence, state)
    return int(process.returncode or 0)


def _sleep_child(evidence: Path) -> None:
    evidence.mkdir(parents=True, exist_ok=True)
    grandchild = subprocess.Popen(["sleep", "60"])
    _write(evidence / "grandchild.pid", {"pid": grandchild.pid})
    _write(evidence / "session-status.json", {"status": "BLOCKED", "unet_loaded": False, "weight_alignment": "unknown"})
    time.sleep(60)


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        return


def run_session(evidence: Path, *, budget_s: float, enable_transfer: bool, comfy_dir: Path) -> int:
    evidence.mkdir(parents=True, exist_ok=True)
    os.environ["H3_T4_EVIDENCE_DIR"] = str(evidence)
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
        "weight_alignment": "unknown",
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
        _write(evidence / "session-status.json", state)
        state["clone"] = clone_comfy(comfy_dir)
        _write(evidence / "session-status.json", state)
        state["install"] = install_model_deps()
        _write(evidence / "session-status.json", state)
        state["actor_deps"] = apply_plan()
        state["actor_imports"] = import_check()
        state["audit"] = audit_imports(comfy_dir)
        state["preflight"] = runtime_gate(cuda_count=_cuda_count(), available_bytes=host_available_bytes())
        window = deadline - time.monotonic() - LOAD_RESERVE_S
        state["required_bytes_per_second"] = None if window <= 0 else int(UNET_SPEC["bytes"]) / window
        destination = choose_weight_dir([Path("/tmp"), Path("/kaggle/tmp")], int(UNET_SPEC["bytes"]), shutil.disk_usage)
        state["weight_dir"] = None if destination is None else str(destination)
        probe_dir = destination or Path("/tmp")
        try:
            state["worker_probe"] = probe_worker_imports(
                runtime_env=build_unet_runtime_env(comfy_dir),
                checkpoint_dir=probe_dir,
            )
        except Exception as exc:
            state["worker_probe"] = {"ok": False, "error_type": type(exc).__name__, "message": str(exc)[:500], "imports": [], "cuda_count": None, "checkpoint_dir_writable": False}
        versions = (state.get("actor_imports") or {}).get("installed_versions") or {}
        state["worker_probe"]["expected_versions"] = {name: versions.get(name) for name in ("torch", "torchvision", "ray", "xfuser", "yunchang")}
        blockers = preflight_blockers(state["actor_deps"], state["actor_imports"], state["audit"], state["preflight"], state["worker_probe"])
        if destination is None and enable_transfer:
            blockers.append("no writable disk can hold the UNet plus 1GiB; /kaggle/working is smaller than the file")
        if enable_transfer and window <= 0:
            blockers.append("remaining time is inside the load reserve")
        state["preflight_blockers"] = blockers
        _write(evidence / "session-status.json", state)
        if blockers:
            state["status"] = "STOP_PREFLIGHT"
            state["reason"] = "; ".join(blockers)
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
            _write(evidence / "session-status.json", state)
            if state["transfer"].get("ok") is not True:
                state["status"] = "STOP_TRANSFER"
                return exit_code
            if deadline - time.monotonic() < LOAD_RESERVE_S:
                state["status"] = "STOP_BUDGET_BEFORE_LOAD"
                state["reason"] = "transfer finished inside the load reserve; load was not started"
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
            deadline=deadline,
            comfy_root=str(comfy_dir),
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
        state["rank_logs"] = _rank_logs(evidence)
        state["snapshot"] = "pre-seal"
        _write(evidence / "session-status.json", state)
        load_result = _load_result(state)
        _write(evidence / "load-result.json", load_result)
        inline = {
            "source": "runner-cell-output",
            "sealed_zip_is_the_original_artifact": True,
            "if_zip_is_lost_label_a_rebuild": "notebook-cell-reconstruction",
            "not_the_sealed_zip": True,
            "load_result": load_result,
        }
        _write(evidence / "inline-evidence.json", inline)
        try:
            state["bundle"] = seal_evidence(evidence)
            state["bundle_status"] = "SEALED"
        except Exception as exc:
            state["bundle_status"] = "FAILED"
            state["bundle_error"] = {"error_type": type(exc).__name__, "message": str(exc)[:300]}
        state["snapshot"] = "final"
        _write(evidence / "session-status.json", state)
        _write(evidence.parent / f"{evidence.name}-manifest.json", state)


def preflight_blockers(
    actor_deps: dict[str, object],
    actor_imports: dict[str, object],
    audit: dict[str, object],
    gate: dict[str, object],
    worker_probe: dict[str, object] | None = None,
) -> list[str]:
    """Stop before any weight transfer when the real worker stack is not ready."""

    blockers: list[str] = []
    if actor_deps.get("ok") is not True:
        blockers.append("xfuser/yunchang install did not succeed")
    if actor_imports.get("materials_ready_for_gpu_stage") is not True:
        blockers.append("worker pins are not xfuser==0.4.5 and yunchang==0.6.4")
    if audit.get("ok") is not True:
        blockers.append("comfy import audit is false")
    extra = gate.get("blockers")
    if isinstance(extra, list):
        blockers.extend(str(item) for item in extra)
    if worker_probe is not None:
        blockers.extend(worker_probe_blockers(worker_probe))
    return blockers


def _rank_logs(evidence: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(evidence.rglob("rank*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        rows.append({"path": str(path.relative_to(evidence)), "text": text[-4000:]})
    return rows


def _load_result(state: dict[str, object]) -> dict[str, object]:
    load = state.get("load") if isinstance(state.get("load"), dict) else {}
    transfer = state.get("transfer") if isinstance(state.get("transfer"), dict) else {}
    return {
        "source": "unet-load-runner",
        "actor_result_json": "not produced by the UNet runner; the actor-admission runner writes that file",
        "status": state.get("status"),
        "unet_loaded": state.get("unet_loaded") is True,
        "weight_alignment": "unknown",
        "error_type": load.get("error_type"),
        "message": str(load.get("message") or "")[:4000],
        "worker_diagnostics": load.get("worker_diagnostics"),
        "cleanup": load.get("cleanup"),
        "rank_logs": state.get("rank_logs") or [],
        "transfer": {
            key: transfer.get(key)
            for key in ("ok", "action", "bytes", "sha256", "bytes_per_second", "download_elapsed_s", "required_bytes_per_second", "started_transfer", "path", "status")
        },
        "required_bytes_per_second": state.get("required_bytes_per_second"),
        "bytes_per_second_is_measured_only_when_present": True,
        "video_inference_claimed": False,
    }


def _read_state(evidence: Path) -> dict[str, object]:
    path = evidence / "session-status.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _seal(evidence: Path, state: dict[str, object]) -> None:
    state["snapshot"] = "final"
    _write(evidence / "session-status.json", state)
    try:
        state["bundle"] = seal_evidence(evidence)
        state["bundle_status"] = "SEALED"
    except Exception as exc:
        state["bundle_status"] = "FAILED"
        state["bundle_error"] = {"error_type": type(exc).__name__, "message": str(exc)[:300]}
    _write(evidence / "session-status.json", state)
    _write(evidence.parent / f"{evidence.name}-manifest.json", state)


def _matching_processes(evidence: Path) -> list[dict[str, object]]:
    me = os.getpid()
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in {me, os.getppid()}:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        except OSError:
            continue
        if "H3T4Worker" in command or str(evidence) in command:
            pid = int(entry.name)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
            found.append({"pid": pid, "cmd": command[:160]})
    return found


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
