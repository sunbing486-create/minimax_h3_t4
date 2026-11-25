#!/usr/bin/env python3
"""Serial MiniMax-H3 smoke: text encoder, UNet sample, video VAE, silent MP4.

The default does not transfer. ``--enable-transfer`` is the explicit switch.
The old 1080 second UNet-load cap is not this session's deadline. The hard
deadline is the sum of the stage caps in ``video_budget``.
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

from minimax_h3_t4.loader_nodes import build_unet_runtime_env, probe_worker_imports
from minimax_h3_t4.model_catalog import OFFICIAL_MODELS, OFFICIAL_REVISION, OFFICIAL_REPO
from minimax_h3_t4.runtime.checkpoint_budget import host_available_bytes
from minimax_h3_t4.runtime.unet_load import choose_weight_dir, persist_process_sample
from minimax_h3_t4.runtime.video_budget import (
    MUX_CAP_S,
    SAMPLE_CAP_S,
    TEXT_CAP_S,
    UNET_LOAD_CAP_S,
    VAE_CAP_S,
    budget_report,
    sample_spec,
    seconds_at_rate,
    smoke_session_budget_s,
)
from minimax_h3_t4.runtime.video_stages import ffmpeg_status, missing_tokenizer_files
from scripts.bootstrap_actor_admission import apply_plan, import_check
from scripts.prepare_model_stage import audit_imports, clone_comfy, install_model_deps
from scripts.run_actor_session import _link_path, seal_evidence, show_session_evidence
from scripts.run_unet_load_session import preflight_blockers

SEAL_RESERVE_S = 180
STAGE_LIMITS_S = {
    "text": TEXT_CAP_S,
    "sample": UNET_LOAD_CAP_S + SAMPLE_CAP_S,
    "decode": VAE_CAP_S + MUX_CAP_S,
}
LOW_AVAILABLE_BYTES = 2 * 1024**3
SPECS = {item["name"]: item for item in OFFICIAL_MODELS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded serial MiniMax-H3 video smoke")
    parser.add_argument("--evidence-dir", default="h3-video-smoke")
    parser.add_argument("--budget-s", type=float, default=smoke_session_budget_s())
    parser.add_argument("--enable-transfer", action="store_true")
    parser.add_argument("--comfy-dir", default=os.environ.get("H3_T4_COMFY_DIR", "/tmp/ComfyUI-h3"))
    parser.add_argument("--mode", choices=("smoke", "trained"), default="smoke")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--sleep-child", action="store_true")
    parser.add_argument("--stage", choices=("text", "sample", "decode"))
    parser.add_argument("--weight", default="")
    parser.add_argument("--conditioning", default="")
    parser.add_argument("--latent", default="")
    parser.add_argument("--mp4", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evidence = Path(args.evidence_dir)
    if args.stage:
        return _run_stage(args, evidence)
    if args.sleep_child:
        _sleep_child(evidence)
        return 0
    if args.child:
        code = run_session(evidence, budget_s=args.budget_s, enable_transfer=args.enable_transfer, comfy_dir=Path(args.comfy_dir), mode=args.mode)
        _show_video_files(evidence)
        return code
    code = run_watchdog(evidence, budget_s=args.budget_s, enable_transfer=args.enable_transfer, comfy_dir=Path(args.comfy_dir), mode=args.mode)
    _show_video_files(evidence)
    return code


def run_watchdog(
    evidence: Path,
    *,
    budget_s: float,
    enable_transfer: bool,
    comfy_dir: Path,
    mode: str = "smoke",
    sleep_child: bool = False,
    memory_reader=host_available_bytes,
) -> int:
    evidence.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--evidence-dir",
        str(evidence),
        "--budget-s",
        str(budget_s),
        "--comfy-dir",
        str(comfy_dir),
        "--mode",
        mode,
    ]
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
        if available < LOW_AVAILABLE_BYTES:
            reason = "STOP_LOW_MEMORY"
            break
        time.sleep(0.2)
    if reason is not None:
        _kill_group(process.pid)
        try:
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ChildProcessError):
            process.kill()
        leftovers = _matching_processes(evidence)
        state = _read_state(evidence)
        state.update({
            "status": reason,
            "watchdog_killed": True,
            "child_pid": process.pid,
            "video_inference_claimed": False,
            "usable_video": False,
            "weight_alignment": "unknown",
            "remaining_processes": leftovers,
        })
        _seal(evidence, state)
        return 2
    state = _read_state(evidence)
    if state.get("bundle_status") != "SEALED":
        _seal(evidence, state)
    return int(process.returncode or 0)


def run_session(evidence: Path, *, budget_s: float, enable_transfer: bool, comfy_dir: Path, mode: str) -> int:
    evidence.mkdir(parents=True, exist_ok=True)
    os.environ["H3_T4_EVIDENCE_DIR"] = str(evidence)
    persist_process_sample(str(evidence), "stage0")
    deadline = time.monotonic() + budget_s
    spec = sample_spec(mode)
    state: dict[str, object] = {
        "stage": "video-smoke-session",
        "status": "RUNNING",
        "mode": mode,
        "spec": spec,
        "budget": budget_report(),
        "budget_s": budget_s,
        "download": False,
        "started_transfer": False,
        "audio_decoded": False,
        "audio_vae_downloaded": False,
        "usable_video": False,
        "mechanism_smoke": mode == "smoke",
        "video_inference_claimed": False,
        "placeholder": False,
        "weight_alignment": "unknown",
        "model_stage_ready": False,
        "snapshot": "pre-seal",
        "bundle_status": "NOT_SEALED",
        "lifecycle": [
            "preflight",
            "text_download",
            "text_encode_exit",
            "unet_download",
            "unet_sample_exit",
            "vae_download",
            "vae_decode_exit",
            "ffprobe",
            "cleanup",
        ],
    }
    exit_code = 2
    try:
        _write(evidence / "session-status.json", state)
        state["clone"] = clone_comfy(comfy_dir)
        state["install"] = install_model_deps()
        state["actor_deps"] = apply_plan()
        state["actor_imports"] = import_check()
        state["audit"] = audit_imports(comfy_dir)
        state["ffmpeg"] = ffmpeg_status()
        state["tokenizer_missing"] = missing_tokenizer_files(comfy_dir)
        state["nvfp4"] = _nvfp4_or_error()
        destination = choose_weight_dir([Path("/tmp"), Path("/kaggle/tmp")], int(SPECS["unet"]["bytes"]), shutil.disk_usage)
        state["weight_dir"] = None if destination is None else str(destination)
        probe_dir = destination or Path("/tmp")
        try:
            state["worker_probe"] = probe_worker_imports(
                runtime_env=build_unet_runtime_env(comfy_dir),
                checkpoint_dir=probe_dir,
                load_entry=True,
            )
        except Exception as exc:
            state["worker_probe"] = {"ok": False, "error_type": type(exc).__name__, "message": str(exc)[:500], "imports": [], "cuda_count": None, "checkpoint_dir_writable": False}
        versions = (state.get("actor_imports") or {}).get("installed_versions") or {}
        state["worker_probe"]["expected_versions"] = {name: versions.get(name) for name in ("torch", "torchvision", "torchaudio", "ray", "xfuser", "yunchang")}
        from minimax_h3_t4.runtime.unet_load import runtime_gate

        state["preflight"] = runtime_gate(cuda_count=_cuda_count(), available_bytes=host_available_bytes())
        blockers = preflight_blockers(state["actor_deps"], state["actor_imports"], state["audit"], state["preflight"], state["worker_probe"])
        if not state["ffmpeg"]["ok"]:
            blockers.append("ffmpeg and ffprobe are required before any weight transfer")
        if state["tokenizer_missing"]:
            blockers.append("pinned Comfy qwen25 tokenizer files are missing")
        if destination is None and enable_transfer:
            blockers.append("no writable disk can hold the UNet plus 1GiB")
        if enable_transfer and deadline - time.monotonic() < SEAL_RESERVE_S:
            blockers.append("remaining time is inside the seal reserve")
        state["preflight_blockers"] = blockers
        persist_process_sample(str(evidence), "after_preflight")
        _write(evidence / "session-status.json", state)
        if blockers:
            state["status"] = "STOP_PREFLIGHT"
            state["reason"] = "; ".join(blockers)
            return exit_code
        if not enable_transfer:
            state["status"] = "NOT_ENABLED"
            state["transfer"] = {"ok": False, "started_transfer": False, "status": "NOT_ENABLED"}
            return exit_code
        assert destination is not None
        text_path = _file_path(destination, "text_encoder")
        unet_path = _file_path(destination, "unet")
        vae_path = _file_path(destination, "video_vae")
        cond_path = evidence / "conditioning.pt"
        latent_path = evidence / "latent.pt"
        mp4_path = evidence / "smoke.mp4"
        state["text_download"] = _transfer("text_encoder", text_path, deadline)
        state["started_transfer"] = state["text_download"].get("started_transfer") is True or state["started_transfer"]
        state["download"] = state["started_transfer"]
        _write(evidence / "session-status.json", state)
        if state["text_download"].get("ok") is not True:
            state["status"] = "STOP_TEXT_TRANSFER"
            return exit_code
        if host_available_bytes() < int(SPECS["text_encoder"]["bytes"]) + LOW_AVAILABLE_BYTES:
            state["status"] = "STOP_TEXT_MEMORY"
            state["reason"] = "available memory cannot hold the packed text encoder plus 2GiB"
            return exit_code
        state["text"] = _stage_process(
            evidence,
            [
                "--stage", "text",
                "--weight", str(text_path),
                "--conditioning", str(cond_path),
                "--mode", mode,
            ],
            comfy_dir,
            "text",
            deadline,
        )
        persist_process_sample(str(evidence), "after_text_exit")
        _write(evidence / "session-status.json", state)
        if state["text"].get("ok") is not True:
            state["status"] = "STOP_TEXT_ENCODE"
            return exit_code
        state["text_file_after_encode"] = _maybe_release_weight(text_path, int(SPECS["unet"]["bytes"]))
        state["unet_download"] = _transfer("unet", unet_path, deadline)
        state["started_transfer"] = state["started_transfer"] or state["unet_download"].get("started_transfer") is True
        _write(evidence / "session-status.json", state)
        if state["unet_download"].get("ok") is not True:
            state["status"] = "STOP_UNET_TRANSFER"
            return exit_code
        state["sample"] = _stage_process(
            evidence,
            [
                "--stage", "sample",
                "--weight", str(unet_path),
                "--conditioning", str(cond_path),
                "--latent", str(latent_path),
                "--mode", mode,
                "--budget-s", str(max(1.0, deadline - time.monotonic() - SEAL_RESERVE_S)),
            ],
            comfy_dir,
            "sample",
            deadline,
        )
        state["sample_leftovers"] = _matching_processes(evidence)
        persist_process_sample(str(evidence), "after_unet_exit")
        _write(evidence / "session-status.json", state)
        if state["sample"].get("ok") is not True:
            state["status"] = "STOP_SAMPLE"
            return exit_code
        state["vae_download"] = _transfer("video_vae", vae_path, deadline)
        state["started_transfer"] = state["started_transfer"] or state["vae_download"].get("started_transfer") is True
        _write(evidence / "session-status.json", state)
        if state["vae_download"].get("ok") is not True:
            state["status"] = "STOP_VAE_TRANSFER"
            return exit_code
        state["decode"] = _stage_process(
            evidence,
            [
                "--stage", "decode",
                "--weight", str(vae_path),
                "--latent", str(latent_path),
                "--mp4", str(mp4_path),
                "--mode", mode,
            ],
            comfy_dir,
            "decode",
            deadline,
        )
        persist_process_sample(str(evidence), "after_vae_exit")
        decode = state["decode"]
        state["model_mp4"] = decode.get("video_inference_claimed") is True
        state["usable_video"] = False if mode == "smoke" else decode.get("usable_video") is True
        state["video_inference_claimed"] = state["model_mp4"] is True
        state["mechanism_smoke"] = mode == "smoke"
        state["mp4"] = str(mp4_path) if mp4_path.is_file() else None
        state["status"] = "SMOKE_MP4" if state["model_mp4"] and mode == "smoke" else "MP4" if state["model_mp4"] else "STOP_DECODE"
        state["cleanup"] = {"leftovers": _matching_processes(evidence)}
        exit_code = 0 if state["model_mp4"] else 2
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
        _write(evidence / "video-result.json", state)
        try:
            state["bundle"] = seal_evidence(evidence)
            state["bundle_status"] = "SEALED"
        except Exception as exc:
            state["bundle_status"] = "FAILED"
            state["bundle_error"] = {"error_type": type(exc).__name__, "message": str(exc)[:300]}
        state["snapshot"] = "final"
        _write(evidence / "session-status.json", state)
        _write(evidence.parent / f"{evidence.name}-manifest.json", state)


def _run_stage(args: argparse.Namespace, evidence: Path) -> int:
    evidence.mkdir(parents=True, exist_ok=True)
    os.environ["H3_T4_EVIDENCE_DIR"] = str(evidence)
    report: dict[str, object]
    try:
        if args.stage == "text":
            from minimax_h3_t4.runtime.video_stages import encode_text

            report = encode_text(weight=Path(args.weight), prompt=sample_spec(args.mode)["prompt"], destination=Path(args.conditioning))
        elif args.stage == "sample":
            from minimax_h3_t4.runtime.video_stages import sample_latent

            report = sample_latent(
                unet_path=Path(args.weight),
                conditioning_path=Path(args.conditioning),
                destination=Path(args.latent),
                comfy_root=Path(args.comfy_dir),
                mode=args.mode,
                evidence_dir=evidence,
                deadline_s=args.budget_s,
            )
        else:
            from minimax_h3_t4.runtime.video_stages import decode_video

            report = decode_video(vae_path=Path(args.weight), latent_path=Path(args.latent), mp4_path=Path(args.mp4))
    except BaseException as exc:
        name = type(exc).__name__
        report = {
            "ok": False,
            "status": "GPU_OOM" if name in {"OutOfMemoryError", "CudaOutOfMemoryError"} or "out of memory" in str(exc).lower() else "FAILED",
            "error_type": name,
            "message": str(exc)[:800],
            "video_inference_claimed": False,
            "usable_video": False,
        }
    _write(evidence / f"stage-{args.stage}.json", report)
    return 0 if report.get("ok") is True else 2


def _stage_process(evidence: Path, extra: list[str], comfy_dir: Path, name: str, deadline: float) -> dict[str, object]:
    if deadline - time.monotonic() < SEAL_RESERVE_S:
        return {"ok": False, "status": "STOP_BUDGET", "stage": name}
    stage_deadline = min(deadline - 30, time.monotonic() + STAGE_LIMITS_S[name])
    command = [sys.executable, str(Path(__file__).resolve()), "--evidence-dir", str(evidence), "--comfy-dir", str(comfy_dir), *extra]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(comfy_dir), str(SRC), env.get("PYTHONPATH", "")])
    env["H3_T4_EVIDENCE_DIR"] = str(evidence)
    process = subprocess.Popen(command, cwd=str(ROOT), env=env, start_new_session=True)
    while process.poll() is None:
        if time.monotonic() >= stage_deadline or host_available_bytes() < LOW_AVAILABLE_BYTES:
            _kill_group(process.pid)
            process.wait(timeout=5)
            return {"ok": False, "status": "STAGE_DEADLINE_OR_LOW_MEMORY", "stage": name, "pid": process.pid, "pid_dead": _dead(process.pid)}
        time.sleep(0.2)
    report_path = evidence / f"stage-{name}.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {"ok": False, "status": "BAD_STAGE_JSON"}
    else:
        report = {"ok": False, "status": "NO_STAGE_REPORT", "returncode": process.returncode}
    report["pid"] = process.pid
    report["pid_dead"] = _dead(process.pid)
    report["returncode"] = process.returncode
    return report


def _transfer(name: str, dest: Path, deadline: float) -> dict[str, object]:
    from minimax_h3_t4.runtime.official_transfer import transfer_file

    spec = SPECS[name]
    file_deadline = min(deadline - SEAL_RESERVE_S, time.monotonic() + seconds_at_rate(int(spec["bytes"]), 25 * 1024 * 1024) + 180)
    if file_deadline <= time.monotonic():
        return {"ok": False, "status": "DEADLINE", "started_transfer": False, "name": name}
    result = transfer_file(
        url=f"https://huggingface.co/{OFFICIAL_REPO}/resolve/{OFFICIAL_REVISION}/{spec['path']}",
        dest=dest,
        expected_bytes=int(spec["bytes"]),
        expected_sha256=str(spec["sha256"]),
        deadline=file_deadline,
    )
    result["name"] = name
    result["expected_bytes"] = int(spec["bytes"])
    result["expected_sha256"] = spec["sha256"]
    return result


def _maybe_release_weight(path: Path, upcoming_bytes: int) -> dict[str, object]:
    if not path.is_file():
        return {"released": False, "reason": "absent"}
    free = shutil.disk_usage(path.parent).free
    if free >= upcoming_bytes + 1024**3:
        return {"released": False, "reason": "disk holds the next file", "free": free}
    path.unlink()
    return {"released": True, "reason": "freed disk for the next weight", "path": str(path), "free_after": shutil.disk_usage(path.parent).free}


def _file_path(root: Path, name: str) -> Path:
    return root / Path(str(SPECS[name]["path"])).name


def _nvfp4_or_error() -> dict[str, object]:
    """Record Comfy's packed-4 gate. A T4 is not a native packed-4 device."""

    try:
        import comfy.model_management as model_management

        supported = bool(model_management.supports_nvfp4_compute())
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "message": str(exc)[:300]}
    return {
        "supports_packed4_compute": supported,
        "t4_native_packed4": False,
        "execution": "native" if supported else "Comfy dequantizes the packed text encoder at forward; unpacked size is not measured",
    }


def _show_video_files(evidence: Path) -> None:
    show_session_evidence(evidence)
    extras = [evidence / "smoke.mp4", evidence / "video-result.json", *sorted(evidence.glob("stage-*.json"))]
    try:
        from IPython.display import FileLink, display
    except Exception:
        display = None
        FileLink = None
    for path in extras:
        if not path.is_file():
            print(f"MISSING {path}", flush=True)
            continue
        print(f"FILE {path.name} bytes={path.stat().st_size} path={path}", flush=True)
        link = _link_path(path)
        print("DOWNLOAD_BEFORE_STOP " + link, flush=True)
        if display is not None and FileLink is not None:
            display(FileLink(link))


def _sleep_child(evidence: Path) -> None:
    evidence.mkdir(parents=True, exist_ok=True)
    grandchild = subprocess.Popen(["sleep", "60"])
    _write(evidence / "grandchild.pid", {"pid": grandchild.pid})
    _write(evidence / "session-status.json", {"status": "BLOCKED", "video_inference_claimed": False, "usable_video": False, "weight_alignment": "unknown"})
    time.sleep(60)


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _dead(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


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
        if "H3T4Worker" in command or ("run_video_session.py" in command and "--stage" in command):
            pid = int(entry.name)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
            found.append({"pid": pid, "cmd": command[:160], "killed": True})
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
    state["video_inference_claimed"] = False
    state["usable_video"] = False
    _write(evidence / "session-status.json", state)
    try:
        state["bundle"] = seal_evidence(evidence)
        state["bundle_status"] = "SEALED"
    except Exception as exc:
        state["bundle_status"] = "FAILED"
        state["bundle_error"] = {"error_type": type(exc).__name__, "message": str(exc)[:300]}
    _write(evidence / "session-status.json", state)
    _write(evidence.parent / f"{evidence.name}-manifest.json", state)


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


if __name__ == "__main__":
    raise SystemExit(main())
