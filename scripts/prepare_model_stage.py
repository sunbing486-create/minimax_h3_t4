#!/usr/bin/env python3
"""CPU preparation for a later MiniMax-H3 model load. The default stages do not download weights.

``plan`` only reads the local machine and the checked-in header summary.
``clone-comfy`` fetches ComfyUI at the pinned revision into a writable directory.
``audit`` imports comfy and folder_paths and separates a missing module from a
CPU build that has no CUDA. ``download`` stays off unless ``--confirm-download``
is passed, and it still refuses a destination that cannot hold the selected file.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimax_h3_t4.model_catalog import OFFICIAL_TOTAL_BYTES, inventory_local_models, official_download_list
from minimax_h3_t4.runtime.checkpoint_budget import host_available_bytes
from minimax_h3_t4.runtime.model_facts import COMFYUI_REVISION, memory_facts, minimum_legal_sample

PROTECTED = ("torch", "torchvision", "torchaudio", "ray")
MODEL_DEPS = (
    "psutil",
    "pyyaml",
    "Pillow",
    "scipy",
    "tqdm",
    "aiohttp",
    "yarl",
    "requests",
    "einops",
    "safetensors",
    "tokenizers",
    "sentencepiece",
    "torchsde",
    "alembic",
    "SQLAlchemy",
    "filelock",
    "av",
    "simpleeval",
    "blake3",
    "transformers==5.0.0",
    "comfy-aimdo==0.6.0",
    "comfy-kitchen==0.2.42",
)
COMFY_REMOTE = "https://github.com/Comfy-Org/ComfyUI.git"
IMPORTS = ("comfy", "folder_paths", "comfy.utils", "comfy.model_detection", "comfy.model_management", "comfy.model_base")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the model stage without downloading weights by default")
    parser.add_argument("--stage", choices=("plan", "clone-comfy", "install-model-deps", "audit", "download"), default="plan")
    parser.add_argument("--report", default="")
    parser.add_argument("--comfy-dir", default=os.environ.get("H3_T4_COMFY_DIR", "/tmp/ComfyUI-h3"))
    parser.add_argument("--search", action="append", default=[])
    parser.add_argument("--confirm-download", action="store_true")
    parser.add_argument("--enable-transfer", action="store_true")
    parser.add_argument("--only", default="")
    parser.add_argument("--dest-dir", default="")
    parser.add_argument("--budget-s", type=float, default=600)
    args = parser.parse_args(argv)
    report_path = Path(args.report) if args.report else Path.cwd() / f"model-{args.stage}.json"
    if args.stage == "plan":
        report = build_plan(search=[Path(item) for item in args.search])
    elif args.stage == "clone-comfy":
        report = clone_comfy(Path(args.comfy_dir))
    elif args.stage == "install-model-deps":
        report = install_model_deps()
    elif args.stage == "audit":
        report = audit_imports(Path(args.comfy_dir))
    else:
        report = download_official(
            confirm=args.confirm_download,
            enable_transfer=args.enable_transfer,
            only=args.only,
            dest_dir=args.dest_dir,
            deadline=time.monotonic() + args.budget_s,
        )
    _write(report_path, report)
    print(json.dumps({"stage": args.stage, "report": str(report_path), "ok": report.get("ok"), "download": report.get("started_transfer") is True, "actor_admission_passed": False, "video_inference_claimed": False}, sort_keys=True))
    return 0 if report.get("ok") else 2


def build_plan(search: list[Path] | None = None) -> dict[str, Any]:
    roots = search or [Path("/kaggle/input"), Path("/kaggle/working"), Path("/tmp")]
    available = host_available_bytes()
    disks = []
    for path in (Path("/kaggle/working"), Path("/tmp"), Path("/kaggle/input")):
        disks.append(_disk(path))
    versions = _versions()
    return {
        "ok": "torch" in versions,
        "stage": "plan",
        "download": False,
        "actor_admission_passed": False,
        "video_inference_claimed": False,
        "inference_memory_proven": False,
        "model_stage_allowed": False,
        "weights_status": inventory_local_models(roots),
        "protected_versions": {name: versions.get(name) for name in PROTECTED},
        "torch_present": "torch" in versions,
        "memory": memory_facts(available or None),
        "minimum_legal_sample": minimum_legal_sample(),
        "disks": disks,
        "weights_bytes": OFFICIAL_TOTAL_BYTES,
        "download_list_if_missing": official_download_list(),
        "comfyui_revision": COMFYUI_REVISION,
        "stop_rules": _stop_rules(),
    }


def clone_comfy(dest: Path) -> dict[str, Any]:
    dest = dest.resolve()
    source_root = ROOT.resolve()
    try:
        dest.relative_to(source_root)
        inside = True
    except ValueError:
        inside = False
    if inside:
        return {"ok": False, "stage": "clone-comfy", "reason": "comfy checkout must not be inside the read-only package", "download": False}
    if (dest / ".git").is_dir():
        current = _git(dest, ["rev-parse", "HEAD"])
        if current == COMFYUI_REVISION:
            return {"ok": True, "stage": "clone-comfy", "path": str(dest), "revision": current, "action": "reused", "download": False}
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        init = subprocess.run(["git", "init", str(dest)], check=False, text=True, capture_output=True)
        if init.returncode != 0:
            return {"ok": False, "stage": "clone-comfy", "reason": init.stderr[-400:], "download": False}
    _git(dest, ["remote", "remove", "origin"], check=False)
    added = subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", COMFY_REMOTE], check=False, text=True, capture_output=True)
    if added.returncode != 0 and "already exists" not in added.stderr:
        return {"ok": False, "stage": "clone-comfy", "reason": added.stderr[-400:], "download": False}
    fetched = subprocess.run(
        ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", COMFYUI_REVISION],
        check=False,
        text=True,
        capture_output=True,
        timeout=600,
    )
    if fetched.returncode != 0:
        return {"ok": False, "stage": "clone-comfy", "reason": (fetched.stderr or fetched.stdout)[-400:], "download": False}
    checked = subprocess.run(["git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD"], check=False, text=True, capture_output=True)
    revision = _git(dest, ["rev-parse", "HEAD"])
    return {
        "ok": checked.returncode == 0 and revision == COMFYUI_REVISION,
        "stage": "clone-comfy",
        "path": str(dest),
        "revision": revision,
        "action": "fetched",
        "download": False,
        "weights_downloaded": False,
    }


def protected_replacements(planned: list[dict[str, Any]], before: dict[str, str | None]) -> list[str]:
    """Refuse a pip plan that would change torch, vision, audio, or ray, including transitive installs."""

    refused = []
    for item in planned:
        name = str(item.get("metadata", {}).get("name", "")).lower().replace("_", "-")
        version = item.get("metadata", {}).get("version")
        canonical = name.replace("-", "")
        for protected in PROTECTED:
            if canonical == protected.replace("-", "") and before.get(protected) not in {None, version}:
                refused.append(f"{protected} {before.get(protected)} would become {version}")
    return refused


def install_model_deps() -> dict[str, Any]:
    """Install Comfy import dependencies. Never install or replace torch, vision, audio, or ray."""

    before = {name: _versions().get(name) for name in PROTECTED}
    if before["torch"] is None:
        return {"ok": False, "stage": "install-model-deps", "reason": "torch is absent; this stage does not install torch", "download": False}
    report_path = Path("/tmp/h3-model-deps-pip.json")
    dry = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--dry-run", "--upgrade-strategy", "only-if-needed", "--report", str(report_path), *MODEL_DEPS],
        check=False,
        text=True,
        capture_output=True,
    )
    planned = []
    if report_path.is_file():
        planned = json.loads(report_path.read_text(encoding="utf-8")).get("install", [])
    refused = protected_replacements(planned, before)
    if dry.returncode != 0 or refused:
        return {
            "ok": False,
            "stage": "install-model-deps",
            "reason": "; ".join(refused) or (dry.stderr or dry.stdout)[-500:],
            "download": False,
            "protected_versions": before,
        }
    installed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", *MODEL_DEPS],
        check=False,
        text=True,
        capture_output=True,
    )
    after = {name: _versions().get(name) for name in PROTECTED}
    changed = {name: {"before": before[name], "after": after[name]} for name in PROTECTED if before[name] != after[name]}
    return {
        "ok": installed.returncode == 0 and not changed,
        "stage": "install-model-deps",
        "download": False,
        "weights_downloaded": False,
        "protected_versions": after,
        "protected_changed": changed,
        "pip_returncode": installed.returncode,
        "reason": None if installed.returncode == 0 and not changed else (installed.stderr or "")[-500:],
    }


def audit_imports(comfy_dir: Path) -> dict[str, Any]:
    versions = _versions()
    if "torch" not in versions:
        return {"ok": False, "stage": "audit", "reason": "torch is absent; this audit does not install torch", "download": False}
    if not (comfy_dir / "comfy").is_dir() or not (comfy_dir / "folder_paths.py").is_file():
        return {"ok": False, "stage": "audit", "reason": f"ComfyUI checkout is incomplete: {comfy_dir}", "download": False}
    sys.path.insert(0, str(comfy_dir))
    rows = []
    ok = True
    for name in IMPORTS:
        rows.append(_import_one(name))
        if rows[-1]["status"] not in {"imported", "cpu_cuda_unavailable"}:
            ok = False
    return {
        "ok": ok,
        "stage": "audit",
        "comfy_dir": str(comfy_dir),
        "imports": rows,
        "protected_versions": {name: versions.get(name) for name in PROTECTED},
        "download": False,
        "actor_admission_passed": False,
        "video_inference_claimed": False,
        "model_stage_allowed": False,
    }


def download_official(*, confirm: bool, enable_transfer: bool, only: str, dest_dir: str, deadline: float) -> dict[str, Any]:
    """Check space first. Transfer bytes only when enable_transfer is set."""

    if not confirm or not only or not dest_dir:
        return {
            "ok": False,
            "stage": "download",
            "download": False,
            "reason": "download stays off until --confirm-download, --only, and --dest-dir are all set",
            "started_transfer": False,
        }
    chosen = next((row for row in official_download_list() if row["path"].endswith(only) or only == Path(row["path"]).name), None)
    if chosen is None:
        return {"ok": False, "stage": "download", "download": False, "reason": f"unknown weight {only}", "started_transfer": False}
    dest = Path(dest_dir)
    disk = _disk(dest if dest.exists() else dest.parent)
    free = int(disk.get("free") or 0)
    need = int(chosen["bytes"]) + 1024**3
    if free < need:
        return {
            "ok": False,
            "stage": "download",
            "download": False,
            "started_transfer": False,
            "reason": f"destination free {free} is below file bytes plus 1GiB ({need})",
            "dest": disk,
            "file": chosen["path"],
        }
    if str(dest).startswith("/kaggle/working") and int(chosen["bytes"]) > int(disk.get("free") or 0):
        return {"ok": False, "stage": "download", "download": False, "started_transfer": False, "reason": "working disk cannot hold this file"}
    if not enable_transfer:
        return {
            "ok": True,
            "stage": "download",
            "download": False,
            "started_transfer": False,
            "reason": "capacity check passed; transfer is implemented but not enabled",
            "file": chosen,
            "dest": str(dest),
        }
    from minimax_h3_t4.runtime.official_transfer import transfer_file

    result = transfer_file(
        url=str(chosen["url"]),
        dest=dest / Path(str(chosen["path"])).name,
        expected_bytes=int(chosen["bytes"]),
        expected_sha256=str(chosen["sha256"]),
        deadline=deadline,
    )
    result.update({"stage": "download", "download": result.get("started_transfer") is True, "file": chosen["path"]})
    return result


def _import_one(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        return {"name": name, "status": "missing_module", "error_type": "ModuleNotFoundError", "message": str(exc)}
    except NotImplementedError as exc:
        text = str(exc)
        if "Accelerator" in text or "CUDA" in text or "No Accelerators" in text:
            return {"name": name, "status": "cpu_cuda_unavailable", "error_type": "NotImplementedError", "message": text[:300]}
        return {"name": name, "status": "error", "error_type": "NotImplementedError", "message": text[:300]}
    except Exception as exc:
        text = str(exc)
        cuda = "CUDA" in text or "cuda" in text
        return {
            "name": name,
            "status": "cpu_cuda_unavailable" if cuda else "error",
            "error_type": type(exc).__name__,
            "message": text[:300],
            "traceback": traceback.format_exc()[-500:],
        }
    return {"name": name, "status": "imported", "file": getattr(module, "__file__", None)}


def _stop_rules() -> dict[str, Any]:
    return {
        "default_notebook_downloads_weights": False,
        "default_notebook_starts_gpu": False,
        "actor_pass_is_not_model_or_video_pass": True,
        "cpu_clone_stop_after_s": 600,
        "future_gpu_session": "not started by this notebook; a later command must exit by itself and record evidence before anyone can leave it running",
        "future_gpu_sample": "only after a measured load says the full tensors were released: 32x32, 5 frames, 1 step, then ffprobe",
        "old_weight_free_budget_is_not_enough": "the 240s command and 6 minute GPU budget covered actor admission only",
    }


def _versions() -> dict[str, str]:
    found: dict[str, str] = {}
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name:
            found[name] = dist.version
    return found


def _disk(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    usage = shutil.disk_usage(path)
    return {"path": str(path), "exists": True, "total": usage.total, "free": usage.free, "used": usage.used}


def _git(dest: Path, args: list[str], check: bool = True) -> str:
    completed = subprocess.run(["git", "-C", str(dest), *args], check=False, text=True, capture_output=True)
    if check and completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


if __name__ == "__main__":
    raise SystemExit(main())
