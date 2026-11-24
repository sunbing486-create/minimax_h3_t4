#!/usr/bin/env python3
"""Prepare a clean environment for real H3T4Worker actor admission.

This does not download weights and does not start actors. It installs the
imports required by ``H3T4Worker._initialize_distributed`` without changing
the torch that is already installed. ComfyUI is not one of those imports.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PINS = (
    "xfuser==0.4.5",
    "yunchang==0.6.4",
    "transformers==5.0.0",
    "diffusers==0.37.1",
)
PROTECTED = ("torch", "torchvision", "torchaudio", "ray")
REQUIRED_IMPORTS = ("torch", "ray", "xfuser.core.distributed", "yunchang")
NOT_REQUIRED_FOR_ADMISSION = ("comfy", "folder_paths")
COMFYUI_REVISION_NOT_REQUIRED = "2fcb6cef9ab1b6dc556b7cff1765a0724ad5e884"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU bootstrap for weight-free Ray actor admission")
    parser.add_argument("--stage", choices=("plan", "apply", "check"), default="check")
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)
    report_path = Path(args.report) if args.report else Path.cwd() / f"bootstrap-{args.stage}.json"
    if args.stage == "plan":
        report = build_plan()
    elif args.stage == "apply":
        report = apply_plan()
    else:
        report = import_check()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": args.stage, "report": str(report_path), "ok": report.get("ok"), "actor_admission_passed": False}, sort_keys=True))
    return 0 if report.get("ok") else 2


def installed_versions() -> dict[str, str]:
    found: dict[str, str] = {}
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name:
            found[name] = dist.version
    return found


def build_plan() -> dict[str, object]:
    current = installed_versions()
    if "torch" not in current:
        return {
            "ok": False,
            "action": "refused",
            "reason": "torch is not installed; this bootstrap does not install or replace torch",
            "installed": {name: current.get(name) for name in PROTECTED},
            "pins": list(PINS),
            "actor_admission_passed": False,
        }
    dry = _pip_dry_run()
    refused = _protected_changes(current, dry.get("install", []))
    return {
        "ok": dry["returncode"] == 0 and not refused,
        "action": "would_install" if not refused and dry["returncode"] == 0 else "refused",
        "reason": refused or dry.get("error"),
        "pins": list(PINS),
        "install": [
            {"name": item.get("metadata", {}).get("name"), "version": item.get("metadata", {}).get("version")}
            for item in dry.get("install", [])
        ],
        "protected_versions": {name: current.get(name) for name in PROTECTED},
        "comfyui_required_for_actor_admission": False,
        "comfyui_revision_not_cloned": COMFYUI_REVISION_NOT_REQUIRED,
        "downloads_weights": False,
        "actor_admission_passed": False,
    }


def apply_plan() -> dict[str, object]:
    plan = build_plan()
    if not plan["ok"]:
        plan["applied"] = False
        return plan
    command = [sys.executable, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", *PINS]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    plan["applied"] = completed.returncode == 0
    plan["ok"] = completed.returncode == 0
    plan["pip_returncode"] = completed.returncode
    if completed.returncode != 0:
        plan["reason"] = (completed.stderr or completed.stdout)[-2000:]
    plan["installed_after"] = {name: installed_versions().get(name) for name in (*PROTECTED, "xfuser", "yunchang", "transformers", "diffusers")}
    return plan


def import_check() -> dict[str, object]:
    modules = {name: _classify_import(name) for name in (*REQUIRED_IMPORTS, *NOT_REQUIRED_FOR_ADMISSION)}
    xfuser_status = modules["xfuser.core.distributed"]
    symbols = _xfuser_symbols() if xfuser_status["status"] == "importable" else {}
    cuda = _cuda_status()
    worker = _classify_import("minimax_h3_t4.runtime.worker")
    versions = installed_versions()
    xfuser_present = versions.get("xfuser") == "0.4.5" and xfuser_status["status"] in {"importable", "cpu_cuda_unavailable"}
    yunchang_present = modules["yunchang"]["status"] == "importable" and versions.get("yunchang") == "0.6.4"
    base_ok = modules["torch"]["status"] == "importable" and modules["ray"]["status"] == "importable"
    materials_ready = base_ok and xfuser_present and yunchang_present
    gpu_permitted = materials_ready and cuda.get("device_count") == 2
    return {
        "ok": materials_ready,
        "materials_ready_for_gpu_stage": materials_ready,
        "gpu_stage_permitted": gpu_permitted,
        "actor_admission_passed": False,
        "gpu_stage_started": False,
        "official_weights_missing_blocks_actor_admission": False,
        "required_imports": {name: modules[name] for name in REQUIRED_IMPORTS},
        "not_required_for_actor_admission": {name: modules[name] for name in NOT_REQUIRED_FOR_ADMISSION},
        "xfuser_symbols": symbols,
        "installed_versions": {
            name: versions.get(name) for name in ("torch", "torchvision", "ray", "xfuser", "yunchang", "transformers", "diffusers", "safetensors")
        },
        "worker_module": worker,
        "cuda": cuda,
        "comfyui_required_for_actor_admission": False,
        "explanation": (
            "Actor admission needs torch, ray, xfuser==0.4.5 and yunchang==0.6.4. "
            "Importing xfuser.core.distributed on a CPU torch build raises NotImplementedError from xfuser.envs.get_device_version; "
            "that means the package is installed and the GPU check inside xfuser refused to run, not that the module is missing. "
            "comfy and folder_paths are load_unet imports and do not block admission. Missing weight files do not block admission."
        ),
        "downloads_weights": False,
    }


def _pip_dry_run() -> dict[str, object]:
    report = Path(os.environ.get("H3_T4_PIP_REPORT", "/tmp/h3-admission-pip-report.json"))
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--upgrade-strategy",
        "only-if-needed",
        "--report",
        str(report),
        *PINS,
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0 or not report.is_file():
        return {"returncode": completed.returncode, "install": [], "error": (completed.stderr or completed.stdout)[-2000:]}
    payload = json.loads(report.read_text(encoding="utf-8"))
    return {"returncode": 0, "install": payload.get("install", []), "error": None}


def _protected_changes(current: dict[str, str], install_items: list[dict]) -> str:
    problems = []
    for item in install_items:
        meta = item.get("metadata") or {}
        name = str(meta.get("name") or "").lower()
        version = str(meta.get("version") or "")
        if name not in PROTECTED:
            continue
        installed = current.get(name)
        if installed is None:
            problems.append(f"{name}=={version} would be installed; refusing to add {name}")
        elif installed != version:
            problems.append(f"{name} {installed} would become {version}")
    return "; ".join(problems)


def _classify_import(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        missing = exc.name or name
        status = "missing_module" if missing == name or missing.split(".")[0] == name.split(".")[0] else "missing_dependency"
        return {"status": status, "error": f"ModuleNotFoundError: {exc}", "origin": None, "version": _version(name)}
    except ImportError as exc:
        return {"status": "import_error", "error": f"ImportError: {exc}", "origin": None, "version": _version(name)}
    except NotImplementedError as exc:
        text = str(exc)
        if "No Accelerators" in text:
            return {
                "status": "cpu_cuda_unavailable",
                "error": f"NotImplementedError: {exc}",
                "origin": _installed_origin(name),
                "version": _version(name),
            }
        return {"status": "error", "error": f"NotImplementedError: {exc}", "origin": None, "version": _version(name)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "origin": None, "version": _version(name)}
    origin = getattr(module, "__file__", None)
    return {"status": "importable", "error": None, "origin": origin, "version": _version(name)}


def _installed_origin(name: str) -> str | None:
    top = name.split(".")[0]
    try:
        dist = metadata.distribution(top)
    except metadata.PackageNotFoundError:
        return None
    files = dist.files or []
    needle = top.replace("-", "_") + "/__init__.py"
    for file in files:
        text = str(file)
        if text.endswith(needle) or text.endswith(top.replace("-", "_") + ".py"):
            return str(file.locate())
    return None


def _version(name: str) -> str | None:
    top = name.split(".")[0]
    try:
        return metadata.version(top)
    except metadata.PackageNotFoundError:
        return None


def _xfuser_symbols() -> dict[str, bool]:
    module = importlib.import_module("xfuser.core.distributed")
    return {
        "init_distributed_environment": callable(getattr(module, "init_distributed_environment", None)),
        "initialize_model_parallel": callable(getattr(module, "initialize_model_parallel", None)),
    }


def _cuda_status() -> dict[str, object]:
    import torch

    cuda_build = getattr(torch.version, "cuda", None)
    available = bool(torch.cuda.is_available())
    if not available:
        kind = "cpu_build" if (not cuda_build or "+cpu" in torch.__version__) else "cuda_build_without_device"
        return {
            "kind": kind,
            "actor_admission_passed": False,
            "version": torch.__version__,
            "cuda_build": cuda_build,
            "device_count": 0,
            "note": "CPU or no visible device. This is not a passed actor admission and not a mock GPU result.",
        }
    return {
        "kind": "cuda_devices_present",
        "actor_admission_passed": False,
        "version": torch.__version__,
        "cuda_build": cuda_build,
        "device_count": int(torch.cuda.device_count()),
        "note": "Devices are visible, but this check does not start H3T4Worker.",
    }


if __name__ == "__main__":
    raise SystemExit(main())
