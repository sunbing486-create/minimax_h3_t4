"""Load only the official UNet onto the two existing workers, then stop.

This does not encode text, decode a VAE, or call the sampler. A missing UNet
file or fewer than two CUDA devices never starts Ray.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from ..model_catalog import OFFICIAL_MODELS

LOW_AVAILABLE_BYTES = 2 * 1024**3
UNET_SPEC = next(item for item in OFFICIAL_MODELS if item["name"] == "unet")


def required_files() -> list[dict[str, Any]]:
    """load_unet opens one checkpoint path. Text and VAE files are not read."""

    return [dict(UNET_SPEC)]


def choose_weight_dir(candidates: list[Path], need_bytes: int, usage) -> Path | None:
    for candidate in candidates:
        if not candidate.exists():
            continue
        disk = usage(candidate)
        if disk.total <= need_bytes:
            continue
        if disk.free < need_bytes + 1024**3:
            continue
        return candidate
    return None


def decide_unet_load(
    *,
    unet_reusable: bool,
    cuda_count: int,
    allow: bool,
    available_bytes: int,
) -> dict[str, Any]:
    blockers = []
    if not allow:
        blockers.append("H3_T4_ALLOW_UNET_LOAD is not 1")
    if not unet_reusable:
        blockers.append("official unet file is missing or its sha256 does not match")
    if available_bytes < LOW_AVAILABLE_BYTES:
        blockers.append("available memory is below 2GiB")
    if cuda_count != 2:
        blockers.append(f"need 2 CUDA devices, found {cuda_count}")
    return {
        "action": "RUN" if not blockers else "NOT_RUN",
        "blockers": blockers,
        "required_files": [UNET_SPEC["path"]],
        "text_encoder_required": False,
        "vae_required": False,
        "unet_loaded": False,
        "video_inference_claimed": False,
        "inference_memory_proven": False,
        "actor_admission_passed": False,
        "model_stage_ready": False,
        "weight_alignment": "unknown",
    }


def summarize_module_storage(module: Any) -> dict[str, Any]:
    logical = meta = materialized = 0
    devices: dict[str, int] = {}
    dtypes: dict[str, int] = {}
    count = 0
    if module is None:
        return {"parameter_count": 0, "logical_bytes": 0, "meta_bytes": 0, "materialized_bytes": 0, "bytes_by_device": {}, "bytes_by_dtype": {}}
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return {"parameter_count": 0, "logical_bytes": 0, "meta_bytes": 0, "materialized_bytes": 0, "bytes_by_device": {}, "bytes_by_dtype": {}}
    for parameter in parameters():
        if parameter is None:
            continue
        count += 1
        nbytes = int(parameter.numel()) * int(parameter.element_size())
        logical += nbytes
        device = str(parameter.device)
        devices[device] = devices.get(device, 0) + nbytes
        dtype = str(parameter.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + nbytes
        if parameter.device.type == "meta":
            meta += nbytes
        else:
            materialized += nbytes
    return {
        "parameter_count": count,
        "logical_bytes": logical,
        "meta_bytes": meta,
        "materialized_bytes": materialized,
        "bytes_by_device": devices,
        "bytes_by_dtype": dtypes,
        "meta_counts_as_loaded": False,
    }


def bounded_weight_check(module: Any, path: str | None, *, limit: int = 64) -> dict[str, Any]:
    """Compare a few materialized values with the checkpoint. Meta tensors are not a match."""

    if not path or module is None or not callable(getattr(module, "named_parameters", None)):
        return {"status": "NOT_CHECKED", "compared": 0}
    from safetensors import safe_open

    compared = matched = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        names = set(handle.keys())
        for name, parameter in module.named_parameters():
            if compared >= 3:
                break
            if parameter is None or parameter.device.type == "meta" or name not in names:
                continue
            reference = handle.get_tensor(name).flatten()[:limit]
            actual = parameter.detach().flatten()[:limit].cpu()
            compared += 1
            if reference.shape == actual.shape and reference.dtype == actual.dtype and bool((reference == actual).all()):
                matched += 1
    return {"status": "CHECKED" if compared else "NOT_COMPARABLE", "compared": compared, "matched": matched}


def persist_process_sample(evidence_dir: str | None, label: str, *, rank: int | None = None) -> dict[str, Any]:
    """Append one sample and fsync it before the next import can fail."""

    import json

    sample = sample_process_memory(label)
    sample["rank"] = rank
    sample["sampled_max_is_not_a_continuous_profiler_peak"] = True
    sample["weight_alignment"] = "unknown"
    if evidence_dir:
        path = Path(evidence_dir) / "memory-samples.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return sample


def sample_process_memory(label: str) -> dict[str, Any]:
    rss = pss = available = cached = None
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
    rollup = Path("/proc/self/smaps_rollup")
    if rollup.is_file():
        for line in rollup.read_text(encoding="utf-8").splitlines():
            if line.startswith("Pss:"):
                pss = int(line.split()[1]) * 1024
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
            elif line.startswith("Cached:"):
                cached = int(line.split()[1]) * 1024
    return {
        "label": label,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "rss_bytes": rss,
        "pss_bytes": pss,
        "mem_available_bytes": available,
        "page_cache_bytes": cached,
        "gpus": _gpus(),
    }


def runtime_gate(*, cuda_count: int, available_bytes: int) -> dict[str, Any]:
    """Import what H3T4Worker.load_unet actually needs. Missing modules block the download."""

    checks = []
    for name in ("torch", "ray", "xfuser.core.distributed", "yunchang", "comfy", "folder_paths", "comfy.model_management", "comfy.model_detection", "comfy.utils"):
        checks.append(_import_status(name))
    missing = [row["name"] for row in checks if row["status"] == "missing_module"]
    blockers = []
    if missing:
        blockers.append("missing modules: " + ", ".join(missing))
    if available_bytes < LOW_AVAILABLE_BYTES:
        blockers.append("available memory is below 2GiB")
    if cuda_count != 2:
        blockers.append(f"need 2 CUDA devices before any weight transfer, found {cuda_count}")
    return {
        "ok": not blockers,
        "checks": checks,
        "missing_modules": missing,
        "cuda_count": cuda_count,
        "available_bytes": available_bytes,
        "blockers": blockers,
        "download": False,
    }


def _import_status(name: str) -> dict[str, Any]:
    import importlib

    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        return {"name": name, "status": "missing_module", "error_type": "ModuleNotFoundError", "message": str(exc)}
    except (NotImplementedError, AssertionError) as exc:
        text = str(exc)
        if "CUDA" in text or "Accelerator" in text or "No Accelerators" in text:
            return {"name": name, "status": "cpu_cuda_unavailable", "error_type": type(exc).__name__, "message": text[:240]}
        return {"name": name, "status": "error", "error_type": type(exc).__name__, "message": text[:240]}
    except Exception as exc:
        text = str(exc)
        status = "cpu_cuda_unavailable" if "CUDA" in text or "cuda" in text else "error"
        return {"name": name, "status": status, "error_type": type(exc).__name__, "message": text[:240]}
    return {"name": name, "status": "imported"}


def worker_probe_blockers(probe: dict[str, Any], *, require_cuda: int = 2) -> list[str]:
    """Block the download when the Ray worker, not the driver, is unready."""

    blockers: list[str] = []
    if probe.get("error_type"):
        blockers.append(f"ray worker probe failed: {probe.get('error_type')}")
    for row in probe.get("imports") or []:
        status = row.get("status")
        name = row.get("name")
        if status == "missing_module":
            blockers.append(f"ray worker missing {name}")
        elif require_cuda == 2 and status != "imported":
            blockers.append(f"ray worker import {name} status {status}")
    if probe.get("cuda_count") != require_cuda:
        blockers.append(f"ray worker needs {require_cuda} CUDA devices, found {probe.get('cuda_count')}")
    if probe.get("checkpoint_dir_writable") is not True:
        blockers.append("ray worker cannot write the checkpoint directory")
    expected = probe.get("expected_versions") or {}
    versions = probe.get("versions") or {}
    for name, wanted in expected.items():
        if wanted and versions.get(name) != wanted:
            blockers.append(f"ray worker {name} {versions.get(name)} != driver {wanted}")
    entry = probe.get("load_entry")
    if isinstance(entry, dict):
        for name in entry.get("missing") or []:
            blockers.append(f"load entry missing {name}")
        if require_cuda == 2 and entry.get("ok") is not True:
            blockers.append("load entry import did not succeed inside the ray worker")
    return blockers


def run_unet_load_only(
    *,
    unet_path: str | None,
    cuda_count: int,
    allow: bool,
    available_bytes: int,
    evidence: Any | None = None,
    deadline: float | None = None,
    comfy_root: str | None = None,
) -> dict[str, Any]:
    reusable = False
    if unet_path:
        from .weights import inspect_local_weight

        row = inspect_local_weight(name="unet", path=unet_path, expected_sha256=str(UNET_SPEC["sha256"]), expected_bytes=int(UNET_SPEC["bytes"]))
        reusable = row["reusable"] is True
    decision = decide_unet_load(unet_reusable=reusable, cuda_count=cuda_count, allow=allow, available_bytes=available_bytes)
    if decision["action"] != "RUN":
        return decision
    load_timeout = None
    if deadline is not None:
        load_timeout = deadline - time.monotonic() - 20
        if load_timeout <= 0:
            decision["action"] = "STOP_BUDGET_BEFORE_LOAD"
            decision["reason"] = "the remaining session time is inside the cleanup reserve"
            return decision
    persist_process_sample(os.environ.get("H3_T4_EVIDENCE_DIR"), "before_worker_load")
    if evidence is not None:
        evidence.event("unet_load_only_start", path=unet_path, timeout_seconds=load_timeout)
    from ..loader_nodes import H3T4Initializer, H3T4UNETLoader, build_unet_runtime_env

    group = None
    try:
        builder = (lambda: build_unet_runtime_env(Path(comfy_root))) if comfy_root else None
        initializer = H3T4Initializer(runtime_env_builder=builder) if builder is not None else H3T4Initializer()
        group = initializer.initialize(load_after=object(), manage_parent_memory=False, for_admission=False, force_on_failure=True)[0]
        group = H3T4UNETLoader(path_resolver=lambda _name: str(unet_path)).load(
            group,
            str(unet_path),
            object(),
            manage_parent_memory=False,
            force_on_failure=True,
            timeout_seconds=load_timeout,
        )[0]
        reports = []
        from .lifecycle import remote
        from .startup import resolve_one

        for worker in group.workers:
            left = 20 if deadline is None else max(1.0, deadline - time.monotonic() - 20)
            reports.append(resolve_one(group.ray_module, remote(worker, "shard_report"), left, label="shard_report"))
        decision.update({
            "action": "LOADED",
            "unet_loaded": _loaded(reports),
            "shard_reports": reports,
            "weight_alignment": "unknown",
            "weight_alignment_reason": "materialized byte counts do not prove quantized parameter values match the checkpoint",
            "video_inference_claimed": False,
            "inference_memory_proven": False,
        })
        if evidence is not None:
            evidence.event("unet_load_only_done", unet_loaded=decision["unet_loaded"])
        return decision
    except Exception as exc:
        decision["worker_diagnostics"] = _worker_diagnostics(group)
        decision.update({"action": "FAILED", "unet_loaded": False, "error_type": type(exc).__name__, "message": str(exc)[:2000], "weight_alignment": "unknown"})
        if evidence is not None:
            evidence.event("unet_load_only_failed", error_type=type(exc).__name__)
        return decision
    finally:
        termination = getattr(group, "termination", None) if group is not None else None
        alive = group is not None and getattr(group, "alive", False)
        cleanup: dict[str, Any] = {"attempted": bool(alive or termination), "alive_before_outer_close": alive}
        if alive:
            try:
                group.close(timeout_seconds=float(os.environ.get("H3_T4_CLEANUP_TIMEOUT_S", "20")), force=True)
                cleanup["ok"] = True
                cleanup["termination"] = getattr(group, "termination", None)
                cleanup["closed_by"] = "load_entry"
            except Exception as exc:
                cleanup["ok"] = False
                cleanup["error_type"] = type(exc).__name__
                cleanup["message"] = str(exc)[:500]
                cleanup["termination"] = getattr(group, "termination", None)
        elif termination:
            unconfirmed = [row for row in termination if isinstance(row, dict) and row.get("confirmed") is not True]
            cleanup["ok"] = not unconfirmed
            cleanup["termination"] = termination
            cleanup["closed_by"] = "loader"
        else:
            cleanup["ok"] = False
            cleanup["reason"] = "worker termination was not recorded"
        decision["cleanup"] = cleanup


def _worker_diagnostics(group: Any) -> list[dict[str, Any]]:
    if group is None:
        return []
    from .lifecycle import remote
    from .startup import resolve_one

    rows = []
    for index, worker in enumerate(getattr(group, "workers", []) or []):
        try:
            rows.append(resolve_one(group.ray_module, remote(worker, "load_diagnostics"), 15, label=f"load_diagnostics rank {index}"))
        except Exception as exc:
            rows.append({"rank": index, "error_type": type(exc).__name__, "message": str(exc)[:300]})
    return rows


def _loaded(reports: list[dict[str, Any]]) -> bool:
    if len(reports) != 2:
        return False
    for report in reports:
        storage = report.get("storage") or {}
        if int(storage.get("materialized_bytes") or 0) <= 0:
            return False
        if int(storage.get("meta_bytes") or 0) >= int(storage.get("logical_bytes") or 0):
            return False
    return True


def _gpus() -> list[dict[str, str]] | None:
    import shutil
    import subprocess

    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    completed = subprocess.run(
        [binary, "--query-gpu=memory.used,memory.total,timestamp", "--format=csv,noheader,nounits"],
        check=False,
        text=True,
        capture_output=True,
        timeout=5,
    )
    if completed.returncode != 0:
        return None
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            rows.append({"memory_used_mib": parts[0], "memory_total_mib": parts[1], "timestamp": parts[2] if len(parts) > 2 else ""})
    return rows
