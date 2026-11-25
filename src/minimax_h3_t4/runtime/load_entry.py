"""Import the modules and symbols H3T4Worker.load_unet reaches before any weight read."""

from __future__ import annotations

import importlib
from typing import Any

# Names load_unet and h3_ulysses_forward resolve on the pinned Comfy checkout.
COMFY_ATTRIBUTES = (
    ("comfy.ldm.minimax.model", ("pack_audio", "patchify_video", "rope_rotation_table", "time_shift_sigma", "unpack_audio", "unpatchify_video", "PackedLayout", "VISUAL_COND_TIMESTEP", "AUDIO_COND_TIMESTEP")),
    ("comfy.ldm.common_dit", ("pad_to_patch_size",)),
    ("comfy.model_management", ("cast_to", "get_torch_device", "unet_offload_device", "module_size")),
    ("comfy.model_detection", ("model_config_from_unet",)),
    ("comfy.model_patcher", ("ModelPatcher",)),
    ("comfy.model_base", ("MiniMaxH3",)),
    ("comfy.utils", ("convert_old_quants",)),
    ("comfy.model_prefetch", ("make_prefetch_queue", "prefetch_queue_pop")),
    ("comfy.quant_ops", ("QUANT_ALGOS", "QuantizedTensor", "get_layout_class")),
)

LOAD_MODULES = (
    "comfy.model_detection",
    "comfy.model_management",
    "comfy.model_patcher",
    "comfy.utils",
    "comfy.model_base",
    "minimax_h3_t4.runtime.schedule",
    "minimax_h3_t4.runtime.fsdp",
    "minimax_h3_t4.runtime.int8",
    "minimax_h3_t4.runtime.ulysses",
    "minimax_h3_t4.runtime.patcher",
    "minimax_h3_t4.runtime.h3_forward",
)


def import_unet_load_entry() -> dict[str, Any]:
    """Import the same modules load_unet imports. Keep the original exception."""

    missing: list[str] = []
    imported: list[str] = []
    cpu_unavailable: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for name in LOAD_MODULES:
        status = _import_module(name)
        if status["status"] == "imported":
            imported.append(name)
        elif status["status"] == "cpu_cuda_unavailable":
            cpu_unavailable.append(status)
        elif status["status"] == "missing_module":
            missing.append(name)
            failures.append(status)
        else:
            failures.append(status)
    for module_name, attributes in COMFY_ATTRIBUTES:
        module_status = _import_module(module_name)
        if module_status["status"] != "imported":
            if module_status["status"] == "cpu_cuda_unavailable":
                cpu_unavailable.append(module_status)
            elif module_status["status"] == "missing_module":
                missing.append(module_name)
                failures.append(module_status)
            else:
                failures.append(module_status)
            continue
        module = importlib.import_module(module_name)
        for attribute in attributes:
            if not hasattr(module, attribute):
                missing.append(f"{module_name}.{attribute}")
    kitchen = _import_module("comfy.quant_ops")
    if kitchen["status"] == "imported":
        ck = getattr(importlib.import_module("comfy.quant_ops"), "ck", None)
        for attribute in ("rms_rope_split_half", "rms_rope_split_half_"):
            if ck is None or not hasattr(ck, attribute):
                missing.append(f"comfy.quant_ops.ck.{attribute}")
    slope = _import_module("minimax_h3_t4.runtime.schedule")
    slope_callable = False
    if slope["status"] == "imported":
        slope_callable = callable(getattr(importlib.import_module("minimax_h3_t4.runtime.schedule"), "time_shift_slope", None))
    if not slope_callable:
        missing.append("minimax_h3_t4.runtime.schedule.time_shift_slope")
    return {
        "ok": not missing and not failures and not cpu_unavailable,
        "missing": missing,
        "failures": failures,
        "imported": imported,
        "cpu_cuda_unavailable": cpu_unavailable,
        "time_shift_slope_callable": slope_callable,
        "download": False,
        "weight_alignment": "unknown",
    }


def _import_module(name: str) -> dict[str, str]:
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        status = "missing_module" if _names_this_module(name, exc) else "dependency_error"
        return _error(name, status, exc)
    except ImportError as exc:
        return _error(name, "dependency_error", exc)
    except (NotImplementedError, AssertionError) as exc:
        text = str(exc)
        status = "cpu_cuda_unavailable" if ("CUDA" in text or "Accelerator" in text or "No Accelerators" in text) else "dependency_error"
        return _error(name, status, exc)
    except Exception as exc:
        text = str(exc)
        status = "cpu_cuda_unavailable" if ("CUDA" in text or "cuda" in text or "No Accelerators" in text) else "dependency_error"
        return _error(name, status, exc)
    return {"name": name, "status": "imported", "error_type": "", "message": "", "file": getattr(module, "__file__", None), "cause_type": "", "cause_message": ""}


def _names_this_module(name: str, exc: ModuleNotFoundError) -> bool:
    text = str(exc)
    return text in {f"No module named {name!r}", f"No module named '{name}'"} or text.startswith(f"No module named '{name}'")


def _error(name: str, status: str, exc: BaseException) -> dict[str, str]:
    cause = exc.__cause__ or exc.__context__
    return {
        "name": name,
        "status": status,
        "error_type": type(exc).__name__,
        "message": str(exc)[:500],
        "cause_type": "" if cause is None else type(cause).__name__,
        "cause_message": "" if cause is None else str(cause)[:500],
    }
