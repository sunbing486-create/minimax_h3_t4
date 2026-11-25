"""Explicit model-load gate that calls the official sampler when it is allowed.

The default retest never reaches this. A missing weight file stops at
``BLOCKED_MISSING`` before ``H3T4Sampler.sample``. A successful return from
the sampler is still not a claim that an MP4 was written.
"""

from __future__ import annotations

from typing import Any, Callable


def run_gated_model_stage(
    *,
    inventory: dict[str, Any],
    allow_model_load: bool,
    cuda_count: int,
    official_call: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    base = {
        "stage": "model-load",
        "called_loader": False,
        "called_sampler": False,
        "video_inference_claimed": False,
        "inference_memory_proven": False,
        "download": False,
    }
    if not allow_model_load:
        return {**base, "status": "BLOCKED_GATE", "reason": "H3_T4_ALLOW_MODEL_LOAD is not 1"}
    if inventory.get("status") != "REUSABLE":
        return {
            **base,
            "status": "BLOCKED_MISSING",
            "reason": "official files are missing or their sha256 does not match",
            "missing": inventory.get("missing", []),
            "next_call": "H3T4Sampler.sample",
        }
    if cuda_count != 2:
        return {**base, "status": "BLOCKED_DEVICES", "reason": f"need 2 CUDA devices, found {cuda_count}"}
    call = official_call or call_official_sampler
    try:
        result = call(inventory)
    except ImportError as exc:
        return {**base, "status": "BLOCKED_IMPORT", "error_type": "ImportError", "message": str(exc), "next_call": "H3T4Sampler.sample"}
    except Exception as exc:
        return {
            **base,
            "status": "FAILED",
            "called_loader": True,
            "called_sampler": True,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    return {
        **base,
        "status": "CALLED",
        "called_loader": True,
        "called_sampler": True,
        "result_type": type(result).__name__,
    }


def call_official_sampler(inventory: dict[str, Any]) -> Any:
    """Call the real ComfyUI node. This imports ComfyUI and does not invent frames."""

    from ..public_nodes import H3T4ModelSpec, H3T4Sampler

    unet = next(row for row in inventory["files"] if row["name"] == "unet")
    local = str(unet.get("local_path") or "")
    if not local:
        raise RuntimeError("reusable unet has no local_path")
    model = H3T4ModelSpec(local, "Exact")
    conditioning = [["", {}]]
    latent = {"samples": None}
    return H3T4Sampler().sample(
        model,
        conditioning,
        latent,
        0,
        "euler",
        "simple",
        1,
        1.0,
        False,
    )
