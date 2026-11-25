"""Official MiniMax-H3 file list and local reuse checks.

This module never downloads. A missing file stays missing.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .runtime.weights import inspect_local_weight

# Comfy-Org/MiniMax-H3 revision verified by the earlier official download.
# Sizes sum to 42470585471 bytes. Paths are repo-relative.
OFFICIAL_REPO = "Comfy-Org/MiniMax-H3"
OFFICIAL_REVISION = "4b53a35191437de05370da548e59aeabe2a5a09a"
OFFICIAL_MODELS: tuple[dict[str, Any], ...] = (
    {
        "name": "unet",
        "path": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "sha256": "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
        "bytes": 20970379616,
    },
    {
        "name": "text_encoder",
        "path": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "sha256": "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
        "bytes": 15687142551,
    },
    {
        "name": "video_vae",
        "path": "vae/minimax_h3_video_vae_fp16.safetensors",
        "sha256": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
        "bytes": 5207808496,
    },
    {
        "name": "audio_vae",
        "path": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "sha256": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
        "bytes": 605254808,
    },
)
OFFICIAL_TOTAL_BYTES = 42470585471

# Last measured Tesla T4 and host figures. Live preflight must measure again.
PREVIOUS_T4_TOTAL_BYTES = 15636037632
PREVIOUS_HOST_TOTAL_BYTES = 33658159104
PREVIOUS_HOST_AVAILABLE_BYTES = 31475970048
PREVIOUS_WORKING_TOTAL_BYTES = 20957446144


def official_download_list() -> list[dict[str, Any]]:
    """Reference list only. Nothing in this function contacts the network."""

    rows = []
    for item in OFFICIAL_MODELS:
        rows.append(
            {
                "repo": OFFICIAL_REPO,
                "revision": OFFICIAL_REVISION,
                "path": item["path"],
                "sha256": item["sha256"],
                "bytes": item["bytes"],
                "url": f"https://huggingface.co/{OFFICIAL_REPO}/resolve/{OFFICIAL_REVISION}/{item['path']}",
            }
        )
    return rows


def inventory_local_models(roots: list[Path]) -> dict[str, Any]:
    files = []
    for spec in OFFICIAL_MODELS:
        filename = Path(str(spec["path"])).name
        matches = _find(roots, filename)
        if not matches:
            files.append(
                {
                    "name": spec["name"],
                    "path": spec["path"],
                    "status": "MISSING",
                    "reusable": False,
                    "reason": "missing",
                    "bytes": None,
                    "sha256": None,
                    "expected_bytes": spec["bytes"],
                    "expected_sha256": spec["sha256"],
                }
            )
            continue
        sized = [path for path in matches if path.is_file() and path.stat().st_size == int(spec["bytes"])]
        if not sized:
            files.append(
                {
                    "name": spec["name"],
                    "path": spec["path"],
                    "status": "MISSING",
                    "reusable": False,
                    "reason": "size_mismatch",
                    "bytes": matches[0].stat().st_size if matches[0].is_file() else None,
                    "sha256": None,
                    "expected_bytes": spec["bytes"],
                    "expected_sha256": spec["sha256"],
                    "candidates": [str(path) for path in matches],
                }
            )
            continue
        checked = []
        for path in sized:
            row = inspect_local_weight(
                name=str(spec["name"]),
                path=str(path),
                expected_sha256=str(spec["sha256"]),
                expected_bytes=int(spec["bytes"]),
            )
            row["local_path"] = str(path)
            checked.append(row)
        reusable = next((row for row in checked if row["reusable"]), None)
        chosen = reusable or checked[0]
        files.append(
            {
                **chosen,
                "path": spec["path"],
                "status": "REUSABLE" if chosen["reusable"] else "MISSING",
                "expected_bytes": spec["bytes"],
                "expected_sha256": spec["sha256"],
            }
        )
    missing = [row["path"] for row in files if row["status"] != "REUSABLE"]
    return {
        "repo": OFFICIAL_REPO,
        "revision": OFFICIAL_REVISION,
        "files": files,
        "status": "REUSABLE" if not missing else "MISSING",
        "missing": missing,
        "inference_allowed": False,
        "download": False,
    }


def disk_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    usage = shutil.disk_usage(path)
    return {"path": str(path), "exists": True, "total": usage.total, "free": usage.free, "used": usage.used}


def memory_budget(
    *,
    logical_memory_bytes: int,
    device_memory_bytes: int,
    unet_bytes: int = 20970379616,
    text_encoder_bytes: int = 15687142551,
) -> dict[str, Any]:
    """Compare file sizes with host and one GPU. This is not a measured runtime peak."""

    return {
        "logical_memory_bytes": int(logical_memory_bytes),
        "device_memory_bytes": int(device_memory_bytes),
        "unet_file_bytes": int(unet_bytes),
        "text_encoder_file_bytes": int(text_encoder_bytes),
        "unet_exceeds_one_gpu": int(unet_bytes) > int(device_memory_bytes),
        "text_encoder_exceeds_one_gpu": int(text_encoder_bytes) > int(device_memory_bytes),
        "two_unet_copies_exceed_host": int(unet_bytes) * 2 > int(logical_memory_bytes),
        "working_disk_previously_smaller_than_weights": PREVIOUS_WORKING_TOTAL_BYTES < OFFICIAL_TOTAL_BYTES,
        "admission_does_not_authorize_model_load": True,
        "inference_memory_proven": False,
        "note": "File size is not resident RSS. FSDP plus CPU offload was not measured in the NCCL probe.",
    }


def _find(roots: list[Path], filename: str) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.name == filename:
            found.append(root)
            continue
        if root.is_dir():
            found.extend(path for path in root.rglob(filename) if path.is_file())
    return found
