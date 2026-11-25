"""Measured header facts and the code-legal minimum MiniMax-H3 sample.

The tensor sizes come from HTTP range reads of the official safetensors
headers at Comfy-Org/MiniMax-H3 revision 4b53a35191437de05370da548e59aeabe2a5a09a.
No tensor payload was downloaded. The frame grid is the pinned ComfyUI
function ``align_frame_count`` at 2fcb6cef9ab1b6dc556b7cff1765a0724ad5e884.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

COMFYUI_REVISION = "2fcb6cef9ab1b6dc556b7cff1765a0724ad5e884"
CANVAS_MULTIPLE = 32
MIN_CANVAS = 32
MIN_FRAMES = 5
FRAME_STRIDE = 17
FPS = 24
AUDIO_LATENT_FPS = 40
MIN_STEPS = 1
# Last admission measurement. The next session must measure again.
LAST_ADMISSION_AVAILABLE_BYTES = 26594414592
LAST_ADMISSION_OBJECT_STORE_BYTES = 4 * 1024**3
LAST_WORKING_DISK_BYTES = 20957446144

_SUMMARY_PATH = Path(__file__).with_name("official_header_summary.json")


def header_summary() -> dict[str, Any]:
    return json.loads(_SUMMARY_PATH.read_text(encoding="utf-8"))


def align_frame_count(length: int) -> int:
    """Same grid as ComfyUI ``align_frame_count``: n % 17 == 5."""

    frame_count = max(MIN_FRAMES, int(length))
    while frame_count % FRAME_STRIDE != MIN_FRAMES:
        frame_count += 1
    return frame_count


def video_latent_t(frame_count: int) -> int:
    if frame_count <= MIN_FRAMES:
        return 2
    return ((frame_count - MIN_FRAMES) // FRAME_STRIDE) * 5 + 2


def minimum_legal_sample() -> dict[str, Any]:
    """Smallest sample the pinned node inputs accept. Not a trained-quality target.

    ``EmptyMiniMaxH3LatentAV`` allows width and height from 32 with step 32,
    and length from 5 with step 17. Length 5 is already on the 17k+5 grid.
    The node tooltip says the trained range starts near 124 frames. ``steps``
    on ``H3T4Sampler`` starts at 1.
    """

    frames = align_frame_count(MIN_FRAMES)
    return {
        "width": MIN_CANVAS,
        "height": MIN_CANVAS,
        "length_frames": frames,
        "fps": FPS,
        "video_latent_t": video_latent_t(frames),
        "audio_latent_t": round(frames / FPS * AUDIO_LATENT_FPS),
        "steps": MIN_STEPS,
        "sampler": "euler",
        "scheduler": "simple",
        "trained_length_note": "node tooltip says the trained range is about 124-362 frames; 124 is the official workflow length",
        "official_workflow_width": 736,
        "official_workflow_height": 416,
        "official_workflow_length": 124,
        "source": f"ComfyUI {COMFYUI_REVISION} comfy_extras/nodes_minimax_h3.py",
    }


def memory_facts(host_available_bytes: int | None = None) -> dict[str, Any]:
    summary = header_summary()
    unet = summary["minimax_h3_fl2va_pruned_int8_convrot.safetensors"]
    text = summary["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"]
    unet_bytes = int(unet["declared_payload_bytes"])
    text_bytes = int(text["declared_payload_bytes"])
    unet_max = int(unet["largest"][0]["bytes"])
    text_max = int(text["largest"][0]["bytes"])
    available = None if host_available_bytes is None else int(host_available_bytes)
    two_full = unet_bytes * 2
    packed_together = unet_bytes + text_bytes

    def fits(need: int) -> bool | None:
        if available is None:
            return None
        return need < available

    return {
        "host_available_bytes": available,
        "host_available_is_live": available is not None and available != LAST_ADMISSION_AVAILABLE_BYTES,
        "last_admission_available_bytes": LAST_ADMISSION_AVAILABLE_BYTES,
        "last_admission_object_store_bytes": LAST_ADMISSION_OBJECT_STORE_BYTES,
        "last_working_disk_bytes": LAST_WORKING_DISK_BYTES,
        "working_disk_can_hold_all_weights": LAST_WORKING_DISK_BYTES >= 42470585471,
        "unet_declared_payload_bytes": unet_bytes,
        "unet_max_tensor_bytes": unet_max,
        "unet_max_tensor_key": unet["largest"][0]["key"],
        "unet_max_tensor_dtype": unet["largest"][0]["dtype"],
        "unet_max_tensor_shape": unet["largest"][0]["shape"],
        "text_declared_payload_bytes": text_bytes,
        "text_max_tensor_bytes": text_max,
        "text_max_tensor_key": text["largest"][0]["key"],
        "text_max_tensor_dtype": text["largest"][0]["dtype"],
        "text_max_tensor_shape": text["largest"][0]["shape"],
        "one_full_unet_payload_fits": fits(unet_bytes),
        "two_full_unet_payloads_fit": fits(two_full),
        "packed_unet_plus_text_fits": fits(packed_together),
        "max_unet_tensor_fits": fits(unet_max),
        "lazy_pop_reads_payload": True,
        "load_model_weights_uses_pop": True,
        "load_model_weights_accumulates_every_pop": True,
        "model_stage_ready": False,
        "load_unet_reason": "LazyCheckpoint.pop now returns the tensor and drops its own reference. ComfyUI load_model_weights still appends every popped tensor to one dict before load_state_dict, so that call is not a one-tensor stream. load_unet does not use it; FSDP reads one key at a time. Module-construction RSS and text-encoder coexistence are not measured.",
        "nvfp4_unpacked_resident_bytes": None,
        "vae_decode_peak_bytes": None,
        "inference_memory_proven": False,
        "video_inference_claimed": False,
    }
