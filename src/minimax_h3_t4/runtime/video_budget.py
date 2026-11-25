"""Bounded stage budget for one serial MiniMax-H3 smoke.

The transfer rate is the measured 5367ade UNet download. The frame grid is
the pinned ComfyUI ``temporal_shape``. This module does not download weights
and does not start a GPU.
"""

from __future__ import annotations

from typing import Any

from .model_facts import AUDIO_LATENT_FPS, align_frame_count, minimum_legal_sample, video_latent_t

# 5367ade: 20970379616 bytes in 167.309 seconds.
MEASURED_TRANSFER_BYTES = 20970379616
MEASURED_TRANSFER_SECONDS = 167.309
MEASURED_BYTES_PER_SECOND = MEASURED_TRANSFER_BYTES / MEASURED_TRANSFER_SECONDS

# Floor used only to size the hard cap. A live transfer still cancels itself
# when its own rate cannot finish before that transfer's deadline.
FLOOR_BYTES_PER_SECOND = 25 * 1024 * 1024

UNET_BYTES = 20970379616
TEXT_BYTES = 15687142551
VIDEO_VAE_BYTES = 5207808496
AUDIO_VAE_BYTES = 605254808

BOOTSTRAP_CAP_S = 600
DOWNLOAD_CAP_S = 1500
TEXT_CAP_S = 900
UNET_LOAD_CAP_S = 900
SAMPLE_CAP_S = 900
VAE_CAP_S = 420
MUX_CAP_S = 120

PROMPT = "A red paper boat floats slowly across a calm pond, gentle ripples, static camera, soft daylight."
SEED = 1234

# Official workflow canvas from the pinned node defaults' companion workflow
# numbers already recorded in model_facts. Not the first-run sample.
TRAINED_WIDTH = 736
TRAINED_HEIGHT = 416
TRAINED_FRAMES = 124
# H3T4Sampler INPUT_TYPES default. Not used by the first-run smoke.
TRAINED_STEPS = 20


def smoke_session_budget_s() -> int:
    """One hard deadline for bootstrap, downloads, text, UNet, sample, VAE, and mux."""

    return BOOTSTRAP_CAP_S + DOWNLOAD_CAP_S + TEXT_CAP_S + UNET_LOAD_CAP_S + SAMPLE_CAP_S + VAE_CAP_S + MUX_CAP_S


def stage_caps() -> dict[str, int]:
    return {
        "bootstrap_s": BOOTSTRAP_CAP_S,
        "download_s": DOWNLOAD_CAP_S,
        "text_encode_s": TEXT_CAP_S,
        "unet_load_s": UNET_LOAD_CAP_S,
        "sample_s": SAMPLE_CAP_S,
        "vae_decode_s": VAE_CAP_S,
        "mux_s": MUX_CAP_S,
        "session_s": smoke_session_budget_s(),
    }


def seconds_at_rate(nbytes: int, rate: float = MEASURED_BYTES_PER_SECOND) -> float:
    return float(nbytes) / float(rate)


def sample_spec(mode: str) -> dict[str, Any]:
    """Return a Comfy-legal sample. ``smoke`` is the first run.

    32x32, 5 frames, and 1 step come from the pinned node minima. That output
    is a mechanism smoke. ``trained`` uses the recorded 736x416x124 workflow
    and is not selected until a smoke forward has real resource evidence.
    """

    if mode not in {"smoke", "trained"}:
        raise ValueError(f"unknown video mode {mode}")
    legal = minimum_legal_sample()
    if mode == "smoke":
        spec = {
            "mode": "smoke",
            "usable_video": False,
            "label": "mechanism smoke; not a trained-length video",
            "width": int(legal["width"]),
            "height": int(legal["height"]),
            "length_frames": int(legal["length_frames"]),
            "steps": int(legal["steps"]),
        }
    else:
        frames = align_frame_count(TRAINED_FRAMES)
        spec = {
            "mode": "trained",
            "usable_video": False,
            "label": "official workflow length; activation memory on T4 is not measured",
            "width": TRAINED_WIDTH,
            "height": TRAINED_HEIGHT,
            "length_frames": frames,
            "steps": TRAINED_STEPS,
        }
    spec.update(
        {
            "fps": int(legal["fps"]),
            "video_latent_t": video_latent_t(int(spec["length_frames"])),
            "audio_latent_t": round(int(spec["length_frames"]) / int(legal["fps"]) * AUDIO_LATENT_FPS),
            "sampler": "euler",
            "scheduler": "simple",
            "seed": SEED,
            "prompt": PROMPT,
            "audio_decoded": False,
            "audio_note": "The sampler still denoises the audio latent. This run does not load the audio VAE and writes a silent MP4.",
            "source": legal["source"],
        }
    )
    return spec


def budget_report() -> dict[str, Any]:
    text_and_vae = TEXT_BYTES + VIDEO_VAE_BYTES
    return {
        "measured_unet_bytes": MEASURED_TRANSFER_BYTES,
        "measured_unet_seconds": MEASURED_TRANSFER_SECONDS,
        "measured_bytes_per_second": MEASURED_BYTES_PER_SECOND,
        "floor_bytes_per_second": FLOOR_BYTES_PER_SECOND,
        "text_encoder_bytes": TEXT_BYTES,
        "video_vae_bytes": VIDEO_VAE_BYTES,
        "audio_vae_bytes": AUDIO_VAE_BYTES,
        "audio_vae_downloaded": False,
        "text_plus_video_vae_seconds_at_measured_rate": seconds_at_rate(text_and_vae),
        "text_plus_video_vae_seconds_at_floor": seconds_at_rate(text_and_vae, FLOOR_BYTES_PER_SECOND),
        "unet_seconds_at_measured_rate": seconds_at_rate(UNET_BYTES),
        "stage_caps": stage_caps(),
        "session_budget_s": smoke_session_budget_s(),
        "old_unet_load_budget_s": 1080,
        "old_budget_applies": False,
        "unlimited": False,
        "recommended_browser_minutes": 100,
    }
