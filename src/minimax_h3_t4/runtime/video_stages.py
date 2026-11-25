"""Serial text encode, H3 sample, and video-VAE decode.

Each function is one process. The caller must not keep the text encoder,
UNet workers, and VAE alive together. A CPU call does not claim a T4 result.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .video_budget import sample_spec


def tokenizer_files(comfy_root: Path) -> list[Path]:
    directory = comfy_root / "comfy" / "text_encoders" / "qwen25_tokenizer"
    return [directory / name for name in ("vocab.json", "merges.txt", "tokenizer_config.json")]


def missing_tokenizer_files(comfy_root: Path) -> list[str]:
    return [str(path) for path in tokenizer_files(comfy_root) if not path.is_file()]


def ffmpeg_status() -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    return {
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "ok": bool(ffmpeg and ffprobe),
    }


def _cpu_conditioning(conditioning: Any) -> list[Any]:
    import torch

    moved = []
    for item in conditioning:
        tensor = item[0].detach().cpu()
        extra = item[1] if len(item) > 1 and isinstance(item[1], dict) else {}
        clean: dict[str, Any] = {}
        for key, value in extra.items():
            clean[key] = value.detach().cpu() if torch.is_tensor(value) else value
        moved.append([tensor, clean])
    return moved


def encode_text(*, weight: Path, prompt: str, destination: Path) -> dict[str, Any]:
    """Load the official text encoder, write conditioning, and return.

    The process that calls this must exit afterwards. No UNet is opened.
    """

    import comfy.sd

    destination.parent.mkdir(parents=True, exist_ok=True)
    clip = comfy.sd.load_clip([str(weight)], clip_type=comfy.sd.CLIPType.MINIMAX)
    tokens = clip.tokenize(prompt)
    conditioning = _cpu_conditioning(clip.encode_from_tokens_scheduled(tokens))
    import torch

    payload = {"prompt": prompt, "conditioning": conditioning}
    torch.save(payload, destination)
    return {
        "ok": True,
        "conditioning_path": str(destination),
        "conditioning_bytes": destination.stat().st_size,
        "items": len(conditioning),
        "audio_decoded": False,
    }


def sample_latent(
    *,
    unet_path: Path,
    conditioning_path: Path,
    destination: Path,
    comfy_root: Path,
    mode: str,
    evidence_dir: Path | None,
    deadline_s: float | None,
) -> dict[str, Any]:
    """Load the UNet on two ranks, run the real sampler, and write both latents."""

    import os
    import time

    import torch

    spec = sample_spec(mode)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        return {
            "ok": False,
            "status": "NO_DUAL_CUDA",
            "cuda": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "video_inference_claimed": False,
            "spec": spec,
        }
    import comfy_extras.nodes_custom_sampler as custom_sampler
    import comfy.samplers
    import comfy_extras.nodes_minimax_h3 as h3_nodes

    from ..loader_nodes import H3T4Initializer, H3T4UNETLoader, build_unet_runtime_env
    from ..sampler_nodes import H3T4BasicGuider, H3T4BasicScheduler, H3T4SamplerAdvanced
    from .unet_load import persist_process_sample

    if evidence_dir is not None:
        os.environ["H3_T4_EVIDENCE_DIR"] = str(evidence_dir)
        persist_process_sample(str(evidence_dir), "before_sample_load")
    saved = torch.load(conditioning_path, map_location="cpu", weights_only=False)
    conditioning = saved["conditioning"]
    latent, frame_count = h3_nodes._empty_av_latent(spec["width"], spec["height"], spec["length_frames"])
    if frame_count != spec["length_frames"]:
        raise RuntimeError(f"Comfy snapped {spec['length_frames']} frames to {frame_count}")
    group = None
    try:
        initializer = H3T4Initializer(runtime_env_builder=lambda: build_unet_runtime_env(comfy_root))
        group = initializer.initialize(load_after=conditioning, manage_parent_memory=False, for_admission=False, force_on_failure=True)[0]
        timeout = None if deadline_s is None else max(1.0, deadline_s)
        group = H3T4UNETLoader(path_resolver=lambda _name: str(unet_path)).load(
            group,
            str(unet_path),
            conditioning,
            manage_parent_memory=False,
            force_on_failure=True,
            timeout_seconds=timeout,
        )[0]
        if evidence_dir is not None:
            persist_process_sample(str(evidence_dir), "after_unet_load_before_forward")
        sigmas = H3T4BasicScheduler().get_sigmas(group, spec["scheduler"], spec["steps"], 1.0)[0]
        guider = H3T4BasicGuider().get_guider(group, conditioning)[0]
        noise = custom_sampler.Noise_RandomNoise(spec["seed"])
        sampler = comfy.samplers.sampler_object(spec["sampler"])
        started = time.monotonic()
        output, denoised = H3T4SamplerAdvanced().sample(
            True,
            noise,
            guider,
            sampler,
            sigmas,
            latent,
            close_group=False,
        )
        elapsed = time.monotonic() - started
    finally:
        if group is not None and getattr(group, "alive", False):
            group.close(timeout_seconds=30.0, force=True)
    video, audio = _latent_parts(output["samples"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "video": video.detach().cpu(),
            "audio": audio.detach().cpu(),
            "spec": spec,
            "prompt": saved.get("prompt"),
            "seed": spec["seed"],
            "sample_seconds": elapsed,
        },
        destination,
    )
    if evidence_dir is not None:
        persist_process_sample(str(evidence_dir), "after_sample")
    return {
        "ok": True,
        "latent_path": str(destination),
        "video_shape": list(video.shape),
        "audio_shape": list(audio.shape),
        "sample_seconds": elapsed,
        "spec": spec,
        "audio_decoded": False,
        "video_inference_claimed": False,
    }


def decode_video(*, vae_path: Path, latent_path: Path, mp4_path: Path) -> dict[str, Any]:
    """Decode the video latent with the official video VAE and mux a silent MP4."""

    import comfy.sd
    import comfy.utils
    import torch

    saved = torch.load(latent_path, map_location="cpu", weights_only=False)
    video = saved["video"]
    if video.ndim != 5 or video.shape[1] != 24:
        raise RuntimeError(f"video latent must be [B, 24, T, H, W], got {tuple(video.shape)}")
    state = comfy.utils.load_torch_file(str(vae_path))
    vae = comfy.sd.VAE(sd=state)
    images = vae.decode(video)
    if images.ndim == 5:
        frames = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    elif images.ndim == 4:
        frames = images
    else:
        raise RuntimeError(f"VAE decode returned {tuple(images.shape)}")
    spec = saved["spec"]
    write_mp4(frames, mp4_path, fps=int(spec["fps"]))
    probed = ffprobe_video(mp4_path)
    expected = {
        "codec_name": "h264",
        "width": int(spec["width"]),
        "height": int(spec["height"]),
        "frames": int(spec["length_frames"]),
        "fps": int(spec["fps"]),
    }
    duration = probed.get("duration")
    expected_duration = expected["frames"] / expected["fps"]
    duration_ok = duration is not None and abs(float(duration) - expected_duration) <= (1.0 / expected["fps"]) + 0.05
    matches = (
        probed.get("codec_name") == expected["codec_name"]
        and probed.get("width") == expected["width"]
        and probed.get("height") == expected["height"]
        and probed.get("nb_frames") == expected["frames"]
        and duration_ok
        and probed.get("audio_streams") == 0
        and probed.get("decode_ok") is True
    )
    return {
        "ok": matches,
        "mp4": str(mp4_path),
        "mp4_bytes": mp4_path.stat().st_size,
        "decoded_shape": list(frames.shape),
        "expected": expected,
        "ffprobe": probed,
        "audio_decoded": False,
        "audio_note": spec.get("audio_note"),
        "usable_video": False if spec.get("mode") == "smoke" else matches,
        "mechanism_smoke": spec.get("mode") == "smoke",
        "video_inference_claimed": matches,
        "placeholder": False,
    }


def write_mp4(frames: Any, path: Path, *, fps: int) -> None:
    """Mux float frames in [0, 1] with ffmpeg. The caller supplies the pixels."""

    import torch

    if not torch.is_tensor(frames) or frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(f"frames must be [N, H, W, 3], got {getattr(frames, 'shape', None)}")
    height = int(frames.shape[1])
    width = int(frames.shape[2])
    if width % 2 or height % 2:
        raise RuntimeError(f"yuv420p needs even width and height, got {width}x{height}")
    raw = (frames.detach().clamp(0, 1) * 255).to(torch.uint8).cpu().contiguous().numpy().tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    completed = subprocess.run(command, input=raw, capture_output=True, check=False)
    if completed.returncode != 0 or not path.is_file():
        raise RuntimeError(completed.stderr.decode(errors="replace")[:500])


def ffprobe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,nb_read_frames,avg_frame_rate,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {"ok": False, "message": completed.stderr[:300]}
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        check=False,
    )
    rate = str(video.get("avg_frame_rate") or "0/1")
    numerator, _, denominator = rate.partition("/")
    fps = None
    if denominator and float(denominator) != 0:
        fps = float(numerator) / float(denominator)
    frames = video.get("nb_read_frames")
    return {
        "ok": True,
        "codec_name": video.get("codec_name"),
        "width": int(video["width"]) if video.get("width") is not None else None,
        "height": int(video["height"]) if video.get("height") is not None else None,
        "nb_frames": int(frames) if frames not in (None, "N/A") else None,
        "avg_frame_rate": rate,
        "fps": fps,
        "duration": (payload.get("format") or {}).get("duration"),
        "audio_streams": sum(item.get("codec_type") == "audio" for item in streams),
        "decode_ok": decode.returncode == 0,
        "decode_error": decode.stderr.decode(errors="replace")[:300],
    }


def _latent_parts(samples: Any) -> tuple[Any, Any]:
    unbind = getattr(samples, "unbind", None)
    if getattr(samples, "is_nested", False) and callable(unbind):
        parts = tuple(unbind())
    elif isinstance(samples, (list, tuple)):
        parts = tuple(samples)
    else:
        raise RuntimeError("sampler did not return a nested AV latent")
    if len(parts) != 2:
        raise RuntimeError(f"sampler returned {len(parts)} latent parts")
    return parts[0], parts[1]
