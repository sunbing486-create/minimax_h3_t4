from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from minimax_h3_t4.runtime.model_facts import minimum_legal_sample
from minimax_h3_t4.runtime.video_budget import budget_report, sample_spec, smoke_session_budget_s
from minimax_h3_t4.runtime.video_stages import ffprobe_video, write_mp4

COMFY = Path("/tmp/ComfyUI-h3")


def test_smoke_budget_is_bounded_and_replaces_the_unet_cap() -> None:
    report = budget_report()
    assert report["unlimited"] is False
    assert report["session_budget_s"] == 5340
    assert smoke_session_budget_s() == 5340
    assert report["old_unet_load_budget_s"] == 1080
    assert report["old_budget_applies"] is False
    assert report["audio_vae_downloaded"] is False
    assert report["measured_bytes_per_second"] == 20970379616 / 167.309
    assert report["text_plus_video_vae_seconds_at_floor"] < report["stage_caps"]["download_s"]


def test_smoke_shape_is_the_pinned_minimum_and_not_a_usable_video() -> None:
    spec = sample_spec("smoke")
    legal = minimum_legal_sample()
    assert spec["width"] == legal["width"] == 32
    assert spec["height"] == legal["height"] == 32
    assert spec["length_frames"] == legal["length_frames"] == 5
    assert spec["video_latent_t"] == legal["video_latent_t"] == 2
    assert spec["audio_latent_t"] == legal["audio_latent_t"] == 8
    assert spec["steps"] == 1
    assert spec["seed"] == 1234
    assert spec["usable_video"] is False
    assert spec["audio_decoded"] is False
    trained = sample_spec("trained")
    assert trained["width"] == 736
    assert trained["height"] == 416
    assert trained["length_frames"] == 124
    assert trained["video_latent_t"] == 37
    assert trained["steps"] == 20


def test_smoke_shape_matches_pinned_comfy_temporal_shape() -> None:
    if not COMFY.is_dir():
        raise AssertionError("pinned Comfy checkout is absent")
    script = r"""
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--cpu"]
from comfy_extras.nodes_minimax_h3 import temporal_shape
from minimax_h3_t4.runtime.video_budget import sample_spec
smoke = sample_spec("smoke")
trained = sample_spec("trained")
print(json.dumps({
    "smoke": temporal_shape(smoke["length_frames"]),
    "trained": temporal_shape(trained["length_frames"]),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(COMFY), "/workspace/src"],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    smoke = sample_spec("smoke")
    trained = sample_spec("trained")
    assert report["smoke"] == [smoke["length_frames"], smoke["video_latent_t"], smoke["audio_latent_t"]]
    assert report["trained"] == [trained["length_frames"], trained["video_latent_t"], trained["audio_latent_t"]]


def test_transfer_switch_defaults_off() -> None:
    from scripts.run_video_session import build_parser

    disabled = build_parser().parse_args(["--evidence-dir", "h3-video-smoke"])
    enabled = build_parser().parse_args(["--enable-transfer", "--mode", "smoke"])
    assert disabled.enable_transfer is False
    assert disabled.mode == "smoke"
    assert enabled.enable_transfer is True
    assert disabled.budget_s == 5340


def test_ffmpeg_mux_roundtrip_is_not_a_model_video(tmp_path: Path) -> None:
    frames = torch.zeros(5, 32, 32, 3)
    frames[:, :, :, 0] = torch.linspace(0, 1, 5)[:, None, None]
    path = tmp_path / "mux-tool.mp4"
    write_mp4(frames, path, fps=24)
    probed = ffprobe_video(path)
    assert probed["codec_name"] == "h264"
    assert probed["width"] == 32
    assert probed["height"] == 32
    assert probed["nb_frames"] == 5
    assert probed["audio_streams"] == 0
    assert probed["decode_ok"] is True
    assert abs(probed["fps"] - 24) < 0.01


def test_video_watchdog_kills_on_deadline_and_low_memory(tmp_path: Path) -> None:
    from scripts.run_video_session import run_watchdog

    evidence = tmp_path / "h3-video-smoke"
    code = run_watchdog(evidence, budget_s=1.2, enable_transfer=False, comfy_dir=tmp_path / "unused", sleep_child=True)
    assert code == 2
    status = json.loads((evidence / "session-status.json").read_text(encoding="utf-8"))
    assert status["status"] == "DEADLINE"
    assert status["video_inference_claimed"] is False
    assert status["usable_video"] is False
    evidence = tmp_path / "h3-video-low"
    code = run_watchdog(
        evidence,
        budget_s=30,
        enable_transfer=False,
        comfy_dir=tmp_path / "unused",
        sleep_child=True,
        memory_reader=lambda: 1,
    )
    assert code == 2
    status = json.loads((evidence / "session-status.json").read_text(encoding="utf-8"))
    assert status["status"] == "STOP_LOW_MEMORY"
    assert status["video_inference_claimed"] is False
