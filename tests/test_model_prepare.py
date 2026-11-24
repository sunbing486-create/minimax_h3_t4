from __future__ import annotations

import ast
import json
from pathlib import Path

from minimax_h3_t4.runtime.checkpoint_budget import LazyCheckpoint
from minimax_h3_t4.runtime.model_facts import LAST_ADMISSION_AVAILABLE_BYTES, align_frame_count, memory_facts, minimum_legal_sample
from scripts.prepare_model_stage import build_plan, clone_comfy, refuse_or_plan_download


class _Tensor:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes


def test_header_facts_use_the_measured_largest_tensors() -> None:
    facts = memory_facts(LAST_ADMISSION_AVAILABLE_BYTES)
    assert facts["unet_max_tensor_bytes"] == 308281344
    assert facts["unet_max_tensor_dtype"] == "BF16"
    assert facts["unet_max_tensor_shape"] == [28672, 5376]
    assert facts["text_max_tensor_bytes"] == 777912320
    assert facts["text_max_tensor_dtype"] == "I8"
    assert facts["one_full_unet_payload_fits"] is True
    assert facts["two_full_unet_payloads_fit"] is False
    assert facts["packed_unet_plus_text_fits"] is False
    assert facts["max_unet_tensor_fits"] is True
    assert facts["lazy_pop_reads_payload"] is True
    assert facts["load_model_weights_accumulates_every_pop"] is True
    assert facts["model_stage_ready"] is False
    assert facts["inference_memory_proven"] is False
    assert facts["nvfp4_unpacked_resident_bytes"] is None
    assert facts["working_disk_can_hold_all_weights"] is False


def test_minimum_legal_sample_follows_the_pinned_frame_grid() -> None:
    sample = minimum_legal_sample()
    assert sample["width"] == 32
    assert sample["height"] == 32
    assert sample["length_frames"] == 5
    assert sample["steps"] == 1
    assert sample["video_latent_t"] == 2
    assert sample["audio_latent_t"] == 8
    assert align_frame_count(6) == 22
    assert align_frame_count(124) == 124


def test_pop_returns_the_tensor_and_drops_the_dict_reference() -> None:
    sizes = {"a": 10, "b": 30, "c": 5}
    lazy = LazyCheckpoint(lambda key: _Tensor(sizes[key]), list(sizes))
    first = lazy.pop("b")
    assert first.nbytes == 30
    assert lazy.resident_bytes == 0
    assert "b" not in lazy
    assert lazy.peak_bytes == 30


def test_plan_does_not_download_or_claim_admission(tmp_path: Path) -> None:
    report = build_plan(search=[tmp_path])
    assert report["download"] is False
    assert report["actor_admission_passed"] is False
    assert report["video_inference_claimed"] is False
    assert report["model_stage_allowed"] is False
    assert report["weights_status"]["status"] == "MISSING"


def test_download_stays_off_without_explicit_confirmation(tmp_path: Path) -> None:
    refused = refuse_or_plan_download(confirm=False, only="minimax_h3_audio_vae_fp32.safetensors", dest_dir=str(tmp_path))
    assert refused["download"] is False
    assert refused["started_transfer"] is False
    assert refused["ok"] is False


def test_model_prepare_notebook_compiles() -> None:
    notebook = json.loads(Path("deploy/kaggle_actor_admission/kaggle_model_prepare.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 1
    ast.parse(cells[0])
    compile(cells[0], "<notebook>", "exec")
    assert "confirm-download" not in cells[0]
    assert "prepare_model_stage.py" in cells[0]


def test_existing_pinned_comfy_checkout_is_reused() -> None:
    checkout = Path("/tmp/ComfyUI-h3")
    if not (checkout / ".git").is_dir():
        return
    report = clone_comfy(checkout)
    assert report["ok"] is True
    assert report["revision"] == "2fcb6cef9ab1b6dc556b7cff1765a0724ad5e884"
    assert report["download"] is False
    assert report["action"] == "reused"
