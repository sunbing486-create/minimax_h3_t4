from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from src.minimax_h3_t4.runtime.checkpoint_budget import (
    LazyCheckpoint,
    load_row_shard,
    read_safetensors_header,
    sequential_resident_plan,
)
from src.minimax_h3_t4.runtime.model_stage import run_gated_model_stage

torch = pytest.importorskip("torch")


def _write_f32(path: Path, rows: int, cols: int) -> None:
    payload = struct.pack("<" + "f" * (rows * cols), *[float(i) for i in range(rows * cols)])
    header = {"weight": {"dtype": "F32", "shape": [rows, cols], "data_offsets": [0, len(payload)]}}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_header_read_does_not_require_the_tensor_payload_to_stay_resident(tmp_path: Path) -> None:
    path = tmp_path / "tiny.safetensors"
    _write_f32(path, 8, 4)
    header = read_safetensors_header(path)
    weight = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    assert header["total_bytes"] == weight.nbytes
    assert header["header_bytes_read"] < path.stat().st_size
    rank0 = load_row_shard(weight, 0, 2)
    rank1 = load_row_shard(weight, 1, 2)
    assert rank0.shape == (4, 4)
    assert rank1.shape == (4, 4)
    assert rank0.nbytes + rank1.nbytes == weight.nbytes


def test_lazy_release_keeps_only_the_current_tensor() -> None:
    def loader(key: str) -> torch.Tensor:
        return torch.ones(8, dtype=torch.float32) + (1 if key == "b" else 0)

    lazy = LazyCheckpoint(loader, ["a", "b"])
    first = lazy["a"]
    assert lazy.peak_bytes == first.nbytes
    lazy["a"] = None
    second = lazy["b"]
    assert lazy.resident_bytes == second.nbytes
    assert lazy.peak_bytes == first.nbytes
    assert lazy.materialized_keys == ["a", "b"]


def test_official_unet_second_copy_does_not_fit_previous_host() -> None:
    plan = sequential_resident_plan(
        file_bytes=20970379616,
        max_tensor_bytes=1024**3,
        host_available_bytes=31475970048,
    )
    assert plan["old_second_copy_fits"] is False
    assert plan["streaming_second_rank_fits"] is True
    assert plan["inference_memory_proven"] is False


def test_gated_stage_calls_the_official_sampler_only_when_files_match() -> None:
    missing = run_gated_model_stage(inventory={"status": "MISSING", "missing": ["unet"]}, allow_model_load=True, cuda_count=2)
    assert missing["status"] == "BLOCKED_MISSING"
    assert missing["called_sampler"] is False
    calls: list[str] = []

    def official(_inventory: dict) -> str:
        calls.append("H3T4Sampler.sample")
        return "latent-handle"

    ready = run_gated_model_stage(
        inventory={"status": "REUSABLE", "files": [{"name": "unet", "local_path": "/models/unet.safetensors"}]},
        allow_model_load=True,
        cuda_count=2,
        official_call=official,
    )
    assert calls == ["H3T4Sampler.sample"]
    assert ready["status"] == "CALLED"
    assert ready["video_inference_claimed"] is False
    assert json.dumps(ready)
