from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

COMFY = Path("/tmp/ComfyUI-h3")


def test_time_shift_slope_matches_the_pinned_comfy_derivative() -> None:
    if not (COMFY / "comfy" / "ldm" / "minimax" / "model.py").is_file():
        raise AssertionError(f"pinned Comfy checkout is missing: {COMFY}")
    script = r"""
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--cpu"]
import torch
from comfy.ldm.minimax.model import time_shift_sigma
from minimax_h3_t4.runtime.schedule import time_shift_slope

def check(sigma, source, target):
    value = time_shift_slope(torch.tensor(sigma, dtype=torch.float64), source, target).double()
    step = 1e-5
    left = time_shift_sigma(torch.tensor(sigma - step, dtype=torch.float64), source, target)
    right = time_shift_sigma(torch.tensor(sigma + step, dtype=torch.float64), source, target)
    finite = (right - left) / (2 * step)
    return abs(float(value - finite))

same = float(time_shift_slope(torch.tensor(0.4), 5.0, 5.0))
report = {
    "same_shift": same,
    "errors": [check(0.2, 1.0, 7.0), check(0.4, 3.0, 7.0), check(0.8, 6.0, 2.0)],
    "callable": True,
}
print(json.dumps(report))
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
    assert abs(report["same_shift"] - 1.0) < 1e-5
    assert max(report["errors"]) < 1e-4


def test_pinned_comfy_load_entry_has_no_missing_symbol() -> None:
    script = r"""
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--cpu"]
from minimax_h3_t4.runtime.load_entry import import_unet_load_entry
print(json.dumps(import_unet_load_entry()))
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
    assert report["missing"] == []
    assert report["time_shift_slope_callable"] is True
    unavailable = " ".join(item["message"] for item in report["cpu_cuda_unavailable"])
    assert "time_shift_slope" not in unavailable
    assert report["ok"] is False
    assert any("h3_forward" in item["name"] for item in report["cpu_cuda_unavailable"])


def test_import_failure_still_seals_the_stage0_sample(tmp_path: Path) -> None:
    evidence = tmp_path / "h3-unet-load"
    evidence.mkdir()
    script = r"""
import os, sys
sys.path.insert(0, "/workspace/src")
os.environ["H3_T4_EVIDENCE_DIR"] = sys.argv[1]
from minimax_h3_t4.runtime.unet_load import persist_process_sample
persist_process_sample(sys.argv[1], "stage0")
persist_process_sample(sys.argv[1], "load_unet_enter", rank=0)
raise ImportError("cannot import name 'time_shift_slope'")
"""
    completed = subprocess.run([sys.executable, "-c", script, str(evidence)], cwd="/tmp", capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert "time_shift_slope" in completed.stderr
    from scripts.run_actor_session import seal_evidence

    sealed = seal_evidence(evidence)
    with zipfile.ZipFile(sealed["path"]) as archive:
        payload = archive.read("memory-samples.jsonl").decode()
    rows = [json.loads(line) for line in payload.splitlines()]
    labels = {row["label"] for row in rows}
    assert {"stage0", "load_unet_enter"} <= labels
    for row in rows:
        assert isinstance(row["rss_bytes"], int) and row["rss_bytes"] > 0
        assert isinstance(row["mem_available_bytes"], int) and row["mem_available_bytes"] > 0
        assert "gpus" in row
        assert row["sampled_max_is_not_a_continuous_profiler_peak"] is True
        assert row["weight_alignment"] == "unknown"


def test_missing_load_entry_blocks_transfer() -> None:
    from minimax_h3_t4.runtime.unet_load import worker_probe_blockers

    blockers = worker_probe_blockers(
        {
            "imports": [],
            "cuda_count": 2,
            "checkpoint_dir_writable": True,
            "load_entry": {"ok": False, "missing": ["comfy.ldm.minimax.model.time_shift_slope"]},
        }
    )
    assert any("time_shift_slope" in item for item in blockers)
    started = []
    result = {"started_transfer": False} if blockers else started.append("transfer")
    assert started == []
    assert result["started_transfer"] is False
