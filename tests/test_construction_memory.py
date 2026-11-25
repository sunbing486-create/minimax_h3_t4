from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

PYTHON = Path("/tmp/h3-cpu-venv/bin/python")
COMFY = Path("/tmp/ComfyUI-h3/comfy/ldm/minimax/model.py")


@pytest.fixture(scope="module")
def construction_report() -> dict[str, object]:
    if not PYTHON.is_file() or not COMFY.is_file():
        pytest.skip("CPU torch venv or pinned ComfyUI checkout is not available")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/tmp/ComfyUI-h3:/workspace/src"
    completed = subprocess.run(
        [str(PYTHON), "scripts/audit_construction_memory.py"],
        check=False,
        text=True,
        capture_output=True,
        env=env,
        cwd="/workspace",
    )
    assert completed.returncode == 0, completed.stderr[-1500:]
    return json.loads(completed.stdout)


def _case(report: dict[str, object], name: str) -> dict[str, object]:
    found = next(row for row in report["cases"] if row["case"] == name)
    assert found["ok"] is True
    return found


def test_cpu_construction_allocates_and_meta_does_not_match_that_rss(construction_report: dict[str, object]) -> None:
    cpu = _case(construction_report, "medium-cpu")
    meta = _case(construction_report, "medium-meta")
    logical = int(cpu["audit"]["logical_parameter_bytes"])
    assert logical > 1_000_000
    assert int(cpu["pss_delta_bytes"]) > logical // 2
    assert int(meta["pss_delta_bytes"]) < int(cpu["pss_delta_bytes"])
    assert int(meta["audit"]["meta_parameter_bytes"]) == logical


def test_moving_to_meta_does_not_return_the_cpu_allocation(construction_report: dict[str, object]) -> None:
    row = _case(construction_report, "medium-cpu-then-meta")
    assert int(row["rss_after_to_meta_bytes"]) >= int(row["rss_after_construct_bytes"]) - 1_000_000


def test_real_architecture_meta_bytes_are_not_rss(construction_report: dict[str, object]) -> None:
    row = _case(construction_report, "real-meta-bfloat16")
    assert int(row["audit"]["logical_parameter_bytes"]) > 60_000_000_000
    assert int(row["rss_delta_bytes"]) < 50_000_000
    assert construction_report["model_stage_ready"] is False


def test_holding_shard_tensors_keeps_rss_and_delete_releases_it(construction_report: dict[str, object]) -> None:
    row = _case(construction_report, "shard-dict-committed")
    held = int(row["while_holding_128MiB"]["rss_bytes"]) - int(row["before"]["rss_bytes"])
    released = int(row["after_delete"]["rss_bytes"])
    assert held > 100_000_000
    assert released < int(row["while_holding_128MiB"]["rss_bytes"]) - 50_000_000


def test_aimdo_skips_linear_storage(construction_report: dict[str, object]) -> None:
    assert _case(construction_report, "tiny-aimdo-on")["weight_is_none"] is True
    assert _case(construction_report, "tiny-aimdo-off")["weight_is_none"] is False


def test_loader_constructs_on_meta() -> None:
    source = Path("src/minimax_h3_t4/runtime/worker.py").read_text(encoding="utf-8")
    assert 'config.get_model(state_dict, "", torch.device("meta"))' in source
