from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def _tiny_comfy(root: Path) -> None:
    package = root / "comfy"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "model_detection.py").write_text("MARKER = 'h3-probe'\n", encoding="utf-8")
    (root / "folder_paths.py").write_text("def get_folder_paths(kind):\n    return []\n", encoding="utf-8")


def test_isolated_process_does_not_see_the_driver_sys_path(tmp_path: Path) -> None:
    comfy = tmp_path / "ComfyUI"
    other = tmp_path / "other-cwd"
    _tiny_comfy(comfy)
    other.mkdir()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    missing = subprocess.run(
        [sys.executable, "-c", "import comfy.model_detection"],
        cwd=other,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode != 0
    assert "No module named" in (missing.stderr + missing.stdout)
    env["PYTHONPATH"] = str(comfy)
    found = subprocess.run(
        [sys.executable, "-c", "import comfy.model_detection; print(comfy.model_detection.MARKER)"],
        cwd=other,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert found.returncode == 0
    assert "h3-probe" in found.stdout


def test_ray_worker_import_uses_runtime_env_not_driver_path(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("ray")
    comfy = tmp_path / "ComfyUI"
    _tiny_comfy(comfy)
    runtime = Path("/tmp/h3wp")
    shutil.rmtree(runtime, ignore_errors=True)
    monkeypatch.setenv("H3_T4_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    from minimax_h3_t4.loader_nodes import build_admission_runtime_env, build_unet_runtime_env, probe_worker_imports

    checkpoint_dir = Path("/tmp/h3wp-ck")
    missed = probe_worker_imports(
        runtime_env=build_admission_runtime_env(),
        checkpoint_dir=checkpoint_dir,
        module_names=("comfy.model_detection",),
    )
    assert any(row["status"] == "missing_module" and row["name"] == "comfy.model_detection" for row in missed["imports"])
    assert Path(missed["cwd"]) != comfy.resolve()
    found = probe_worker_imports(
        runtime_env=build_unet_runtime_env(comfy),
        checkpoint_dir=checkpoint_dir,
        module_names=("comfy.model_detection", "folder_paths"),
    )
    detected = next(row for row in found["imports"] if row["name"] == "comfy.model_detection")
    assert detected["status"] == "imported"
    assert str(comfy.resolve()) in str(detected["file"])
    assert Path(found["cwd"]) != comfy.resolve()
    assert found["checkpoint_dir_writable"] is True


def test_worker_import_failure_blocks_transfer_without_calling_it() -> None:
    from minimax_h3_t4.runtime.unet_load import worker_probe_blockers
    from scripts.run_unet_load_session import preflight_blockers

    probe = {
        "imports": [{"name": "comfy.model_detection", "status": "missing_module"}],
        "cuda_count": 2,
        "checkpoint_dir_writable": True,
        "versions": {"torch": "2.11.0+cu130"},
        "expected_versions": {"torch": "2.11.0+cu130"},
    }
    blockers = preflight_blockers(
        {"ok": True},
        {"materials_ready_for_gpu_stage": True},
        {"ok": True},
        {"blockers": []},
        probe,
    )
    assert any("comfy.model_detection" in item for item in blockers)
    started = []

    def transfer():
        started.append("transfer")
        return {"started_transfer": True}

    result = {"started_transfer": False, "status": "STOP_PREFLIGHT"} if blockers else transfer()
    assert started == []
    assert result["started_transfer"] is False
    assert worker_probe_blockers({"imports": [], "cuda_count": 0, "checkpoint_dir_writable": True})


def test_evidence_zip_omits_weight_files(tmp_path: Path) -> None:
    from scripts.run_actor_session import seal_evidence

    evidence = tmp_path / "h3-unet-load"
    evidence.mkdir()
    (evidence / "load-result.json").write_text('{"weight_alignment":"unknown"}\n', encoding="utf-8")
    (evidence / "weight.safetensors").write_bytes(b"not-a-real-weight")
    sealed = seal_evidence(evidence)
    names = zipfile.ZipFile(sealed["path"]).namelist()
    assert "load-result.json" in names
    assert "weight.safetensors" not in names
