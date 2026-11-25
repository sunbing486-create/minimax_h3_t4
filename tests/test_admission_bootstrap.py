from __future__ import annotations

from pathlib import Path

import pytest

from src.minimax_h3_t4.loader_nodes import H3T4Initializer, build_admission_runtime_env
from tests.test_exact_nodes_contract import FakeRay


def test_admission_runtime_env_does_not_require_comfy(monkeypatch, tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("H3_T4_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("H3_T4_RUNTIME_DIR", str(runtime_root))
    env = build_admission_runtime_env()
    assert "CUDA_VISIBLE_DEVICES" not in env["env_vars"]
    assert "COMFYUI_BASE_DIRECTORY" not in env["env_vars"]
    assert Path(env["working_dir"]) == (runtime_root / "working_dir").resolve()
    assert not Path(env["working_dir"]).is_relative_to(Path("src").resolve())
    ray = FakeRay()
    ray.available_resources = lambda: {"CPU": 4, "GPU": 2, "memory": 8 * 1024**3}  # type: ignore[attr-defined]
    initializer = H3T4Initializer(
        ray_module=ray,
        worker_factory=lambda _rank, _config: object(),
    )
    initializer.initialize(load_after=object(), manage_parent_memory=False, for_admission=True)
    runtime_env = ray.init_kwargs["runtime_env"]
    assert "COMFYUI_BASE_DIRECTORY" not in runtime_env["env_vars"]
    assert "CUDA_VISIBLE_DEVICES" not in runtime_env["env_vars"]


def test_runtime_dir_inside_the_source_tree_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("H3_T4_RUNTIME_DIR", str(Path("src").resolve() / "_ray_runtime_env"))
    with pytest.raises(RuntimeError, match="must not be inside the source tree"):
        build_admission_runtime_env()


def test_model_runtime_env_uses_the_writable_runtime_dir(monkeypatch, tmp_path) -> None:
    import sys
    import types

    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("H3_T4_RUNTIME_DIR", str(runtime_root))
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    (comfy / "main.py").write_text("", encoding="utf-8")
    (comfy / "execution.py").write_text("", encoding="utf-8")
    (comfy / "folder_paths.py").write_text("", encoding="utf-8")
    (comfy / "comfy").mkdir()
    (comfy / "comfy" / "model_detection.py").write_text("", encoding="utf-8")
    folder = types.ModuleType("folder_paths")
    folder.__file__ = str(comfy / "folder_paths.py")
    monkeypatch.setitem(sys.modules, "folder_paths", folder)
    from src.minimax_h3_t4.loader_nodes import _build_runtime_env

    env = _build_runtime_env()
    assert Path(env["working_dir"]) == (runtime_root / "working_dir").resolve()
    assert "CUDA_VISIBLE_DEVICES" not in env["env_vars"]


def test_bootstrap_refuses_to_change_torch(monkeypatch) -> None:
    from scripts.bootstrap_actor_admission import _protected_changes

    refused = _protected_changes({"torch": "2.11.0+cu130"}, [{"metadata": {"name": "torch", "version": "2.15.0+cpu"}}])
    assert "2.11.0+cu130 would become 2.15.0+cpu" in refused
    assert _protected_changes({"torch": "2.11.0+cpu"}, [{"metadata": {"name": "torch", "version": "2.11.0+cpu"}}]) == ""
