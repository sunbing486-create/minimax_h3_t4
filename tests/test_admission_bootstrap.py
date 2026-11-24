from __future__ import annotations

from src.minimax_h3_t4.loader_nodes import H3T4Initializer, build_admission_runtime_env
from tests.test_exact_nodes_contract import FakeRay


def test_admission_runtime_env_does_not_require_comfy(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("H3_T4_EVIDENCE_DIR", str(tmp_path))
    env = build_admission_runtime_env()
    assert "CUDA_VISIBLE_DEVICES" not in env["env_vars"]
    assert "COMFYUI_BASE_DIRECTORY" not in env["env_vars"]
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


def test_bootstrap_refuses_to_change_torch(monkeypatch) -> None:
    from scripts.bootstrap_actor_admission import _protected_changes

    refused = _protected_changes({"torch": "2.11.0+cu130"}, [{"metadata": {"name": "torch", "version": "2.15.0+cpu"}}])
    assert "2.11.0+cu130 would become 2.15.0+cpu" in refused
    assert _protected_changes({"torch": "2.11.0+cpu"}, [{"metadata": {"name": "torch", "version": "2.11.0+cpu"}}]) == ""
