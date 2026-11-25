"""CPU/mock coverage for two-rank admission, startup order, timeouts, and logs.

Nothing in this module starts an NCCL job or claims a dual-T4 video result.
"""

from __future__ import annotations

import ast
import hashlib
import json
import multiprocessing
import zipfile
from pathlib import Path

import pytest

from src.minimax_h3_t4.loader_nodes import H3T4Initializer, H3T4UNETLoader, build_runtime_env_vars
from src.minimax_h3_t4.runtime.dual_gpu_probe import _gpu_mapping
from src.minimax_h3_t4.runtime.dual_gpu_probe import main as probe_main
from src.minimax_h3_t4.runtime.dual_gpu_probe import run_probe
from src.minimax_h3_t4.runtime.evidence import EvidenceStore
from src.minimax_h3_t4.runtime.lifecycle import H3T4ActorGroup
from src.minimax_h3_t4.runtime.placement import (
    FAILED_HOST_AVAILABLE_BYTES,
    FAILED_HOST_OBJECT_STORE_BYTES,
    FAILED_HOST_TOTAL_BYTES,
    assert_single_visible_gpu,
    logical_memory_bytes,
    model_stage_allowed,
    plan_worker_placement,
)
from src.minimax_h3_t4.runtime.startup import StartupTimeout, rendezvous_endpoint
from src.minimax_h3_t4.runtime.topology import ExactH3T4Topology
from src.minimax_h3_t4.runtime.weights import inspect_local_weight
from tests.test_exact_nodes_contract import FakeRay, FakeWorker, RemoteMethod

ROOT = Path(__file__).parents[1]
GIB = 1024**3


def test_failed_host_cannot_admit_two_14gib_reservations() -> None:
    logical = logical_memory_bytes(FAILED_HOST_AVAILABLE_BYTES, FAILED_HOST_OBJECT_STORE_BYTES)
    plan = plan_worker_placement(
        available_memory_bytes=FAILED_HOST_AVAILABLE_BYTES,
        object_store_bytes=FAILED_HOST_OBJECT_STORE_BYTES,
        gpu_count=2,
        cpu_count=2,
        reservation_cap_bytes=14 * GIB,
    )
    seventy_available = int(FAILED_HOST_AVAILABLE_BYTES * 0.70)
    seventy_total = int(FAILED_HOST_TOTAL_BYTES * 0.70)

    assert logical < 28 * GIB
    assert seventy_available < 28 * GIB
    assert seventy_total < 28 * GIB
    assert plan["naive_reservation_fits"] is False
    assert plan["fits"] is True
    assert plan["reason"] == "clamped reservation to the schedulable fair share"
    assert plan["per_actor_memory_bytes"] < 14 * GIB
    assert plan["per_actor_memory_bytes"] * 2 <= plan["usable_memory_bytes"]

    starved = plan_worker_placement(
        available_memory_bytes=2 * GIB,
        object_store_bytes=0,
        gpu_count=2,
        cpu_count=2,
        reservation_cap_bytes=14 * GIB,
        logical_memory_already_excludes_object_store=True,
    )
    assert starved["fits"] is False


def test_gpu_mapping_requires_two_different_device_identities() -> None:
    same = _gpu_mapping(
        [
            {"rank": 0, "cuda_visible_devices": "0", "device_uuid": "GPU-A", "device_name": "Tesla T4"},
            {"rank": 1, "cuda_visible_devices": "1", "device_uuid": "GPU-A", "device_name": "Tesla T4"},
        ]
    )
    distinct = _gpu_mapping(
        [
            {"rank": 0, "cuda_visible_devices": "0", "device_uuid": "GPU-A", "device_name": "Tesla T4"},
            {"rank": 1, "cuda_visible_devices": "1", "device_uuid": "GPU-B", "device_name": "Tesla T4"},
        ]
    )

    assert same["distinct"] is False
    assert distinct["distinct"] is True
    assert model_stage_allowed(
        {
            "mode": "cuda-nccl",
            "collective_ok": True,
            "device_count": 2,
            "weights_loaded": False,
            "used_mock": False,
            "distinct_gpu_mapping": False,
        }
    ) is False


def test_rendezvous_stays_on_loopback() -> None:
    assert rendezvous_endpoint({"_master_addr": "127.0.0.1", "_master_port": 29500}) == ("127.0.0.1", "29500")
    with pytest.raises(RuntimeError, match="127.0.0.1"):
        rendezvous_endpoint({"_master_addr": "0.0.0.0", "_master_port": 29500})


def test_visible_gpu_check_rejects_the_runtime_env_override() -> None:
    assert_single_visible_gpu(1, "0", 1)
    with pytest.raises(RuntimeError, match="exactly one visible GPU"):
        assert_single_visible_gpu(2, "0,1", 0)


def test_runtime_env_does_not_pin_both_gpus() -> None:
    env = build_runtime_env_vars(pythonpath="/comfy", comfy_root="/comfy", evidence_dir="/evidence/run")

    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["MASTER_ADDR"] == "127.0.0.1"
    assert env["H3_T4_EVIDENCE_DIR"] == "/evidence/run"
    assert '"CUDA_VISIBLE_DEVICES": "0,1"' not in (ROOT / "src/minimax_h3_t4/loader_nodes.py").read_text(encoding="utf-8")


def test_worker_constructor_does_not_enter_the_process_group() -> None:
    module = ast.parse((ROOT / "src/minimax_h3_t4/runtime/worker.py").read_text(encoding="utf-8"))
    worker = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "H3T4Worker")
    init = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    calls = [
        node.func.attr
        for node in ast.walk(init)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "_initialize_distributed" not in calls
    assert "init_process_group" not in calls


def test_initializer_clamps_reservation_and_starts_both_ranks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("H3_T4_EVIDENCE_DIR", str(tmp_path))
    events: list[tuple[str, int]] = []
    logical = logical_memory_bytes(FAILED_HOST_AVAILABLE_BYTES, FAILED_HOST_OBJECT_STORE_BYTES)
    ray = FakeRay()
    ray.available_resources = lambda: {"CPU": 4, "GPU": 2, "memory": logical}  # type: ignore[attr-defined]
    initializer = H3T4Initializer(
        ray_module=ray,
        worker_factory=lambda rank, _config: (events.append(("created", rank)), _stage_worker(rank, events))[1],
        runtime_env_builder=lambda: {},
    )

    (group,) = initializer.initialize(load_after=object(), manage_parent_memory=False)

    assert [event for event in events if event[0] == "created"] == [("created", 0), ("created", 1)]
    assert [event for event in events if event[0] in {"ready", "join"}] == [
        ("ready", 0),
        ("ready", 1),
        ("join", 0),
        ("join", 1),
    ]
    assert group.alive is True
    assert initializer.last_placement is not None
    assert initializer.last_placement["naive_reservation_fits"] is False
    assert initializer.last_placement["fits"] is True
    assert "CUDA_VISIBLE_DEVICES" not in ray.init_kwargs["runtime_env"].get("env_vars", {})


def test_initializer_refuses_unschedulable_rank_and_keeps_logs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("H3_T4_EVIDENCE_DIR", str(tmp_path))
    created: list[int] = []
    ray = FakeRay()
    ray.available_resources = lambda: {"CPU": 4, "GPU": 1, "memory": 30 * GIB}  # type: ignore[attr-defined]
    initializer = H3T4Initializer(
        ray_module=ray,
        worker_factory=lambda rank, _config: created.append(rank),
        runtime_env_builder=lambda: {},
    )

    with pytest.raises(RuntimeError, match="need 2 GPUs"):
        initializer.initialize(load_after=object(), manage_parent_memory=False, force_on_failure=True)

    assert created == []
    _assert_bundle(tmp_path, "need 2 GPUs")


def test_ready_timeout_kills_actors_before_join(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("H3_T4_EVIDENCE_DIR", str(tmp_path))
    monkeypatch.setenv("H3_T4_ACTOR_READY_TIMEOUT_S", "0.3")
    monkeypatch.setenv("H3_T4_CLEANUP_TIMEOUT_S", "0.4")
    events: list[tuple[str, int]] = []
    ray = _PendingRay()
    _release_scheduler_on_close(monkeypatch)
    initializer = H3T4Initializer(
        ray_module=ray,
        worker_factory=lambda rank, _config: _stage_worker(rank, events),
        runtime_env_builder=lambda: {},
    )

    with pytest.raises(StartupTimeout, match="ready exceeded"):
        initializer.initialize(load_after=object(), manage_parent_memory=False)

    assert ("join", 0) not in events
    assert ("join", 1) not in events
    assert len(ray.killed) == 2
    assert all(call["no_restart"] is True for call in ray.killed)
    _assert_bundle(tmp_path, "ready exceeded")


def test_model_load_timeout_closes_the_group(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("H3_T4_MODEL_LOAD_TIMEOUT_S", "0.3")
    monkeypatch.setenv("H3_T4_CLEANUP_TIMEOUT_S", "0.4")
    monkeypatch.delenv("H3_T4_REQUIRE_SMOKE", raising=False)
    events: list[object] = []
    ray = _PendingRay()
    _release_scheduler_on_close(monkeypatch)
    group = H3T4ActorGroup(workers=[FakeWorker(0, events), FakeWorker(1, events)], ray_module=ray, topology=ExactH3T4Topology())
    loader = H3T4UNETLoader(path_resolver=lambda name: f"/models/{name}")

    with pytest.raises(StartupTimeout, match="load_unet rank 0"):
        loader.load(group, "h3.safetensors", load_after=object(), manage_parent_memory=False, force_on_failure=True)

    assert group.alive is False
    assert [event for event in events if isinstance(event, tuple) and event[0] == "load"][0][1] == 0


def test_smoke_gate_blocks_weights_until_real_nccl_probe(monkeypatch, tmp_path: Path) -> None:
    smoke = tmp_path / "smoke_result.json"
    monkeypatch.setenv("H3_T4_REQUIRE_SMOKE", "1")
    monkeypatch.setenv("H3_T4_SMOKE_FILE", str(smoke))
    calls: list[str] = []

    def resolver(name: str) -> str:
        calls.append(name)
        return name

    blocked = H3T4UNETLoader(path_resolver=resolver)
    group = _group()
    with pytest.raises(RuntimeError, match="model stage blocked"):
        blocked.load(group, "h3.safetensors", load_after=object(), manage_parent_memory=False, force_on_failure=True)
    assert calls == []
    assert group.alive is False

    smoke.write_text(json.dumps({"mode": "builtin-cpu", "collective_ok": True, "device_count": 0, "weights_loaded": False}), encoding="utf-8")
    group = _group()
    with pytest.raises(RuntimeError, match="cuda-nccl"):
        blocked.load(group, "h3.safetensors", load_after=object(), manage_parent_memory=False, force_on_failure=True)
    assert calls == []

    smoke.write_text(
        json.dumps(
            {
                "mode": "cuda-nccl",
                "collective_ok": True,
                "device_count": 2,
                "weights_loaded": False,
                "used_mock": False,
                "distinct_gpu_mapping": False,
            }
        ),
        encoding="utf-8",
    )
    assert model_stage_allowed(json.loads(smoke.read_text(encoding="utf-8"))) is False
    group = _group()
    with pytest.raises(RuntimeError, match="model stage blocked"):
        blocked.load(group, "h3.safetensors", load_after=object(), manage_parent_memory=False, force_on_failure=True)
    assert calls == []

    smoke.write_text(
        json.dumps(
            {
                "mode": "cuda-nccl",
                "collective_ok": True,
                "device_count": 2,
                "weights_loaded": False,
                "used_mock": False,
                "distinct_gpu_mapping": True,
            }
        ),
        encoding="utf-8",
    )
    assert model_stage_allowed(json.loads(smoke.read_text(encoding="utf-8"))) is True
    group = _group()
    blocked.load(group, "h3.safetensors", load_after=object(), manage_parent_memory=False)
    assert calls == ["h3.safetensors"]


def test_failure_evidence_is_incremental_and_sha_verified(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "run", base=tmp_path)
    store.event("placement", fits=False, reason="rank 1 unschedulable")
    flushed = (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8")
    assert "rank 1 unschedulable" in flushed
    store.record_error(RuntimeError("ready exceeded 0.300s"))
    store.record_history({"status_str": "error", "messages": ["execution error body"]}, prompt_id="prompt-test")
    store.record_model_metadata({"name": "unet", "bytes": 3, "sha256": "abc", "reusable": False, "reason": "missing"})
    store.snapshot_processes(limit=5)
    after_error = (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8")
    assert "ready exceeded" in after_error
    bundle = store.quick_save()

    digest = hashlib.sha256(Path(bundle["zip"]).read_bytes()).hexdigest()
    assert digest == bundle["sha256"]
    assert digest == (tmp_path / "evidence.zip.sha256").read_text(encoding="utf-8").split()[0]
    with zipfile.ZipFile(bundle["zip"]) as archive:
        names = set(archive.namelist())
        assert {"events.jsonl", "status.json", "error.txt", "history.json", "model_metadata.json", "processes.json"} <= names
        history = json.loads(archive.read("history.json"))
        processes = json.loads(archive.read("processes.json"))
    assert history["status"] == "error"
    assert "execution error body" in history["error"]
    assert processes["count"] <= 5
    assert all(len(row["cmd"]) <= 160 for row in processes["processes"])


def test_local_weights_reuse_requires_bytes_on_disk_and_matching_hash(tmp_path: Path) -> None:
    missing = inspect_local_weight(name="unet", path=str(tmp_path / "missing.safetensors"), expected_sha256="ab")
    assert missing["reusable"] is False
    assert missing["reason"] == "missing"
    payload = b"not-a-real-weight"
    path = tmp_path / "unet.safetensors"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    assert inspect_local_weight(name="unet", path=str(path), expected_sha256=None)["reason"] == "sha256_required"
    assert inspect_local_weight(name="unet", path=str(path), expected_sha256="0" * 64)["reason"] == "sha256_mismatch"
    match = inspect_local_weight(name="unet", path=str(path), expected_sha256=digest, expected_bytes=len(payload))
    assert match["reusable"] is True
    probe_source = (ROOT / "src/minimax_h3_t4/runtime/dual_gpu_probe.py").read_text(encoding="utf-8")
    weight_source = (ROOT / "src/minimax_h3_t4/runtime/weights.py").read_text(encoding="utf-8")
    assert "hf_hub_download" not in probe_source
    assert "hf_hub_download" not in weight_source
    assert "urlopen" not in weight_source


def test_builtin_cpu_probe_is_not_a_dual_t4_result(tmp_path: Path) -> None:
    result = run_probe(
        mode="builtin-cpu",
        evidence_dir=tmp_path,
        ready_timeout=5,
        collective_timeout=5,
        rank_values=(2, 3),
    )

    assert result["collective_ok"] is True
    assert result["sum"] == 5
    assert result["ready_ranks"] == [0, 1]
    assert result["model_stage_allowed"] is False
    assert result["video_inference_claimed"] is False
    assert result["weights_loaded"] is False
    assert "not a dual T4" in result["note"]
    assert result["bundle"]["sha256"] == hashlib.sha256(Path(result["bundle"]["zip"]).read_bytes()).hexdigest()
    assert multiprocessing.active_children() == []


def test_probe_rerun_does_not_reuse_previous_control_files(tmp_path: Path) -> None:
    first = run_probe(mode="builtin-cpu", evidence_dir=tmp_path, ready_timeout=5, collective_timeout=5)
    assert first["collective_ok"] is True

    with pytest.raises(StartupTimeout, match="rank readiness exceeded"):
        run_probe(
            mode="builtin-cpu",
            evidence_dir=tmp_path,
            ready_timeout=0.15,
            collective_timeout=0.15,
            hold_seconds=(30.0, 30.0),
        )

    smoke = json.loads((tmp_path / "smoke_result.json").read_text(encoding="utf-8"))
    assert smoke["collective_ok"] is False
    assert smoke["model_stage_allowed"] is False
    controls = list((tmp_path / "probe" / "controls").iterdir())
    assert len(controls) == 2
    assert all(path.is_dir() for path in controls)
    assert multiprocessing.active_children() == []


def test_probe_timeout_stops_and_keeps_a_verifiable_bundle(tmp_path: Path) -> None:
    with pytest.raises(StartupTimeout, match="rank readiness exceeded"):
        run_probe(
            mode="builtin-cpu",
            evidence_dir=tmp_path,
            ready_timeout=0.4,
            collective_timeout=1,
            hold_seconds=(0.0, 30.0),
        )

    _assert_bundle(tmp_path, "rank readiness exceeded")
    smoke = json.loads((tmp_path / "smoke_result.json").read_text(encoding="utf-8"))
    assert smoke["collective_ok"] is False
    assert smoke["model_stage_allowed"] is False
    assert smoke["video_inference_claimed"] is False
    assert multiprocessing.active_children() == []


def test_mock_and_cuda_probe_do_not_unlock_weights_without_two_gpus(tmp_path: Path) -> None:
    mock = run_probe(mode="mock", evidence_dir=tmp_path / "mock")
    assert mock["used_mock"] is True
    assert mock["mock_control_ok"] is True
    assert mock["collective_ok"] is False
    assert mock["model_stage_allowed"] is False
    assert "not a dual T4" in mock["note"]

    with pytest.raises(RuntimeError, match="2 visible CUDA devices"):
        run_probe(mode="cuda-nccl", evidence_dir=tmp_path / "cuda")
    smoke = json.loads((tmp_path / "cuda" / "smoke_result.json").read_text(encoding="utf-8"))
    assert smoke["collective_ok"] is False
    assert model_stage_allowed(smoke) is False


def test_probe_cli_prints_sha_and_exits_on_cuda_refusal(tmp_path: Path) -> None:
    assert probe_main(["--mode", "mock", "--evidence-dir", str(tmp_path / "mock")]) == 0
    assert probe_main(["--mode", "cuda-nccl", "--evidence-dir", str(tmp_path / "cuda")]) == 2
    assert (
        probe_main(
            [
                "--mode",
                "builtin-cpu",
                "--evidence-dir",
                str(tmp_path / "hold"),
                "--ready-timeout",
                "0.2",
                "--collective-timeout",
                "0.2",
                "--hold-seconds",
                "30",
            ]
        )
        == 2
    )
    held = json.loads((tmp_path / "hold" / "smoke_result.json").read_text(encoding="utf-8"))
    assert held["collective_ok"] is False
    assert held["distinct_gpu_mapping"] is False
    sha = (tmp_path / "cuda" / "evidence.zip.sha256").read_text(encoding="utf-8").split()[0]
    assert sha == hashlib.sha256((tmp_path / "cuda" / "evidence.zip").read_bytes()).hexdigest()


class _PendingRay(FakeRay):
    def __init__(self) -> None:
        super().__init__()
        self.killed: list[dict[str, object]] = []
        self.block = True

    def release(self) -> None:
        self.block = False

    def wait(self, refs, *, num_returns=None, timeout=None):  # noqa: ANN001
        if self.block:
            return [], list(refs)
        return list(refs), []

    def kill(self, worker, *, no_restart):  # noqa: ANN001
        assert no_restart is True
        self.killed.append({"worker": worker, "no_restart": no_restart})


def _release_scheduler_on_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake scheduler blocks until cleanup starts, then reports the actors dead."""

    original = H3T4ActorGroup.close

    def close(self: H3T4ActorGroup, timeout_seconds: float = 30.0, *, force: bool = False) -> None:
        release = getattr(self.ray_module, "release", None)
        if release is not None:
            release()
        original(self, timeout_seconds, force=force)

    monkeypatch.setattr(H3T4ActorGroup, "close", close)


def _stage_worker(rank: int, events: list[tuple[str, int]]) -> object:
    worker = FakeWorker(rank, [])
    worker.ready = RemoteMethod(lambda: events.append(("ready", rank)) or {"rank": rank, "pid": rank})
    worker.join_process_group = RemoteMethod(lambda: events.append(("join", rank)) or {"rank": rank, "status": "joined"})
    return worker


def _group() -> H3T4ActorGroup:
    return H3T4ActorGroup(workers=[FakeWorker(0, []), FakeWorker(1, [])], ray_module=FakeRay(), topology=ExactH3T4Topology())


def _assert_bundle(base: Path, snippet: str) -> None:
    zip_path = base / "evidence.zip"
    sha_path = base / "evidence.zip.sha256"
    assert zip_path.is_file()
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    assert digest == sha_path.read_text(encoding="utf-8").split()[0]
    with zipfile.ZipFile(zip_path) as archive:
        errors = archive.read("error.txt").decode("utf-8")
        status = json.loads(archive.read("status.json"))
    assert snippet in errors
    assert status["updated_at"]
