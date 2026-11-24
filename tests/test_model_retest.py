from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.model_retest import main as retest_main
from src.minimax_h3_t4.loader_nodes import H3T4Initializer, H3T4UNETLoader
from src.minimax_h3_t4.runtime.actor_admission import require_actor_admission_file, run_actor_admission, summarize_actor_join
from src.minimax_h3_t4.runtime.lifecycle import H3T4ActorGroup
from src.minimax_h3_t4 import model_catalog
from src.minimax_h3_t4.model_catalog import OFFICIAL_TOTAL_BYTES, inventory_local_models, memory_budget
from src.minimax_h3_t4.runtime.topology import ExactH3T4Topology
from tests.test_exact_nodes_contract import FakeRay, RemoteMethod


def _join(rank: int, uuid: str, visible: str, sum_value: int = 5) -> dict:
    return {
        "rank": rank,
        "status": "joined",
        "pid": 1000 + rank,
        "sum": sum_value,
        "cuda_visible_devices": visible,
        "device_name": "Tesla T4",
        "device_uuid": uuid,
        "weights_loaded": False,
    }


def test_join_summary_accepts_only_two_distinct_t4_allreduces() -> None:
    good = summarize_actor_join([_join(1, "GPU-B", "1"), _join(0, "GPU-A", "0")])
    assert good["collective_ok"] is True
    assert good["sum"] == 5
    assert good["inference_memory_proven"] is False
    same = summarize_actor_join([_join(0, "GPU-A", "0"), _join(1, "GPU-A", "1")])
    assert same["collective_ok"] is False
    probe_shaped = summarize_actor_join([{"rank": 0, "status": "joined"}, {"rank": 1, "status": "joined"}])
    assert probe_shaped["collective_ok"] is False


def test_actor_admission_uses_initializer_and_does_not_load_weights(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("H3_T4_EVIDENCE_DIR", str(tmp_path))
    events: list[tuple[str, int]] = []
    ray = FakeRay()
    ray.available_resources = lambda: {"CPU": 4, "GPU": 2, "memory": 8 * 1024**3}  # type: ignore[attr-defined]

    def factory(rank: int, _config: dict) -> object:
        worker = type("Worker", (), {})()
        worker.ready = RemoteMethod(lambda: {"rank": rank, "pid": 1000 + rank, "cuda_visible_devices": str(rank)})
        worker.join_process_group = RemoteMethod(lambda: events.append(("join", rank)) or _join(rank, f"GPU-{rank}", str(rank)))
        worker.shutdown = RemoteMethod(lambda: f"actor-exited:{rank}")
        worker.__ray_ready__ = RemoteMethod(lambda: f"actor-exited:{rank}")
        worker.load_unet = RemoteMethod(lambda *_args: (_ for _ in ()).throw(AssertionError("weights loaded")))
        return worker

    initializer = H3T4Initializer(ray_module=ray, worker_factory=factory, runtime_env_builder=lambda: {})
    report = run_actor_admission(evidence_dir=tmp_path, initializer=initializer)

    assert [event[0] for event in events] == ["join", "join"]
    assert report["collective_ok"] is True
    assert report["weights_loaded"] is False
    saved = json.loads((tmp_path / "actor_admission.json").read_text(encoding="utf-8"))
    assert saved["kind"] == "ray-h3t4-actor-admission"
    assert saved["model_stage_allowed"] is False


def test_subprocess_probe_file_does_not_satisfy_actor_admission(tmp_path: Path) -> None:
    probe = tmp_path / "smoke_result.json"
    probe.write_text(json.dumps({"mode": "cuda-nccl", "collective_ok": True, "distinct_gpu_mapping": True, "weights_loaded": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="actor admission"):
        require_actor_admission_file(str(probe))


def test_loader_blocks_until_actor_admission_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("H3_T4_REQUIRE_ACTOR_ADMISSION", "1")
    monkeypatch.setenv("H3_T4_ACTOR_ADMISSION_FILE", str(tmp_path / "missing.json"))
    calls: list[str] = []
    group = H3T4ActorGroup(workers=[], ray_module=FakeRay(), topology=ExactH3T4Topology())
    loader = H3T4UNETLoader(path_resolver=lambda name: calls.append(name) or name)
    with pytest.raises(RuntimeError, match="actor admission"):
        loader.load(group, "unet.safetensors", load_after=object(), manage_parent_memory=False)
    assert calls == []


def test_inventory_marks_missing_without_hashing_a_different_size(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    wrong = root / "minimax_h3_audio_vae_fp32.safetensors"
    wrong.write_bytes(b"short")
    report = inventory_local_models([root])
    audio = next(row for row in report["files"] if row["name"] == "audio_vae")
    assert report["status"] == "MISSING"
    assert audio["reason"] == "size_mismatch"
    assert audio["sha256"] is None
    assert report["inference_allowed"] is False
    assert report["download"] is False


def test_inventory_reuses_only_matching_bytes_and_hash(tmp_path: Path, monkeypatch) -> None:
    payload = b"abc"
    path = tmp_path / "vae" / "tiny.safetensors"
    path.parent.mkdir()
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    spec = {"name": "audio_vae", "path": "vae/tiny.safetensors", "sha256": digest, "bytes": len(payload)}
    monkeypatch.setattr(model_catalog, "OFFICIAL_MODELS", (spec,))
    assert inventory_local_models([tmp_path])["status"] == "REUSABLE"
    wrong = dict(spec)
    wrong["sha256"] = "0" * 64
    monkeypatch.setattr(model_catalog, "OFFICIAL_MODELS", (wrong,))
    mismatch = inventory_local_models([tmp_path])
    assert mismatch["status"] == "MISSING"
    assert mismatch["files"][0]["reason"] == "sha256_mismatch"


def test_memory_budget_does_not_treat_nccl_as_room_for_the_unet() -> None:
    budget = memory_budget(logical_memory_bytes=31475970048, device_memory_bytes=15636037632)
    assert budget["unet_exceeds_one_gpu"] is True
    assert budget["two_unet_copies_exceed_host"] is True
    assert budget["inference_memory_proven"] is False
    assert budget["admission_does_not_authorize_model_load"] is True
    assert OFFICIAL_TOTAL_BYTES == 42470585471


def test_preflight_cli_reports_missing_and_does_not_admit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("H3_T4_MODEL_ROOTS", raising=False)
    code = retest_main(["--stage", "preflight", "--search", str(tmp_path), "--evidence-dir", str(tmp_path / "out")])
    assert code == 0
    report = json.loads((tmp_path / "out" / "preflight.json").read_text(encoding="utf-8"))
    assert report["inventory"]["status"] == "MISSING"
    assert report["inference_allowed"] is False
    assert report["download"] is False
    assert len(report["download_list_if_missing"]) == 4
    assert "huggingface.co/Comfy-Org/MiniMax-H3/resolve/" in report["download_list_if_missing"][0]["url"]


def test_actor_admission_cli_does_not_run_without_two_gpus(tmp_path: Path) -> None:
    code = retest_main(["--stage", "actor-admission", "--search", str(tmp_path), "--evidence-dir", str(tmp_path / "out")])
    assert code == 2
    report = json.loads((tmp_path / "out" / "preflight.json").read_text(encoding="utf-8"))
    assert report["status"] == "NOT_RUN"
    assert report["inference_allowed"] is False


def test_model_load_stays_off_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("H3_T4_ALLOW_MODEL_LOAD", raising=False)
    code = retest_main(["--stage", "model-load", "--search", str(tmp_path), "--evidence-dir", str(tmp_path / "out")])
    assert code == 0
    report = json.loads((tmp_path / "out" / "preflight.json").read_text(encoding="utf-8"))
    assert report["status"] == "NOT_RUN"
    assert "H3_T4_ALLOW_MODEL_LOAD" in report["reason"]
