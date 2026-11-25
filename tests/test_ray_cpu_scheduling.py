"""Real local Ray scheduling on CPU.

These tests start a Ray process, not a fake. They do not create NCCL and they
do not claim a dual-T4 result. GPU placement is not exercised here.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from src.minimax_h3_t4.runtime.evidence import EvidenceStore
from src.minimax_h3_t4.runtime.placement import plan_worker_placement
from src.minimax_h3_t4.runtime.startup import StartupTimeout, run_stage, wait_all

ray = pytest.importorskip("ray")


@pytest.fixture
def ray_cpu():
    # Ray's plasma socket must stay under the AF_UNIX 107-byte limit.
    short_dir = tempfile.mkdtemp(prefix="ry", dir="/tmp")
    ray.init(
        include_dashboard=False,
        num_cpus=2,
        num_gpus=0,
        object_store_memory=80 * 1024 * 1024,
        logging_level="ERROR",
        _memory=3 * 1024 * 1024 * 1024,
        _temp_dir=short_dir,
    )
    try:
        yield ray
    finally:
        ray.shutdown()
        shutil.rmtree(short_dir, ignore_errors=True)


def test_real_ray_schedules_two_fitting_ranks(ray_cpu, tmp_path: Path) -> None:
    resources = ray_cpu.available_resources()
    logical = int(resources["memory"])
    assert logical > 0
    assert int(resources.get("GPU", 0)) == 0
    # gpu_count=2 is the target topology's admission number. This Ray process
    # was started with num_gpus=0, so the assertion is about measured memory,
    # not about a CUDA device.
    plan = plan_worker_placement(
        world_size=2,
        available_memory_bytes=logical,
        object_store_bytes=80 * 1024 * 1024,
        gpu_count=2,
        cpu_count=float(resources.get("CPU", 0)),
        reservation_cap_bytes=logical,
        logical_memory_bytes_override=logical,
    )
    assert plan["naive_reservation_fits"] is False
    assert plan["fits"] is True, plan["reason"]
    assert plan["per_actor_memory_bytes"] * 2 <= plan["usable_memory_bytes"]

    @ray_cpu.remote
    class Rank:
        def __init__(self, rank: int) -> None:
            self.rank = rank

        def ready(self) -> dict[str, int]:
            return {"rank": self.rank, "pid": os.getpid()}

        def join_process_group(self) -> dict[str, object]:
            return {"rank": self.rank, "pid": os.getpid(), "status": "joined"}

    store = EvidenceStore(tmp_path / "run", base=tmp_path)
    workers = [Rank.options(num_cpus=1, memory=plan["per_actor_memory_bytes"]).remote(rank) for rank in (0, 1)]
    ready = run_stage(workers, ray_cpu, "ready", 20, store)
    joined = run_stage(workers, ray_cpu, "join_process_group", 20, store)
    ready_pids = {int(item["pid"]) for item in ready}
    assert len(ready_pids) == 2
    assert sorted(int(item["rank"]) for item in ready) == [0, 1]
    assert {item["status"] for item in joined} == {"joined"}
    for worker in workers:
        ray_cpu.kill(worker, no_restart=True)


def test_real_ray_leaves_the_second_oversized_actor_pending(ray_cpu) -> None:
    logical = int(ray_cpu.available_resources()["memory"])
    oversized = int(logical * 0.6)

    @ray_cpu.remote
    class Heavy:
        def ready(self) -> int:
            return os.getpid()

    first = Heavy.options(num_cpus=1, memory=oversized).remote()
    second = Heavy.options(num_cpus=1, memory=oversized).remote()
    refs = [first.ready.remote(), second.ready.remote()]
    finished, pending = ray_cpu.wait(refs, num_returns=2, timeout=15)
    assert len(finished) == 1
    assert len(pending) == 1
    ray_cpu.kill(first, no_restart=True)
    ray_cpu.kill(second, no_restart=True)


def test_real_ray_timeout_kills_the_actor(ray_cpu) -> None:
    @ray_cpu.remote(num_cpus=0.1)
    class Sleeper:
        def ready(self) -> int:
            return os.getpid()

        def hang(self) -> str:
            import time

            time.sleep(30)
            return "done"

    actor = Sleeper.remote()
    pid = ray_cpu.get(actor.ready.remote(), timeout=15)
    assert isinstance(pid, int) and pid > 0
    with pytest.raises(StartupTimeout, match="hang exceeded"):
        wait_all(ray_cpu, [actor.hang.remote()], 0.4, label="hang")
    ray_cpu.kill(actor, no_restart=True)
    import time

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            ray_cpu.get(actor.ready.remote(), timeout=1)
        except ray_cpu.exceptions.RayActorError:
            return
    raise AssertionError("Ray actor still answered after kill")
