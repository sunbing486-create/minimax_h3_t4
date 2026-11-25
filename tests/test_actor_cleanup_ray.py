from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from src.minimax_h3_t4.runtime.lifecycle import H3T4ActorGroup, confirm_actor_exit, process_liveness
from src.minimax_h3_t4.runtime.topology import ExactH3T4Topology

try:
    import ray
except ImportError:
    ray = None


def test_gone_pid_confirms_exit_while_the_ray_probe_is_still_pending() -> None:
    class Ray:
        class exceptions:
            class RayActorError(RuntimeError):
                pass

        @staticmethod
        def wait(refs, *, num_returns, timeout):  # noqa: ANN001
            return [], list(refs)

        @staticmethod
        def get(_ref):
            raise AssertionError("a pending Ray ref is not proof of exit")

    process = subprocess.Popen(["/bin/sleep", "0"])
    process.wait(timeout=5)
    worker = type("Worker", (), {"__ray_ready__": type("Method", (), {"remote": staticmethod(lambda: "ref")})()})()
    records = confirm_actor_exit(Ray(), [worker], {0: process.pid}, time.monotonic() + 1)
    assert records[0]["confirmed"] is True
    assert records[0]["basis"] == "pid_gone"
    assert records[0]["proc_state"] == "gone"


def test_live_pid_stays_unconfirmed_for_the_whole_deadline() -> None:
    class Ray:
        class exceptions:
            class RayActorError(RuntimeError):
                pass

        @staticmethod
        def wait(refs, *, num_returns, timeout):  # noqa: ANN001
            raise AssertionError("a running PID must not be cleared by a Ray call")

    process = subprocess.Popen(["/bin/sleep", "30"])
    try:
        records = confirm_actor_exit(Ray(), [object()], {0: process.pid}, time.monotonic() + 0.3)
    finally:
        process.kill()
        process.wait(timeout=5)
    assert records[0]["confirmed"] is False
    assert records[0]["basis"] == "pid_alive"
    assert records[0]["proc_state"] == "alive"
    assert process_liveness(process.pid) == "gone"


@pytest.fixture
def ray_session():
    if ray is None:
        pytest.skip("ray is not installed")
    root = Path("/tmp/h3cu")
    if root.exists():
        subprocess.run(["rm", "-rf", str(root)], check=False)
    (root / "temp").mkdir(parents=True)
    ray.init(
        address="local",
        include_dashboard=False,
        num_cpus=2,
        object_store_memory=80 * 1024 * 1024,
        namespace="h3-cleanup",
        logging_level="ERROR",
        _temp_dir=str(root / "temp"),
    )
    try:
        yield ray
    finally:
        if ray.is_initialized():
            ray.shutdown()
        subprocess.run(["rm", "-rf", str(root)], check=False)


def test_cpu_ray_version_is_reported(ray_session) -> None:
    assert isinstance(ray_session.__version__, str) and ray_session.__version__


def test_ready_join_and_long_task_are_killed_and_pids_are_gone(ray_session) -> None:
    actors = [CleanupProbe.remote(rank) for rank in range(2)]
    ready = ray_session.get([actor.ready.remote() for actor in actors])
    joined = ray_session.get([actor.join_process_group.remote() for actor in actors])
    assert [item["status"] for item in joined] == ["joined", "joined"]
    long_task = actors[0].sleep_for.remote(30)
    group = H3T4ActorGroup(actors, ray_session, ExactH3T4Topology())
    group.note_ready(ready)
    started = time.monotonic()
    group.close(timeout_seconds=5, force=True)
    elapsed = time.monotonic() - started
    assert elapsed < 5
    assert group.alive is False
    assert all(row["confirmed"] is True for row in group.termination)
    for item in ready:
        assert process_liveness(int(item["pid"])) in {"gone", "zombie"}
    with pytest.raises(Exception):
        ray_session.get(long_task)


def test_graceful_exit_actor_is_confirmed_by_pid(ray_session) -> None:
    actors = [CleanupProbe.remote(rank) for rank in range(2)]
    ready = ray_session.get([actor.ready.remote() for actor in actors])
    group = H3T4ActorGroup(actors, ray_session, ExactH3T4Topology())
    group.note_ready(ready)
    group.close(timeout_seconds=10, force=False)
    assert all(row["confirmed"] is True for row in group.termination)
    for item in ready:
        assert process_liveness(int(item["pid"])) in {"gone", "zombie"}


def test_shutdown_that_returns_still_requires_the_process_to_exit(ray_session) -> None:
    actors = [StubbornProbe.remote(rank) for rank in range(2)]
    ready = ray_session.get([actor.ready.remote() for actor in actors])
    group = H3T4ActorGroup(actors, ray_session, ExactH3T4Topology())
    group.note_ready(ready)
    with pytest.raises(RuntimeError, match="returned without terminating"):
        group.close(timeout_seconds=5, force=False)
    assert group.alive is False
    assert all(row["confirmed"] is True for row in group.termination)
    for item in ready:
        assert process_liveness(int(item["pid"])) in {"gone", "zombie"}


@ray.remote
class CleanupProbe:
    def __init__(self, rank: int) -> None:
        self.rank = rank

    def ready(self) -> dict[str, int]:
        return {"rank": self.rank, "pid": os.getpid()}

    def join_process_group(self) -> dict[str, object]:
        return {"rank": self.rank, "status": "joined", "pid": os.getpid()}

    def sleep_for(self, seconds: float) -> int:
        time.sleep(seconds)
        return os.getpid()

    def shutdown(self) -> None:
        import ray as ray_module

        ray_module.actor.exit_actor()


@ray.remote
class StubbornProbe:
    def __init__(self, rank: int) -> None:
        self.rank = rank

    def ready(self) -> dict[str, int]:
        return {"rank": self.rank, "pid": os.getpid()}

    def shutdown(self) -> str:
        return "still-here"
