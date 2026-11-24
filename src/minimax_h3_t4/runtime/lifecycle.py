"""Actor-group ownership and bounded, confirmed resource release.

Apache-2.0 donor provenance is recorded in NOTICE.md.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .topology import ExactH3T4Topology


def remote(actor: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    target = getattr(actor, method)
    remote_method = getattr(target, "remote", None)
    return remote_method(*args, **kwargs) if remote_method is not None else target(*args, **kwargs)


def resolve(ray_module: Any, value: Any) -> Any:
    getter = getattr(ray_module, "get", None)
    return getter(value) if getter is not None else value


def resolve_interruptibly(
    ray_module: Any,
    refs: list[Any],
    interrupt_checker: Any,
    *,
    poll_seconds: float = 0.25,
) -> list[Any]:
    """Resolve actor refs while giving Comfy's parent interrupt flag a bounded poll point."""
    waiter = getattr(ray_module, "wait", None)
    if waiter is None:
        interrupt_checker()
        result = resolve(ray_module, refs)
        interrupt_checker()
        return result if isinstance(result, list) else [result]

    pending = list(refs)
    results: list[Any] = [None] * len(refs)
    while pending:
        interrupt_checker()
        ready, pending = waiter(
            pending,
            num_returns=1,
            timeout=poll_seconds,
        )
        for ref in ready:
            index = next(index for index, original in enumerate(refs) if results[index] is None and (ref is original or ref == original))
            results[index] = resolve(ray_module, ref)
    interrupt_checker()
    return results


def _actor_error_type(ray_module: Any) -> type[BaseException] | tuple[()]:
    error = getattr(getattr(ray_module, "exceptions", None), "RayActorError", None)
    return error if isinstance(error, type) and issubclass(error, BaseException) else ()


def _run_bounded(operation: Any, timeout_seconds: float, label: str) -> None:
    error: list[BaseException] = []

    def target() -> None:
        try:
            operation()
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=target, daemon=True, name=f"h3-t4-{label}")
    thread.start()
    thread.join(max(0.0, timeout_seconds))
    if thread.is_alive():
        raise TimeoutError(f"{label} exceeded its {timeout_seconds:.3f}s deadline")
    if error:
        raise error[0]


def _force_kill(ray_module: Any, workers: list[Any], deadline: float) -> None:
    errors: list[BaseException] = []
    threads: list[threading.Thread] = []

    def kill(worker: Any) -> None:
        try:
            ray_module.kill(worker, no_restart=True)
        except BaseException as exc:
            errors.append(exc)

    for index, worker in enumerate(workers):
        thread = threading.Thread(target=kill, args=(worker,), daemon=True, name=f"h3-t4-kill-{index}")
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    if any(thread.is_alive() for thread in threads):
        raise TimeoutError("force-killing MiniMax-H3 workers exceeded the shutdown deadline")
    if errors:
        raise errors[0]


def process_liveness(pid: int | None) -> str:
    """Return gone, zombie, alive, or unknown from the OS process table.

    ``os.kill(pid, 0)`` is not enough: a zombie still accepts that signal.
    A missing ``/proc`` entry means the process has been reaped. State ``Z``
    means it has exited and is no longer running.
    """

    if not isinstance(pid, int) or pid <= 0:
        return "unknown"
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        text = stat_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "gone"
    except OSError:
        return "unknown"
    try:
        state = text.rsplit(")", 1)[1].split()[0]
    except (IndexError, ValueError):
        return "unknown"
    if state == "Z":
        return "zombie"
    return "alive"


def _probe_actor_death(ray_module: Any, worker: Any, timeout: float) -> str:
    """One bounded Ray observation. ``dead`` is an actor error, not a successful reply."""

    actor_error = _actor_error_type(ray_module)
    try:
        ref = remote(worker, "__ray_ready__")
    except actor_error:
        return "dead"
    except BaseException:
        return "unseen"
    try:
        ready, _pending = ray_module.wait([ref], num_returns=1, timeout=max(0.0, timeout))
    except BaseException:
        return "unseen"
    if not ready:
        return "unseen"
    try:
        ray_module.get(ready[0])
    except actor_error:
        return "dead"
    except BaseException:
        return "unseen"
    return "alive"


def confirm_actor_exit(ray_module: Any, workers: list[Any], pids: dict[int, int], deadline: float) -> list[dict[str, Any]]:
    """Poll until the deadline. A live PID is not dead, even if one Ray call already returned.

    The GPU failure came from a single ``__ray_ready__`` sample after ``ray.kill``.
    Kill is asynchronous, so that sample can see one actor still answer. This
    keeps looking at ``/proc`` until the cleanup budget ends. A missing or
    zombie PID confirms exit. An unknown PID can be confirmed only by a Ray
    actor error. A running PID is never turned into success.
    """

    records: list[dict[str, Any]] = []
    for index, _worker in enumerate(workers):
        pid = pids.get(index)
        records.append(
            {
                "index": index,
                "pid": pid if isinstance(pid, int) else None,
                "confirmed": False,
                "basis": "unseen",
                "proc_state": "unknown",
            }
        )
    pending = set(range(len(workers)))
    while pending and time.monotonic() < deadline:
        finished: list[int] = []
        for index in list(pending):
            record = records[index]
            proc_state = process_liveness(record["pid"])
            record["proc_state"] = proc_state
            if proc_state in {"gone", "zombie"}:
                record["confirmed"] = True
                record["basis"] = f"pid_{proc_state}"
                finished.append(index)
                continue
            if proc_state == "alive":
                record["basis"] = "pid_alive"
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            probe = _probe_actor_death(ray_module, workers[index], min(0.2, remaining))
            record["basis"] = "ray_actor_error" if probe == "dead" else probe
            if probe == "dead":
                record["confirmed"] = True
                finished.append(index)
        for index in finished:
            pending.discard(index)
        if pending and time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    return records


@dataclass
class H3T4ActorGroup:
    workers: list[Any]
    ray_module: Any
    topology: ExactH3T4Topology = field(default_factory=ExactH3T4Topology)
    checkpoint: str | None = None
    alive: bool = True
    worker_pids: dict[int, int] = field(default_factory=dict)
    termination: list[dict[str, Any]] = field(default_factory=list)

    def note_ready(self, results: list[Any] | None) -> None:
        for item in results or []:
            if not isinstance(item, dict):
                continue
            rank = item.get("rank")
            pid = item.get("pid")
            if isinstance(rank, int) and not isinstance(rank, bool) and isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
                self.worker_pids[rank] = pid

    def require_alive(self) -> None:
        if not self.alive:
            raise RuntimeError("MiniMax-H3 two-T4 actor group is closed")

    def close(self, timeout_seconds: float = 30.0, *, force: bool = False) -> None:
        if not self.alive:
            return
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        deadline = time.monotonic() + timeout_seconds
        actor_error = _actor_error_type(self.ray_module)
        graceful_failure: BaseException | None = None
        if force:
            try:
                _force_kill(self.ray_module, self.workers, deadline)
            except BaseException as exc:
                graceful_failure = exc
        else:
            refs: list[Any] = []
            for worker in self.workers:
                try:
                    refs.append(remote(worker, "shutdown"))
                except BaseException as exc:
                    graceful_failure = graceful_failure or exc
            if refs:
                try:
                    ready, pending = self.ray_module.wait(
                        refs,
                        num_returns=len(refs),
                        timeout=max(0.0, deadline - time.monotonic()),
                    )
                    if pending:
                        graceful_failure = graceful_failure or TimeoutError(f"{len(pending)} worker shutdown RPCs exceeded the deadline")
                    for ref in ready:
                        try:
                            self.ray_module.get(ref)
                        except actor_error:
                            continue
                        except BaseException as exc:
                            graceful_failure = graceful_failure or exc
                        else:
                            graceful_failure = graceful_failure or RuntimeError("worker shutdown returned without terminating its actor")
                except BaseException as exc:
                    graceful_failure = graceful_failure or exc

            if graceful_failure is not None:
                try:
                    _force_kill(self.ray_module, self.workers, deadline)
                except BaseException as exc:
                    graceful_failure = graceful_failure or exc

        self.termination = confirm_actor_exit(self.ray_module, self.workers, self.worker_pids, deadline)
        unconfirmed = [row for row in self.termination if row["confirmed"] is not True]
        if unconfirmed:
            detail = ", ".join(
                f"index={row['index']} pid={row['pid']} basis={row['basis']} proc_state={row['proc_state']}" for row in unconfirmed
            )
            raise RuntimeError(f"Worker termination could not be confirmed for {len(unconfirmed)} actor(s): {detail}") from graceful_failure

        self.alive = False
        remaining = max(0.0, deadline - time.monotonic())
        cluster_failure: BaseException | None = None
        try:
            _run_bounded(self.ray_module.shutdown, remaining, "ray shutdown")
        except BaseException as exc:
            cluster_failure = exc

        if graceful_failure is not None:
            if cluster_failure is not None:
                graceful_failure.add_note(f"Secondary Ray shutdown failure: {type(cluster_failure).__name__}: {cluster_failure}")
            raise graceful_failure
        if cluster_failure is not None:
            raise cluster_failure
