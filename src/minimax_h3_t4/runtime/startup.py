"""Bounded two-rank startup: ready, then collective join, then cleanup."""

from __future__ import annotations

import os
import socket
import time
from typing import Any

from .evidence import EvidenceStore
from .lifecycle import remote


class StartupTimeout(TimeoutError):
    """A startup stage exceeded its deadline. Callers must stop and keep logs."""


def env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number of seconds") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def rendezvous_endpoint(config: dict[str, Any]) -> tuple[str, str]:
    """Loopback TCP store only. A public or unspecified address is rejected."""

    master_addr = str(config.get("_master_addr") or os.environ.get("MASTER_ADDR", "127.0.0.1"))
    master_port = str(config.get("_master_port") or os.environ.get("MASTER_PORT", "29500"))
    if master_addr != "127.0.0.1":
        raise RuntimeError(f"MiniMax-H3 rendezvous must stay on 127.0.0.1, got {master_addr!r}")
    if not master_port.isdigit() or not (1 <= int(master_port) <= 65535):
        raise RuntimeError(f"MASTER_PORT is not a TCP port: {master_port!r}")
    return master_addr, master_port


def choose_master_port(preferred: int = 29500) -> int:
    """Pick a 127.0.0.1 port for the NCCL/TCP store without keeping it bound."""

    if _port_free(preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def actors_support(workers: list[Any], method: str) -> bool:
    return bool(workers) and all(getattr(worker, method, None) is not None for worker in workers)


def wait_all(
    ray_module: Any,
    refs: list[Any],
    timeout_seconds: float,
    *,
    label: str,
    interrupt_checker: Any | None = None,
    poll_seconds: float = 0.25,
) -> list[Any]:
    if not refs:
        return []
    waiter = getattr(ray_module, "wait", None)
    getter = getattr(ray_module, "get", None)
    if waiter is None:
        if interrupt_checker is not None:
            interrupt_checker()
        value = getter(refs) if getter is not None else refs
        if interrupt_checker is not None:
            interrupt_checker()
        return list(value) if isinstance(value, list) else [value]

    pending = list(refs)
    ready_all: list[Any] = []
    deadline = time.monotonic() + timeout_seconds
    while pending:
        if interrupt_checker is not None:
            interrupt_checker()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise StartupTimeout(f"{label} exceeded {timeout_seconds:.3f}s with {len(pending)} call(s) still pending")
        ready, pending = waiter(pending, num_returns=len(pending), timeout=min(poll_seconds, remaining))
        ready_all.extend(ready)
    if interrupt_checker is not None:
        interrupt_checker()
    if getter is None:
        return ready_all
    value = getter(ready_all)
    return list(value) if isinstance(value, list) else [value]


def run_stage(
    workers: list[Any],
    ray_module: Any,
    method: str,
    timeout_seconds: float,
    evidence: EvidenceStore | None,
    *,
    interrupt_checker: Any | None = None,
) -> list[Any]:
    """Run ``method`` on every worker and wait until all finish or the deadline hits.

    Test doubles that do not implement the method are treated as already past
    the stage. Production actors implement ``ready`` and ``join_process_group``.
    """

    if not actors_support(workers, method):
        if evidence is not None:
            evidence.event("stage_skipped", stage=method, reason="worker double has no method")
        return []
    if evidence is not None:
        evidence.event("stage_start", stage=method, ranks=len(workers), timeout_seconds=timeout_seconds)
    refs = [remote(worker, method) for worker in workers]
    results = wait_all(
        ray_module,
        refs,
        timeout_seconds,
        label=method,
        interrupt_checker=interrupt_checker,
    )
    if evidence is not None:
        evidence.event("stage_done", stage=method, results=[_public_result(item) for item in results])
    return results


def resolve_one(
    ray_module: Any,
    ref: Any,
    timeout_seconds: float,
    *,
    label: str,
    interrupt_checker: Any | None = None,
) -> Any:
    results = wait_all(
        ray_module,
        [ref],
        timeout_seconds,
        label=label,
        interrupt_checker=interrupt_checker,
    )
    return results[0]


def _public_result(item: Any) -> Any:
    if isinstance(item, dict):
        return {str(key): item[key] for key in list(item)[:12]}
    return str(item)[:300]
