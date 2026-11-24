"""Admission control for the fixed two-rank Ray placement.

Ray treats ``memory`` on an actor as a hard scheduling reservation. It does not
cap RSS. On the failed two-T4 host the logical memory Ray can hand out is
smaller than two 14 GiB reservations once the 4 GiB object store (or Ray's
70% memory rule) is accounted for, so rank 1 stays pending and never becomes
a process. Rank 0 then enters its constructor alone.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

GIB = 1024**3
DEFAULT_RESERVATION_CAP_BYTES = 14 * GIB
DEFAULT_HEADROOM_BYTES = 256 * 1024**2
DEFAULT_MINIMUM_RESERVATION_BYTES = 1 * GIB

# Host figures preserved from the failed session's preflight output.
FAILED_HOST_TOTAL_BYTES = 33_658_159_104
FAILED_HOST_AVAILABLE_BYTES = 31_475_970_048
FAILED_HOST_OBJECT_STORE_BYTES = 4 * GIB


class PlacementError(RuntimeError):
    """The two ranks cannot be admitted together."""

    def __init__(self, message: str, plan: dict[str, Any]) -> None:
        super().__init__(message)
        self.plan = plan


def logical_memory_bytes(available_memory_bytes: int, object_store_bytes: int) -> int:
    """Logical Ray memory when ``object_store_memory`` is set explicitly.

    Ray's resolver uses ``available - object_store`` in that case. The default
    30/70 split is the same shape when the store is left unspecified.
    """

    return max(0, int(available_memory_bytes) - max(0, int(object_store_bytes)))


def reservation_cap_bytes() -> int:
    raw = os.environ.get("H3_T4_MAX_RAM_GB", "14")
    try:
        gib = float(raw)
    except ValueError as exc:
        raise ValueError("H3_T4_MAX_RAM_GB must be a number of gibibytes") from exc
    if gib <= 0:
        raise ValueError("H3_T4_MAX_RAM_GB must be positive")
    return int(gib * GIB)


def plan_worker_placement(
    *,
    world_size: int = 2,
    available_memory_bytes: int,
    object_store_bytes: int,
    gpu_count: int,
    cpu_count: float = 2,
    reservation_cap_bytes: int = DEFAULT_RESERVATION_CAP_BYTES,
    headroom_bytes: int = DEFAULT_HEADROOM_BYTES,
    minimum_reservation_bytes: int = DEFAULT_MINIMUM_RESERVATION_BYTES,
    logical_memory_bytes_override: int | None = None,
    logical_memory_already_excludes_object_store: bool = False,
) -> dict[str, Any]:
    """Return a JSON-ready plan. ``fits`` is false when both ranks cannot start."""

    if logical_memory_bytes_override is not None:
        logical = max(0, int(logical_memory_bytes_override))
    elif logical_memory_already_excludes_object_store:
        logical = max(0, int(available_memory_bytes))
    else:
        logical = logical_memory_bytes(available_memory_bytes, object_store_bytes)
    usable = max(0, logical - max(0, int(headroom_bytes)))
    fair = usable // world_size if world_size > 0 else 0
    cap = max(0, int(reservation_cap_bytes))
    reservation = min(cap, fair)
    naive_total = cap * world_size
    reason = "ok"
    fits = True
    if world_size < 1:
        fits = False
        reason = "world_size must be positive"
    elif int(gpu_count) < world_size:
        fits = False
        reason = f"need {world_size} GPUs, scheduler reports {int(gpu_count)}"
    elif float(cpu_count) < world_size:
        fits = False
        reason = f"need {world_size} CPUs, scheduler reports {cpu_count}"
    elif reservation < int(minimum_reservation_bytes) or fair < int(minimum_reservation_bytes):
        fits = False
        reason = (
            f"per-actor fair share {fair} bytes is below the "
            f"{int(minimum_reservation_bytes)} byte floor"
        )
    elif reservation < cap:
        reason = "clamped reservation to the schedulable fair share"
    plan = {
        "world_size": world_size,
        "available_memory_bytes": int(available_memory_bytes),
        "object_store_bytes": int(object_store_bytes),
        "logical_memory_bytes": logical,
        "headroom_bytes": int(headroom_bytes),
        "usable_memory_bytes": usable,
        "reservation_cap_bytes": cap,
        "per_actor_memory_bytes": int(reservation),
        "naive_total_bytes": naive_total,
        "naive_reservation_fits": naive_total <= usable and int(gpu_count) >= world_size and float(cpu_count) >= world_size,
        "gpu_count": int(gpu_count),
        "cpu_count": float(cpu_count),
        "fits": fits,
        "reason": reason,
    }
    return plan


def plan_from_scheduler_resources(
    resources: Mapping[str, Any],
    *,
    world_size: int = 2,
    object_store_bytes: int,
    reservation_cap_bytes: int | None = None,
) -> dict[str, Any]:
    """Plan from ``ray.available_resources()``.

    That map's ``memory`` value is already the logical budget, so the object
    store must not be subtracted a second time.
    """

    return plan_worker_placement(
        world_size=world_size,
        available_memory_bytes=int(resources.get("memory", 0)),
        object_store_bytes=object_store_bytes,
        gpu_count=int(resources.get("GPU", 0)),
        cpu_count=float(resources.get("CPU", 0)),
        reservation_cap_bytes=reservation_cap_bytes if reservation_cap_bytes is not None else reservation_cap_bytes_default(),
        logical_memory_bytes_override=int(resources.get("memory", 0)),
        logical_memory_already_excludes_object_store=True,
    )


def reservation_cap_bytes_default() -> int:
    return reservation_cap_bytes()


def assert_single_visible_gpu(device_count: int, cuda_visible_devices: str | None, rank: int) -> None:
    """Ray must isolate one GPU per actor before NCCL starts.

    ``num_gpus=1`` sets ``CUDA_VISIBLE_DEVICES`` to a single device and the
    worker then uses ``cuda:0``. A runtime env that forces ``0,1`` makes both
    ranks bind the same physical GPU. Fail before ``init_process_group``.
    """

    if int(device_count) == 1:
        return
    raise RuntimeError(
        f"rank {rank} expected exactly one visible GPU after Ray placement; "
        f"torch.cuda.device_count()={device_count}, CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}"
    )


def model_stage_allowed(smoke: Mapping[str, Any]) -> bool:
    """Weights stay blocked until a real two-GPU NCCL smoke has passed.

    CPU, gloo, and mock results are control-flow evidence only.
    """

    return bool(
        smoke.get("collective_ok") is True
        and smoke.get("mode") == "cuda-nccl"
        and int(smoke.get("device_count") or 0) == 2
        and smoke.get("weights_loaded") is False
        and smoke.get("used_mock") is not True
    )
