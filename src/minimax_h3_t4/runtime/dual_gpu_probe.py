"""Weight-free two-rank startup and collective probe.

``builtin-cpu`` spawns two processes and checks a readiness barrier plus a sum.
That is not a dual-T4 run and it does not load weights. ``cuda-nccl`` is the
only mode that may unlock the model stage, and only after one allreduce on two
visible CUDA devices. ``mock`` never does. A failure stops immediately, kills
the child processes, and leaves an evidence zip. Nothing is downloaded.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .evidence import EvidenceStore, activate
from .placement import FAILED_HOST_AVAILABLE_BYTES, FAILED_HOST_OBJECT_STORE_BYTES, model_stage_allowed, plan_worker_placement
from .startup import StartupTimeout


def cuda_device_count() -> int:
    try:
        import torch
    except Exception:
        return 0
    try:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def run_probe(
    *,
    mode: str = "auto",
    evidence_dir: str | os.PathLike[str] | None = None,
    ready_timeout: float = 60.0,
    collective_timeout: float = 90.0,
    start_method: str | None = None,
    rank_values: tuple[int, int] = (2, 3),
    hold_seconds: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    if ready_timeout <= 0 or collective_timeout <= 0:
        raise ValueError("probe timeouts must be positive")
    store = _open_store(evidence_dir)
    resolved = _resolve_mode(mode)
    store.event("probe_start", mode=resolved, requested_mode=mode, ready_timeout=ready_timeout, collective_timeout=collective_timeout)
    try:
        if resolved == "mock":
            result = _run_mock(store)
        elif resolved == "builtin-cpu":
            result = _run_processes(
                store,
                backend="builtin-cpu",
                ready_timeout=ready_timeout,
                collective_timeout=collective_timeout,
                start_method=start_method or "fork",
                rank_values=rank_values,
                hold_seconds=hold_seconds,
                device_count=0,
            )
        elif resolved == "cuda-nccl":
            devices = cuda_device_count()
            if devices != 2:
                raise RuntimeError(f"cuda-nccl probe needs 2 visible CUDA devices, found {devices}")
            result = _run_processes(
                store,
                backend="nccl",
                ready_timeout=ready_timeout,
                collective_timeout=collective_timeout,
                start_method=start_method or "spawn",
                rank_values=rank_values,
                hold_seconds=hold_seconds,
                device_count=devices,
            )
        else:
            raise ValueError(f"unknown probe mode {mode!r}")
    except BaseException as exc:
        store.record_error(exc)
        store.snapshot_processes()
        failure = {
            "mode": resolved,
            "collective_ok": False,
            "used_mock": resolved == "mock",
            "weights_loaded": False,
            "device_count": cuda_device_count() if resolved == "cuda-nccl" else 0,
            "model_stage_allowed": False,
            "video_inference_claimed": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        _publish(store, failure)
        raise
    result["model_stage_allowed"] = model_stage_allowed(result)
    result["video_inference_claimed"] = False
    result["weights_loaded"] = False
    return _publish(store, result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Weight-free two-rank startup and collective probe")
    parser.add_argument("--mode", default="auto", choices=("auto", "mock", "builtin-cpu", "cuda-nccl"))
    parser.add_argument("--evidence-dir", default=os.environ.get("H3_T4_EVIDENCE_DIR", ""))
    parser.add_argument("--ready-timeout", type=float, default=60.0)
    parser.add_argument("--collective-timeout", type=float, default=90.0)
    args = parser.parse_args(argv)
    evidence = args.evidence_dir or None
    try:
        result = run_probe(
            mode=args.mode,
            evidence_dir=evidence,
            ready_timeout=args.ready_timeout,
            collective_timeout=args.collective_timeout,
        )
    except Exception as exc:
        summary = {
            "collective_ok": False,
            "model_stage_allowed": False,
            "video_inference_claimed": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        if evidence:
            sha_path = Path(evidence) / "evidence.zip.sha256"
            if sha_path.is_file():
                summary["sha256"] = sha_path.read_text(encoding="utf-8").split()[0]
                summary["zip"] = str(Path(evidence) / "evidence.zip")
        print(json.dumps(summary, sort_keys=True))
        return 2
    printable = {key: value for key, value in result.items() if key != "bundle"}
    bundle = result.get("bundle") or {}
    printable["zip"] = bundle.get("zip")
    printable["sha256"] = bundle.get("sha256")
    print(json.dumps(printable, sort_keys=True))
    return 0 if result.get("collective_ok") and not result.get("used_mock") else 0 if result.get("mock_control_ok") else 2


def _resolve_mode(mode: str) -> str:
    if mode != "auto":
        return mode
    if cuda_device_count() == 2:
        return "cuda-nccl"
    return "builtin-cpu"


def _open_store(evidence_dir: str | os.PathLike[str] | None) -> EvidenceStore:
    if evidence_dir is None:
        return EvidenceStore.open_default()
    base = Path(evidence_dir)
    store = EvidenceStore(base / "probe", base=base)
    activate(store)
    store.event("evidence_open", run_dir=store.run_dir.name)
    return store


def _run_mock(store: EvidenceStore) -> dict[str, Any]:
    plan = plan_worker_placement(
        available_memory_bytes=FAILED_HOST_AVAILABLE_BYTES,
        object_store_bytes=FAILED_HOST_OBJECT_STORE_BYTES,
        gpu_count=2,
        cpu_count=2,
    )
    store.event("mock_plan", fits=plan["fits"], naive_reservation_fits=plan["naive_reservation_fits"])
    return {
        "mode": "mock",
        "used_mock": True,
        "mock_control_ok": bool(plan["fits"] and not plan["naive_reservation_fits"]),
        "collective_ok": False,
        "device_count": 0,
        "weights_loaded": False,
        "note": "mock control flow only; not a dual T4 and not a video result",
    }


def _run_processes(
    store: EvidenceStore,
    *,
    backend: str,
    ready_timeout: float,
    collective_timeout: float,
    start_method: str,
    rank_values: tuple[int, int],
    hold_seconds: tuple[float, float],
    device_count: int,
) -> dict[str, Any]:
    control = _fresh_control_dir(store)
    store.event("control_dir", name=control.name)
    ctx = multiprocessing.get_context(start_method)
    processes = []
    for rank in (0, 1):
        proc = ctx.Process(
            target=_probe_child,
            name=f"h3-probe-rank-{rank}",
            args=(rank, backend, str(control), rank_values[rank], hold_seconds[rank], collective_timeout),
        )
        proc.start()
        processes.append(proc)
        store.event("rank_spawn", rank=rank, pid=proc.pid)
    try:
        ready = _wait_ready(control, ready_timeout, processes, store)
        (control / "go").write_text("go\n", encoding="utf-8")
        results = _wait_results(control, collective_timeout)
    except BaseException:
        _stop(processes)
        store.snapshot_processes()
        raise
    _stop(processes)
    sums = [item.get("sum") for item in results]
    if len(results) != 2 or not all(item.get("ok") for item in results) or len(set(sums)) != 1:
        raise RuntimeError(f"{backend} collective did not agree: {results}")
    expected = rank_values[0] + rank_values[1]
    if sums[0] != expected:
        raise RuntimeError(f"{backend} collective sum {sums[0]} != {expected}")
    store.event("collective_ok", backend=backend, sum=sums[0], ready=ready)
    return {
        "mode": "cuda-nccl" if backend == "nccl" else "builtin-cpu",
        "backend": backend,
        "used_mock": False,
        "collective_ok": True,
        "device_count": device_count,
        "ready_ranks": [item["rank"] for item in ready],
        "sum": sums[0],
        "weights_loaded": False,
        "note": (
            "two-GPU NCCL allreduce only; weights were not loaded"
            if backend == "nccl"
            else "CPU process collective only; not a dual T4 and not a video result"
        ),
    }


def _fresh_control_dir(store: EvidenceStore) -> Path:
    """One empty directory per probe attempt.

    Reusing ``probe/control`` lets a later short run observe the previous
    attempt's ready and result files and report a collective that never ran.
    """

    control = store.run_dir / "controls" / uuid.uuid4().hex
    control.mkdir(parents=True, exist_ok=False)
    return control


def _wait_ready(control: Path, timeout: float, processes: list[Any], store: EvidenceStore) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    expected_pids = {int(proc.pid) for proc in processes if proc.pid is not None}
    found: dict[int, dict[str, Any]] = {}
    while len(found) < 2:
        if time.monotonic() > deadline:
            live = [proc.pid for proc in processes if proc.is_alive()]
            raise StartupTimeout(f"rank readiness exceeded {timeout:.3f}s; missing={sorted(set((0, 1)) - set(found))}; live_pids={live}")
        for rank in (0, 1):
            path = control / f"ready-{rank}.json"
            if rank in found or not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("pid", -1)) not in expected_pids:
                continue
            found[rank] = payload
            store.event("rank_ready", rank=rank, pid=payload.get("pid"))
        if len(found) < 2:
            time.sleep(0.02)
    return [found[0], found[1]]


def _wait_results(control: Path, timeout: float) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while True:
        paths = [control / f"result-{rank}.json" for rank in (0, 1)]
        if all(path.is_file() for path in paths):
            return [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        if time.monotonic() > deadline:
            raise StartupTimeout(f"collective exceeded {timeout:.3f}s")
        time.sleep(0.02)


def _stop(processes: list[Any]) -> None:
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
    for proc in processes:
        proc.join(timeout=1)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1)


def _publish(store: EvidenceStore, result: dict[str, Any]) -> dict[str, Any]:
    store.write_payload("smoke_result.json", result)
    store._write_json(store.base / "smoke_result.json", result)
    store.event("probe_published", mode=result.get("mode"), collective_ok=result.get("collective_ok"))
    result = dict(result)
    result["bundle"] = store.quick_save()
    return result


def _probe_child(rank: int, backend: str, control_dir: str, value: int, hold_seconds: float, collective_timeout: float) -> None:
    control = Path(control_dir)
    if hold_seconds > 0:
        time.sleep(hold_seconds)
    _atomic_json(control / f"ready-{rank}.json", {"rank": rank, "pid": os.getpid()})
    go = control / "go"
    if not _wait_for(go, collective_timeout):
        _atomic_json(control / f"result-{rank}.json", {"rank": rank, "ok": False, "error": "go-timeout"})
        return
    if backend == "builtin-cpu":
        _atomic_json(control / f"value-{rank}.json", {"rank": rank, "value": value})
        paths = [control / "value-0.json", control / "value-1.json"]
        if not all(_wait_for(path, collective_timeout) for path in paths):
            _atomic_json(control / f"result-{rank}.json", {"rank": rank, "ok": False, "error": "peer-value-timeout"})
            return
        total = sum(json.loads(path.read_text(encoding="utf-8"))["value"] for path in paths)
        _atomic_json(control / f"result-{rank}.json", {"rank": rank, "ok": True, "sum": total, "saw_go": True})
        return
    if backend == "nccl":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        _nccl_allreduce(rank, value, collective_timeout, control)
        return
    raise RuntimeError(f"unsupported probe backend {backend}")


def _nccl_allreduce(rank: int, value: int, collective_timeout: float, control: Path) -> None:
    import torch
    import torch.distributed as dist
    from datetime import timedelta

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", os.environ.get("H3_T4_PROBE_PORT", "29511"))
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=collective_timeout),
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    )
    try:
        torch.cuda.set_device(0)
        tensor = torch.tensor([value], dtype=torch.int64, device="cuda:0")
        dist.all_reduce(tensor)
        _atomic_json(control / f"result-{rank}.json", {"rank": rank, "ok": True, "sum": int(tensor.item()), "saw_go": True})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _wait_for(path: Path, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    raise SystemExit(main())
