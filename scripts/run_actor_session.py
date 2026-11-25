#!/usr/bin/env python3
"""One entry point for plan, apply, check, then actor admission.

Check timeout is fixed at 75 seconds and is never reduced. Actor admission
uses ready 45s, collective 60s, and cleanup 20s, and starts only when the
check saw two CUDA devices and at least 125 seconds remain. This process
does not download weights and does not claim a GPU result on its own.

``actor_admission_passed`` is true only when the admission file records a real
collective, distinct GPU mapping, no weights, a closed cleanup, and the actor
process exits 0. A file claim with a non-zero exit stays visible and is not a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CHECK_TIMEOUT_S = 75
READY_TIMEOUT_S = 45
COLLECTIVE_TIMEOUT_S = 60
CLEANUP_TIMEOUT_S = 20
ACTOR_BUDGET_S = READY_TIMEOUT_S + COLLECTIVE_TIMEOUT_S + CLEANUP_TIMEOUT_S
DEFAULT_BUDGET_S = 240
PASS_RULE = (
    "actor_admission_passed requires kind ray-h3t4-actor-admission, collective_ok, "
    "distinct_gpu_mapping, weights_loaded false, cleanup ok, returncode 0, and no timeout"
)
if CHECK_TIMEOUT_S < 60:
    raise RuntimeError("check timeout must stay at least 60 seconds")

StageRunner = Callable[[Path, str], dict[str, Any]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded actor-admission session")
    parser.add_argument("--evidence-dir", default=os.environ.get("H3_T4_EVIDENCE_DIR", "h3-actor-session"))
    parser.add_argument("--budget-s", type=float, default=DEFAULT_BUDGET_S)
    args = parser.parse_args(argv)
    evidence = Path(args.evidence_dir)
    code = run_session(evidence, budget_s=args.budget_s)
    show_session_evidence(evidence)
    return code


def run_session(
    evidence: Path,
    *,
    budget_s: float = DEFAULT_BUDGET_S,
    stage_runner: StageRunner | None = None,
    actor_argv: list[str] | None = None,
    actor_timeout_s: float | None = None,
) -> int:
    evidence.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = time.monotonic() + budget_s
    state: dict[str, Any] = {
        "stage": "session",
        "status": "RUNNING",
        "actor_admission_passed": False,
        "gpu_stage_started": False,
        "weights_downloaded": False,
        "check_timeout_s": CHECK_TIMEOUT_S,
        "ready_timeout_s": READY_TIMEOUT_S,
        "collective_timeout_s": COLLECTIVE_TIMEOUT_S,
        "cleanup_timeout_s": CLEANUP_TIMEOUT_S,
        "budget_s": budget_s,
        "pass_rule": PASS_RULE,
        "bundle_status": "NOT_SEALED",
    }
    exit_code = 2
    try:
        _event(evidence, "session_start", budget_s=budget_s, check_timeout_s=CHECK_TIMEOUT_S)
        _persist(evidence, state)
        state["plan"] = _run_named_stage(evidence, "plan", deadline, minimum_s=5, timeout_s=40, stage_runner=stage_runner)
        _persist(evidence, state)
        if not _ok(state["plan"]):
            state["status"] = "PLAN_FAILED"
            exit_code = 2
            return exit_code
        state["apply"] = _run_named_stage(evidence, "apply", deadline, minimum_s=5, timeout_s=90, stage_runner=stage_runner)
        _persist(evidence, state)
        if not _ok(state["apply"]):
            state["status"] = "APPLY_FAILED"
            exit_code = 2
            return exit_code
        if _remaining(deadline) < CHECK_TIMEOUT_S:
            state["check"] = {"status": "NOT_RUN_BUDGET", "check_timeout_s": CHECK_TIMEOUT_S, "remaining_s": round(_remaining(deadline), 3)}
            state["actor"] = {"status": "NOT_RUN_BUDGET", "actor_admission_passed": False}
            state["status"] = "NOT_RUN_BUDGET"
            _event(evidence, "not_run_budget", remaining_s=state["check"]["remaining_s"], need_s=CHECK_TIMEOUT_S)
            exit_code = 2
            return exit_code
        state["check"] = _run_named_stage(
            evidence, "check", deadline, minimum_s=CHECK_TIMEOUT_S, timeout_s=CHECK_TIMEOUT_S, stage_runner=stage_runner
        )
        _persist(evidence, state)
        if not _ok(state["check"]):
            state["status"] = "CHECK_FAILED"
            state["actor"] = {"status": "NOT_RUN_CHECK", "actor_admission_passed": False}
            exit_code = 2
            return exit_code
        permitted = bool(state["check"].get("gpu_stage_permitted"))
        state["gpu_stage_permitted"] = permitted
        if not permitted:
            state["actor"] = {"status": "NOT_RUN_GPU_GATE", "actor_admission_passed": False}
            state["status"] = "NOT_RUN_GPU_GATE"
            _event(evidence, "gpu_gate_closed")
            exit_code = 2
            return exit_code
        if _remaining(deadline) < ACTOR_BUDGET_S:
            state["actor"] = {
                "status": "NOT_RUN_BUDGET",
                "remaining_s": round(_remaining(deadline), 3),
                "need_s": ACTOR_BUDGET_S,
                "actor_admission_passed": False,
            }
            state["status"] = "NOT_RUN_BUDGET"
            _event(evidence, "actor_not_run_budget", remaining_s=state["actor"]["remaining_s"])
            exit_code = 2
            return exit_code
        state["gpu_stage_started"] = True
        state["status"] = "ACTOR_RUNNING"
        state["actor_admission_passed"] = False
        _persist(evidence, state)
        state["actor"] = _run_actor(evidence, deadline, actor_argv=actor_argv, actor_timeout_s=actor_timeout_s)
        outcome = classify_actor_outcome(
            returncode=state["actor"].get("returncode"),
            timed_out=bool(state["actor"].get("timed_out")),
            admission=state["actor"].get("admission"),
        )
        state.update(outcome)
        exit_code = 0 if outcome["actor_admission_passed"] else 2
        return exit_code
    except Exception as exc:
        state["status"] = "FAILED"
        state["actor_admission_passed"] = False
        state["error_type"] = type(exc).__name__
        state["message"] = str(exc)[:500]
        _event(evidence, "session_error", error_type=type(exc).__name__, message=str(exc)[:500])
        exit_code = 2
        return exit_code
    finally:
        state["elapsed_s"] = round(time.time() - started, 3)
        state["actor_admission_passed"] = state.get("actor_admission_passed") is True
        state["snapshot"] = "pre-seal"
        state["bundle_status"] = "NOT_SEALED"
        state["bundle_note"] = "pre-seal snapshot stored inside the evidence zip; the sibling manifest is the final record"
        _persist(evidence, state)
        _event(evidence, "session_finish", status=state.get("status"), elapsed_s=state["elapsed_s"])
        try:
            state["bundle"] = seal_evidence(evidence)
            state["bundle_status"] = "SEALED"
            state.pop("bundle_error", None)
        except Exception as exc:
            state["bundle_status"] = "FAILED"
            state["bundle_error"] = {"error_type": type(exc).__name__, "message": str(exc)[:500]}
        state["snapshot"] = "final"
        _persist(evidence, state)
        _write(bundle_paths(evidence)["manifest"], state)


def classify_actor_outcome(*, returncode: int | None, timed_out: bool, admission: dict[str, Any] | None) -> dict[str, Any]:
    """Keep a file claim, a process exit, and the session pass bit as separate facts."""

    present = isinstance(admission, dict)
    collective = bool(present and admission.get("collective_ok") is True)
    distinct = bool(present and admission.get("distinct_gpu_mapping") is True)
    weights_clear = bool(present and admission.get("weights_loaded") is False)
    cleanup_ok = bool(present and isinstance(admission.get("cleanup"), dict) and admission["cleanup"].get("ok") is True)
    claim = collective and distinct
    passed = (
        timed_out is False
        and returncode == 0
        and present
        and admission.get("kind") == "ray-h3t4-actor-admission"
        and admission.get("used_mock") is False
        and claim
        and weights_clear
        and cleanup_ok
    )
    if timed_out:
        status = "ACTOR_TIMEOUT"
    elif returncode not in (0, None) and claim:
        status = "ACTOR_FAILED_AFTER_CLAIM"
    elif returncode not in (0, None):
        status = "ACTOR_FAILED"
    elif passed:
        status = "ACTOR_ADMITTED"
    else:
        status = "ACTOR_EXIT_ZERO_UNPROVEN"
    return {
        "status": status,
        "actor_admission_passed": passed,
        "admission_file_present": present,
        "admission_file_collective_ok": collective,
        "admission_file_distinct_gpu_mapping": distinct,
        "cleanup_ok": cleanup_ok,
        "process_returncode": returncode,
        "process_timed_out": timed_out,
        "pass_rule": PASS_RULE,
    }


def show_session_evidence(evidence: Path) -> dict[str, Any]:
    """Print the small status backup and the exact bundle files to download."""

    evidence = Path(evidence)
    status_path = evidence / "session-status.json"
    actor_path = evidence / "actor-result.json"
    bundle = bundle_paths(evidence)
    files = [bundle["manifest"], status_path, actor_path, bundle["zip"], bundle["sidecar"]]
    listing = []
    if bundle["manifest"].is_file():
        print("FINAL_MANIFEST", flush=True)
        print(bundle["manifest"].read_text(encoding="utf-8"), end="", flush=True)
    else:
        print("MISSING " + bundle["manifest"].name, flush=True)
    if status_path.is_file():
        print("SESSION_STATUS", flush=True)
        print(status_path.read_text(encoding="utf-8"), end="", flush=True)
    else:
        print("MISSING session-status.json", flush=True)
    load_path = evidence / "load-result.json"
    inline_path = evidence / "inline-evidence.json"
    files.extend([load_path, inline_path, *sorted(evidence.rglob("rank*.log"))])
    if actor_path.is_file():
        print(actor_path.read_text(encoding="utf-8"), end="", flush=True)
    elif load_path.is_file():
        print("actor-result.json is not written by the UNet runner. load-result.json has the load error, ranks, and cleanup.", flush=True)
    else:
        print("MISSING actor-result.json", flush=True)
    if inline_path.is_file():
        print("BEGIN_INLINE_EVIDENCE source=runner-cell-output sealed_zip_is_original=true", flush=True)
        print(inline_path.read_text(encoding="utf-8"), end="", flush=True)
        print("END_INLINE_EVIDENCE", flush=True)
    for path in files:
        if path.is_file():
            listing.append({"name": path.name, "path": str(path), "bytes": path.stat().st_size, "missing": False})
            print(f"FILE {path.name} bytes={path.stat().st_size} path={path}", flush=True)
            print("DOWNLOAD_BEFORE_STOP " + _link_path(path), flush=True)
        else:
            listing.append({"name": path.name, "path": str(path), "bytes": None, "missing": True})
            print(f"MISSING {path}", flush=True)
    if bundle["sidecar"].is_file():
        print(bundle["sidecar"].read_text(encoding="utf-8"), end="", flush=True)
    try:
        from IPython.display import FileLink, display
    except Exception:
        return {"files": listing}
    for item in listing:
        if item["missing"]:
            continue
        display(FileLink(_link_path(Path(item["path"]))))
    return {"files": listing}


def bundle_paths(evidence: Path) -> dict[str, Path]:
    """Bundle sits beside the evidence directory, so the pack scan cannot include it."""

    evidence = Path(evidence)
    zip_path = evidence.parent / f"{evidence.name}-evidence.zip"
    return {
        "zip": zip_path,
        "sidecar": Path(str(zip_path) + ".sha256"),
        "manifest": evidence.parent / f"{evidence.name}-manifest.json",
    }


def seal_evidence(evidence: Path) -> dict[str, Any]:
    paths = bundle_paths(evidence)
    zip_path = paths["zip"]
    sidecar = paths["sidecar"]
    excluded = {zip_path.resolve(), sidecar.resolve(), paths["manifest"].resolve()}
    selected = []
    for path in sorted(evidence.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() in excluded or path.name in {zip_path.name, sidecar.name, paths["manifest"].name}:
            continue
        if path.suffix.lower() in {".safetensors", ".ckpt", ".pt", ".bin", ".zip"} or path.stat().st_size > 8 * 1024 * 1024:
            continue
        selected.append(path)
    partial = zip_path.with_name(zip_path.name + ".partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, path.relative_to(evidence))
    os.replace(partial, zip_path)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    _write_text(sidecar, digest + f"  {zip_path.name}\n")
    return {"status": "SEALED", "path": str(zip_path), "sha256": digest, "bytes": zip_path.stat().st_size}


def _run_named_stage(
    evidence: Path,
    name: str,
    deadline: float,
    *,
    minimum_s: float,
    timeout_s: float,
    stage_runner: StageRunner | None,
) -> dict[str, Any]:
    if stage_runner is not None:
        body = dict(stage_runner(evidence, name))
        _event(evidence, f"{name}_finish", ok=body.get("ok") is True)
        _write(evidence / "session-partial.json", {"last_stage": name, "ok": body.get("ok") is True})
        return body
    return _stage(evidence, name, deadline, minimum_s=minimum_s, timeout_s=timeout_s)


def _stage(evidence: Path, name: str, deadline: float, *, minimum_s: float, timeout_s: float) -> dict[str, Any]:
    if timeout_s < minimum_s:
        raise RuntimeError(f"{name} timeout {timeout_s} is below the fixed minimum {minimum_s}")
    remaining = _remaining(deadline)
    if remaining < minimum_s:
        payload = {"status": "NOT_RUN_BUDGET", "remaining_s": round(remaining, 3), "need_s": minimum_s}
        _event(evidence, f"{name}_not_run_budget", **payload)
        return payload
    report = evidence / f"{name}.json"
    _event(evidence, f"{name}_start", timeout_s=timeout_s)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "bootstrap_actor_admission.py"), "--stage", name, "--report", str(report)],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = round(time.perf_counter() - started, 3)
        payload = {
            "ok": False,
            "status": "TIMEOUT",
            "elapsed_s": elapsed,
            "timeout_s": timeout_s,
            "stdout": _tail(exc.stdout),
            "stderr": _tail(exc.stderr),
        }
        _event(evidence, f"{name}_timeout", elapsed_s=elapsed, timeout_s=timeout_s)
        return payload
    elapsed = round(time.perf_counter() - started, 3)
    body: dict[str, Any] = {"returncode": completed.returncode, "elapsed_s": elapsed, "timeout_s": timeout_s}
    if report.is_file():
        body.update(json.loads(report.read_text(encoding="utf-8")))
    else:
        body["status"] = "NO_REPORT"
        body["stdout"] = _tail(completed.stdout)
        body["stderr"] = _tail(completed.stderr)
    body["ok"] = completed.returncode == 0 and body.get("ok") is True
    _event(evidence, f"{name}_finish", elapsed_s=elapsed, ok=body["ok"], returncode=completed.returncode)
    _write(evidence / "session-partial.json", {"last_stage": name, "ok": body["ok"], "elapsed_s": elapsed})
    return body


def _run_actor(
    evidence: Path,
    deadline: float,
    *,
    actor_argv: list[str] | None,
    actor_timeout_s: float | None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "H3_T4_EVIDENCE_DIR": str(evidence),
            "H3_T4_ACTOR_READY_TIMEOUT_S": str(READY_TIMEOUT_S),
            "H3_T4_COLLECTIVE_TIMEOUT_S": str(COLLECTIVE_TIMEOUT_S),
            "H3_T4_CLEANUP_TIMEOUT_S": str(CLEANUP_TIMEOUT_S),
            "H3_T4_ALLOW_MODEL_LOAD": "0",
        }
    )
    command = actor_argv or [
        sys.executable,
        str(ROOT / "scripts" / "model_retest.py"),
        "--stage",
        "actor-admission",
        "--evidence-dir",
        str(evidence),
    ]
    timeout_s = ACTOR_BUDGET_S if actor_timeout_s is None else actor_timeout_s
    _event(evidence, "actor_start", ready_s=READY_TIMEOUT_S, collective_s=COLLECTIVE_TIMEOUT_S, cleanup_s=CLEANUP_TIMEOUT_S)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _stop_process_group(process)
        stdout, stderr = process.communicate()
    elapsed = round(time.perf_counter() - started, 3)
    _write_text(evidence / "actor-stdout.txt", _tail(stdout, 4000))
    _write_text(evidence / "actor-stderr.txt", _tail(stderr, 4000))
    admission = _read_admission(evidence / "actor_admission.json")
    body: dict[str, Any] = {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "elapsed_s": elapsed,
        "remaining_s": round(_remaining(deadline), 3),
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
        "admission": admission,
        "actor_admission_passed": False,
    }
    report_path = evidence / "preflight.json"
    if report_path.is_file():
        body["report_present"] = True
    outcome = classify_actor_outcome(returncode=process.returncode, timed_out=timed_out, admission=admission)
    body["actor_admission_passed"] = outcome["actor_admission_passed"]
    body["outcome_status"] = outcome["status"]
    _write(evidence / "actor-result.json", body)
    _event(
        evidence,
        "actor_finish",
        elapsed_s=elapsed,
        returncode=process.returncode,
        timed_out=timed_out,
        actor_admission_passed=body["actor_admission_passed"],
        admission_file_collective_ok=outcome["admission_file_collective_ok"],
    )
    return body


def _read_admission(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"unreadable": True}
    return payload if isinstance(payload, dict) else {"unreadable": True}


def decide_check(remaining_s: float) -> str:
    """Check runs at CHECK_TIMEOUT_S or not at all. Never at a shorter timeout."""

    if remaining_s < CHECK_TIMEOUT_S:
        return "NOT_RUN_BUDGET"
    return "RUN"


def decide_actor(remaining_s: float, *, gpu_permitted: bool) -> str:
    if not gpu_permitted:
        return "NOT_RUN_GPU_GATE"
    if remaining_s < ACTOR_BUDGET_S:
        return "NOT_RUN_BUDGET"
    return "RUN"


def _ok(stage: dict[str, Any]) -> bool:
    return stage.get("ok") is True and stage.get("status") != "NOT_RUN_BUDGET"


def _remaining(deadline: float) -> float:
    return deadline - time.monotonic()


def _persist(evidence: Path, state: dict[str, Any]) -> None:
    _write(evidence / "session-status.json", state)
    _write(evidence / "session.json", state)


def _event(evidence: Path, kind: str, **fields: object) -> None:
    path = evidence / "events.jsonl"
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _tail(value: object, limit: int = 500) -> str:
    if not isinstance(value, str):
        return ""
    return value[-limit:]


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        process.kill()


def _link_path(path: Path) -> str:
    working = Path("/kaggle/working")
    try:
        return str(path.resolve().relative_to(working))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
