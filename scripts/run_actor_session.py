#!/usr/bin/env python3
"""One entry point for plan, apply, check, then actor admission.

Check timeout is fixed at 75 seconds and is never reduced. Actor admission
uses ready 45s, collective 60s, and cleanup 20s, and starts only when the
check saw two CUDA devices and at least 125 seconds remain. This process
does not download weights and does not claim a GPU result on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

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
if CHECK_TIMEOUT_S < 60:
    raise RuntimeError("check timeout must stay at least 60 seconds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded actor-admission session")
    parser.add_argument("--evidence-dir", default=os.environ.get("H3_T4_EVIDENCE_DIR", "h3-actor-session"))
    parser.add_argument("--budget-s", type=float, default=DEFAULT_BUDGET_S)
    args = parser.parse_args(argv)
    evidence = Path(args.evidence_dir)
    code = run_session(evidence, budget_s=args.budget_s)
    session = json.loads((evidence / "session.json").read_text(encoding="utf-8")) if (evidence / "session.json").is_file() else {}
    print(json.dumps({"evidence": str(evidence), "exit": code, "status": session.get("status"), "actor_admission_passed": session.get("actor_admission_passed") is True}, sort_keys=True))
    return code


def run_session(evidence: Path, *, budget_s: float = DEFAULT_BUDGET_S) -> int:
    evidence.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = time.monotonic() + budget_s
    state: dict[str, object] = {
        "stage": "session",
        "actor_admission_passed": False,
        "gpu_stage_started": False,
        "weights_downloaded": False,
        "check_timeout_s": CHECK_TIMEOUT_S,
        "ready_timeout_s": READY_TIMEOUT_S,
        "collective_timeout_s": COLLECTIVE_TIMEOUT_S,
        "cleanup_timeout_s": CLEANUP_TIMEOUT_S,
        "budget_s": budget_s,
    }
    exit_code = 2
    try:
        _event(evidence, "session_start", budget_s=budget_s, check_timeout_s=CHECK_TIMEOUT_S)
        state["plan"] = _stage(evidence, "plan", deadline, minimum_s=5, timeout_s=40)
        if not _ok(state["plan"]):
            state["status"] = "PLAN_FAILED"
            exit_code = 2
            return exit_code
        state["apply"] = _stage(evidence, "apply", deadline, minimum_s=5, timeout_s=90)
        if not _ok(state["apply"]):
            state["status"] = "APPLY_FAILED"
            exit_code = 2
            return exit_code
        if _remaining(deadline) < CHECK_TIMEOUT_S:
            state["check"] = {"status": "NOT_RUN_BUDGET", "check_timeout_s": CHECK_TIMEOUT_S, "remaining_s": round(_remaining(deadline), 3)}
            state["actor"] = {"status": "NOT_RUN_BUDGET"}
            state["status"] = "NOT_RUN_BUDGET"
            _event(evidence, "not_run_budget", remaining_s=state["check"]["remaining_s"], need_s=CHECK_TIMEOUT_S)
            exit_code = 2
            return exit_code
        state["check"] = _stage(evidence, "check", deadline, minimum_s=CHECK_TIMEOUT_S, timeout_s=CHECK_TIMEOUT_S)
        if not _ok(state["check"]):
            state["status"] = "CHECK_FAILED"
            state["actor"] = {"status": "NOT_RUN_CHECK"}
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
        state["actor"] = _run_actor(evidence, deadline)
        state["actor_admission_passed"] = bool(state["actor"].get("actor_admission_passed"))
        state["status"] = "ACTOR_FINISHED" if state["actor"].get("returncode") == 0 else "ACTOR_FAILED"
        exit_code = 0 if state["actor_admission_passed"] else 2
        return exit_code
    except Exception as exc:
        state["status"] = "FAILED"
        state["error_type"] = type(exc).__name__
        state["message"] = str(exc)
        _event(evidence, "session_error", error_type=type(exc).__name__, message=str(exc)[:500])
        exit_code = 2
        return exit_code
    finally:
        state["elapsed_s"] = round(time.time() - started, 3)
        state["actor_admission_passed"] = bool(state.get("actor_admission_passed"))
        _write(evidence / "session.json", state)
        _event(evidence, "session_finish", status=state.get("status"), elapsed_s=state["elapsed_s"])
        bundle = _zip_evidence(evidence)
        state["bundle"] = bundle
        _write(evidence / "session.json", state)


def _stage(evidence: Path, name: str, deadline: float, *, minimum_s: float, timeout_s: float) -> dict[str, object]:
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
        payload = {"ok": False, "status": "TIMEOUT", "elapsed_s": elapsed, "timeout_s": timeout_s, "stdout": (exc.stdout or "")[-500:] if isinstance(exc.stdout, str) else "", "stderr": (exc.stderr or "")[-500:] if isinstance(exc.stderr, str) else ""}
        _event(evidence, f"{name}_timeout", elapsed_s=elapsed, timeout_s=timeout_s)
        return payload
    elapsed = round(time.perf_counter() - started, 3)
    body: dict[str, object] = {"returncode": completed.returncode, "elapsed_s": elapsed, "timeout_s": timeout_s}
    if report.is_file():
        body.update(json.loads(report.read_text(encoding="utf-8")))
    else:
        body["status"] = "NO_REPORT"
        body["stdout"] = (completed.stdout or "")[-500:]
        body["stderr"] = (completed.stderr or "")[-500:]
    body["ok"] = completed.returncode == 0 and body.get("ok") is True
    _event(evidence, f"{name}_finish", elapsed_s=elapsed, ok=body["ok"], returncode=completed.returncode)
    _write(evidence / "session-partial.json", {"last_stage": name, "ok": body["ok"], "elapsed_s": elapsed})
    return body


def _run_actor(evidence: Path, deadline: float) -> dict[str, object]:
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
    _event(evidence, "actor_start", ready_s=READY_TIMEOUT_S, collective_s=COLLECTIVE_TIMEOUT_S, cleanup_s=CLEANUP_TIMEOUT_S)
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "model_retest.py"), "--stage", "actor-admission", "--evidence-dir", str(evidence)],
        check=False,
        text=True,
        capture_output=True,
        timeout=ACTOR_BUDGET_S,
        cwd=str(ROOT),
        env=env,
    )
    elapsed = round(time.perf_counter() - started, 3)
    report_path = evidence / "preflight.json"
    body: dict[str, object] = {"returncode": completed.returncode, "elapsed_s": elapsed, "remaining_s": round(_remaining(deadline), 3)}
    if report_path.is_file():
        body["report"] = json.loads(report_path.read_text(encoding="utf-8"))
    admission = evidence / "actor_admission.json"
    if admission.is_file():
        payload = json.loads(admission.read_text(encoding="utf-8"))
        body["actor_admission_passed"] = payload.get("collective_ok") is True and payload.get("distinct_gpu_mapping") is True
    else:
        body["actor_admission_passed"] = False
    _event(evidence, "actor_finish", elapsed_s=elapsed, actor_admission_passed=body["actor_admission_passed"])
    return body


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


def _ok(stage: dict[str, object]) -> bool:
    return stage.get("ok") is True and stage.get("status") != "NOT_RUN_BUDGET"


def _remaining(deadline: float) -> float:
    return deadline - time.monotonic()


def _event(evidence: Path, kind: str, **fields: object) -> None:
    path = evidence / "events.jsonl"
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _zip_evidence(evidence: Path) -> dict[str, object]:
    zip_path = evidence / "session-evidence.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(evidence.rglob("*")):
            if path.is_file() and path.name != zip_path.name and not path.name.endswith(".sha256"):
                archive.write(path, path.relative_to(evidence))
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    (evidence / "session-evidence.zip.sha256").write_text(digest + "  session-evidence.zip\n", encoding="utf-8")
    return {"zip": str(zip_path), "sha256": digest, "bytes": zip_path.stat().st_size}


if __name__ == "__main__":
    raise SystemExit(main())
