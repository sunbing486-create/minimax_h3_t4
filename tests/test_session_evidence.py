from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path

from scripts.run_actor_session import bundle_paths, classify_actor_outcome, run_session


def _stages(evidence: Path, name: str) -> dict[str, object]:
    if name == "check":
        return {"ok": True, "gpu_stage_permitted": True, "status": "READY"}
    return {"ok": True, "status": "OK"}


def _run(evidence: Path, script: str, *, timeout_s: float | None = None) -> dict[str, object]:
    actor = evidence.parent / "actor.py"
    actor.write_text(textwrap.dedent(script), encoding="utf-8")
    code = run_session(
        evidence,
        budget_s=240,
        stage_runner=_stages,
        actor_argv=[sys.executable, str(actor), str(evidence)],
        actor_timeout_s=timeout_s,
    )
    status = json.loads((evidence / "session-status.json").read_text(encoding="utf-8"))
    result = json.loads((evidence / "actor-result.json").read_text(encoding="utf-8"))
    return {"code": code, "status": status, "result": result}


def _assert_bundle(evidence: Path, status: dict[str, object]) -> None:
    paths = bundle_paths(evidence)
    assert paths["zip"].is_file()
    assert paths["sidecar"].is_file()
    digest = hashlib.sha256(paths["zip"].read_bytes()).hexdigest()
    assert digest in paths["sidecar"].read_text(encoding="utf-8")
    assert status["bundle"]["sha256"] == digest
    assert status["bundle_status"] == "SEALED"
    names = zipfile.ZipFile(paths["zip"]).namelist()
    assert paths["zip"].name not in names
    assert paths["sidecar"].name not in names
    assert "session-status.json" in names
    assert "actor-result.json" in names


def test_claim_with_nonzero_exit_is_not_a_pass(tmp_path: Path) -> None:
    evidence = tmp_path / "h3-actor-session"
    outcome = _run(
        evidence,
        """
        import json, sys
        from pathlib import Path
        evidence = Path(sys.argv[1])
        payload = {
            "kind": "ray-h3t4-actor-admission",
            "collective_ok": True,
            "distinct_gpu_mapping": True,
            "weights_loaded": False,
            "used_mock": False,
            "cleanup": {"ok": False, "status": "FAILED", "message": "ray shutdown exceeded"},
        }
        (evidence / "actor_admission.json").write_text(json.dumps(payload), encoding="utf-8")
        raise SystemExit(2)
        """,
    )
    status = outcome["status"]
    assert outcome["code"] == 2
    assert status["status"] == "ACTOR_FAILED_AFTER_CLAIM"
    assert status["actor_admission_passed"] is False
    assert status["admission_file_collective_ok"] is True
    assert status["process_returncode"] == 2
    assert status["cleanup_ok"] is False
    _assert_bundle(evidence, status)


def test_actor_timeout_seals_a_failed_status(tmp_path: Path) -> None:
    evidence = tmp_path / "h3-actor-session"
    outcome = _run(evidence, "import time\ntime.sleep(30)\n", timeout_s=1)
    status = outcome["status"]
    assert status["status"] == "ACTOR_TIMEOUT"
    assert status["actor_admission_passed"] is False
    assert status["process_timed_out"] is True
    assert outcome["result"]["admission"] is None
    _assert_bundle(evidence, status)


def test_exit_zero_without_admission_file_is_unproven(tmp_path: Path) -> None:
    evidence = tmp_path / "h3-actor-session"
    outcome = _run(evidence, "raise SystemExit(0)\n")
    status = outcome["status"]
    assert status["status"] == "ACTOR_EXIT_ZERO_UNPROVEN"
    assert status["actor_admission_passed"] is False
    assert status["admission_file_present"] is False
    _assert_bundle(evidence, status)


def test_nonzero_without_admission_file_is_actor_failed(tmp_path: Path) -> None:
    evidence = tmp_path / "h3-actor-session"
    outcome = _run(evidence, "import sys\nsys.stderr.write('boom')\nraise SystemExit(3)\n")
    status = outcome["status"]
    assert status["status"] == "ACTOR_FAILED"
    assert status["actor_admission_passed"] is False
    assert status["process_returncode"] == 3
    assert "boom" in (evidence / "actor-stderr.txt").read_text(encoding="utf-8")
    _assert_bundle(evidence, status)


def test_bundle_export_failure_does_not_erase_the_actor_status(tmp_path: Path) -> None:
    evidence = tmp_path / "h3-actor-session"
    bundle_paths(evidence)["zip"].mkdir()
    outcome = _run(evidence, "raise SystemExit(3)\n")
    status = outcome["status"]
    assert status["status"] == "ACTOR_FAILED"
    assert status["actor_admission_passed"] is False
    assert status["process_returncode"] == 3
    assert status["bundle_status"] == "FAILED"
    assert "bundle_error" in status
    assert (evidence / "actor-result.json").is_file()


def test_kill_during_actor_leaves_a_parseable_running_status(tmp_path: Path) -> None:
    evidence = tmp_path / "h3-actor-session"
    pidfile = tmp_path / "actor.pid"
    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent(
            f"""
            import sys
            from pathlib import Path
            sys.path.insert(0, {str(Path.cwd())!r})
            from scripts.run_actor_session import run_session

            def stages(evidence, name):
                if name == "check":
                    return {{"ok": True, "gpu_stage_permitted": True}}
                return {{"ok": True}}

            run_session(
                Path(sys.argv[1]),
                budget_s=240,
                stage_runner=stages,
                actor_argv=[sys.executable, "-c", "import os, time, pathlib, sys; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)", sys.argv[2]],
                actor_timeout_s=30,
            )
            """
        ),
        encoding="utf-8",
    )
    process = subprocess.Popen([sys.executable, str(driver), str(evidence), str(pidfile)], cwd=Path.cwd())
    status_path = evidence / "session-status.json"
    seen = None
    deadline = time.time() + 20
    while time.time() < deadline:
        if status_path.is_file():
            seen = json.loads(status_path.read_text(encoding="utf-8"))
            if seen.get("status") == "ACTOR_RUNNING":
                break
        time.sleep(0.05)
    os.kill(process.pid, signal.SIGKILL)
    process.wait(timeout=5)
    if pidfile.is_file():
        os.kill(int(pidfile.read_text(encoding="utf-8")), signal.SIGKILL)
    assert seen is not None
    assert seen["status"] == "ACTOR_RUNNING"
    assert seen["actor_admission_passed"] is False
    reread = json.loads(status_path.read_text(encoding="utf-8"))
    assert reread["actor_admission_passed"] is False
    assert reread["status"] == "ACTOR_RUNNING"


def test_classifier_keeps_the_file_claim_beside_a_failed_process() -> None:
    outcome = classify_actor_outcome(
        returncode=2,
        timed_out=False,
        admission={
            "kind": "ray-h3t4-actor-admission",
            "collective_ok": True,
            "distinct_gpu_mapping": True,
            "weights_loaded": False,
            "used_mock": False,
            "cleanup": {"ok": False},
        },
    )
    assert outcome["status"] == "ACTOR_FAILED_AFTER_CLAIM"
    assert outcome["actor_admission_passed"] is False
    assert outcome["admission_file_collective_ok"] is True
    assert outcome["process_returncode"] == 2
