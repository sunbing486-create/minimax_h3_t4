from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

SOURCE = Path("/tmp/h3src")
RUNTIME = Path("/tmp/h3rt")
USER = "h3ro"
PYTHON = Path("/tmp/h3-cpu-venv/bin/python")
SCRIPT = textwrap.dedent(
    """\
    import hashlib
    import json
    import os
    import sys
    from pathlib import Path

    source_root = Path(sys.argv[1])
    runtime_dir = Path(sys.argv[2])
    evidence_dir = Path(sys.argv[3])
    os.environ["H3_T4_RUNTIME_DIR"] = str(runtime_dir)
    os.environ["H3_T4_EVIDENCE_DIR"] = str(evidence_dir)
    os.environ["H3_T4_ALLOW_MODEL_LOAD"] = "0"
    sys.path.insert(0, str(source_root / "src"))
    start_cwd = Path.cwd()

    def snapshot(root: Path) -> list[tuple[str, str, int]]:
        rows = []
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rows.append((str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size))
        return rows

    before = snapshot(source_root)
    probe = source_root / "src" / "_ray_runtime_env"
    try:
        probe.mkdir()
    except OSError as exc:
        write_blocked = True
        write_errno = exc.errno
    else:
        write_blocked = False
        write_errno = None

    from minimax_h3_t4.loader_nodes import H3T4Initializer, build_admission_runtime_env, ray_runtime_paths
    import ray

    paths = ray_runtime_paths()
    env = build_admission_runtime_env()
    local_working_dir = str(env["working_dir"])
    local_temp_dir = str(paths["temp_dir"])
    ray.init(
        address="local",
        namespace="minimax-h3-t4-readonly",
        include_dashboard=False,
        num_cpus=1,
        object_store_memory=80 * 1024 * 1024,
        _temp_dir=str(paths["temp_dir"]),
        runtime_env=env,
    )

    @ray.remote
    def ping():
        from minimax_h3_t4.runtime.topology import ExactH3T4Topology
        return ExactH3T4Topology().world_size

    staged = ray.get(ping.remote())
    sessions = [str(path) for path in Path(paths["temp_dir"]).glob("session_*")]
    ray.shutdown()

    requested = {}
    original_init = ray.init

    def capped_init(**kwargs):
        requested["object_store_memory"] = kwargs.get("object_store_memory")
        requested["temp_dir"] = kwargs.get("_temp_dir")
        requested["working_dir"] = (kwargs.get("runtime_env") or {}).get("working_dir")
        kwargs["object_store_memory"] = 80 * 1024 * 1024
        kwargs["num_cpus"] = 1
        return original_init(**kwargs)

    ray.init = capped_init
    initializer_error = None
    try:
        H3T4Initializer(ray_module=ray).initialize(
            load_after=object(),
            manage_parent_memory=False,
            for_admission=True,
            force_on_failure=True,
        )
    except BaseException as exc:
        initializer_error = {"type": type(exc).__name__, "errno": getattr(exc, "errno", None), "message": str(exc)[:500]}
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass
    result = {
        "uid": os.geteuid(),
        "module_file": Path(sys.modules["minimax_h3_t4.loader_nodes"].__file__).resolve().as_posix(),
        "write_blocked": write_blocked,
        "write_errno": write_errno,
        "working_dir": local_working_dir,
        "temp_dir": local_temp_dir,
        "packaged_working_dir": env["working_dir"],
        "py_modules": env["py_modules"],
        "staged_world_size": staged,
        "requested": requested,
        "cwd_unchanged": str(Path.cwd()) == str(start_cwd),
        "source_unchanged": before == snapshot(source_root),
        "initializer_error": initializer_error,
        "actor_admission_passed": False,
        "session_dirs": sessions,
    }
    (evidence_dir / "readonly-result.json").write_text(json.dumps(result, indent=2))
    """
)


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, capture_output=True, **kwargs)


@pytest.fixture(scope="module")
def readonly_result() -> dict[str, object]:
    if not PYTHON.is_file():
        pytest.skip("CPU virtualenv with Ray is not installed")
    sudo = _run(["sudo", "-n", "true"])
    if sudo.returncode != 0:
        pytest.skip("passwordless sudo is required to create the unprivileged user")
    _run(["sudo", "useradd", "--system", "--create-home", "--shell", "/usr/sbin/nologin", USER])
    identity = _run(["id", "-u", USER])
    if identity.returncode != 0:
        pytest.fail(identity.stderr)
    for path in (SOURCE, RUNTIME):
        _run(["sudo", "rm", "-rf", str(path)])
    package = SOURCE / "src" / "minimax_h3_t4"
    shutil.copytree(Path("src/minimax_h3_t4"), package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for directory in (RUNTIME, RUNTIME / "home", RUNTIME / "tmp", RUNTIME / "runtime", RUNTIME / "evidence"):
        directory.mkdir(parents=True)
    (RUNTIME / "check.py").write_text(SCRIPT, encoding="utf-8")
    _run(["sudo", "chown", "-R", "root:root", str(SOURCE)])
    _run(["sudo", "chmod", "-R", "a-w", str(SOURCE)])
    _run(["sudo", "chmod", "-R", "a+rX", str(SOURCE)])
    _run(["sudo", "chown", "-R", f"{USER}:{USER}", str(RUNTIME)])
    env = {
        "HOME": str(RUNTIME / "home"),
        "TMPDIR": str(RUNTIME / "tmp"),
        "PATH": "/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
    }
    try:
        completed = _run(
            ["sudo", "-n", "runuser", "-u", USER, "--", str(PYTHON), str(RUNTIME / "check.py"), str(SOURCE), str(RUNTIME / "runtime"), str(RUNTIME / "evidence")],
            env=env,
            timeout=180,
        )
        result_path = RUNTIME / "evidence" / "readonly-result.json"
        if completed.returncode != 0 or not result_path.is_file():
            pytest.fail(f"readonly staging failed ({completed.returncode})\\n{completed.stdout}\\n{completed.stderr}")
        return json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        _run(["sudo", "pkill", "-u", USER], timeout=10)
        _run(["sudo", "rm", "-rf", str(SOURCE), str(RUNTIME)])


def test_unprivileged_user_cannot_write_the_source_tree(readonly_result: dict[str, object]) -> None:
    assert readonly_result["uid"] != 0
    assert readonly_result["write_blocked"] is True
    assert readonly_result["write_errno"] == 13
    assert readonly_result["source_unchanged"] is True
    assert str(readonly_result["module_file"]).startswith("/tmp/h3src/src/minimax_h3_t4/")


def test_ray_staging_stays_in_the_writable_runtime_dir(readonly_result: dict[str, object]) -> None:
    assert str(readonly_result["working_dir"]).startswith("/tmp/h3rt/runtime/working_dir")
    assert str(readonly_result["temp_dir"]).startswith("/tmp/h3rt/runtime/temp")
    assert str(readonly_result["packaged_working_dir"]).startswith("gcs://_ray_pkg_")
    assert readonly_result["staged_world_size"] == 2
    assert readonly_result["session_dirs"]
    assert all(str(path).startswith("/tmp/h3rt/runtime/temp/") for path in readonly_result["session_dirs"])
    assert readonly_result["cwd_unchanged"] is True
    assert "_ray_runtime_env" not in str(readonly_result["working_dir"])


def test_real_initializer_reaches_placement_without_writing_the_source(readonly_result: dict[str, object]) -> None:
    error = readonly_result["initializer_error"]
    assert isinstance(error, dict)
    assert error["type"] == "PlacementError"
    assert error["errno"] is None
    assert "Read-only" not in str(error["message"])
    requested = readonly_result["requested"]
    assert requested["object_store_memory"] == 4 * 1024**3
    assert str(requested["temp_dir"]).startswith("/tmp/h3rt/runtime/temp")
    assert str(requested["working_dir"]).startswith("/tmp/h3rt/runtime/working_dir")
    assert readonly_result["actor_admission_passed"] is False
