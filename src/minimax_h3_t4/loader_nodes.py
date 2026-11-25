from __future__ import annotations

import gc
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .runtime.evidence import EvidenceStore, note_failure
from .runtime.lifecycle import H3T4ActorGroup, remote
from .runtime.placement import PlacementError, model_stage_allowed, plan_from_scheduler_resources, reservation_cap_bytes
from .runtime.startup import StartupTimeout, choose_master_port, env_seconds, resolve_one, run_stage
from .runtime.topology import ExactH3T4Topology


def _ray_module() -> Any:
    import ray

    return ray


def _model_management() -> Any:
    import comfy.model_management

    return comfy.model_management


def build_runtime_env_vars(*, pythonpath: str, comfy_root: str, evidence_dir: str | None = None) -> dict[str, str]:
    """Job-level env for Ray actors.

    ``CUDA_VISIBLE_DEVICES`` is intentionally absent. Ray sets it per actor from
    ``num_gpus=1``; pinning ``0,1`` here overrides that isolation and makes both
    ranks bind ``cuda:0``.
    """

    env = {
        "PYTHONPATH": pythonpath,
        "COMFYUI_BASE_DIRECTORY": comfy_root,
        "MASTER_ADDR": "127.0.0.1",
    }
    if evidence_dir:
        env["H3_T4_EVIDENCE_DIR"] = evidence_dir
    return env


def source_tree_root() -> Path:
    """Uploaded package root. Kaggle mounts this tree read-only."""

    return Path(__file__).resolve().parents[2]


def ray_runtime_paths() -> dict[str, Path]:
    """Writable Ray working directory and session temp.

    ``working_dir`` has to exist before Ray packages it. The session temp is
    where Ray writes the zipped ``py_modules`` and ``working_dir``, plus logs.
    Both used to be created beside this file, under ``src/_ray_runtime_env``.
    """

    explicit = os.environ.get("H3_T4_RUNTIME_DIR", "").strip()
    root = Path(explicit) if explicit else Path(tempfile.gettempdir()) / "h3-ray-runtime"
    root = root.resolve()
    source_root = source_tree_root().resolve()
    try:
        root.relative_to(source_root)
    except ValueError:
        inside_source = False
    else:
        inside_source = True
    if inside_source:
        raise RuntimeError(f"Ray runtime directory must not be inside the source tree: {root}")
    temp_dir = root / "temp"
    working_dir = root / "working_dir"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        working_dir.mkdir(parents=True, exist_ok=True)
        for directory in (temp_dir, working_dir):
            probe = directory / ".write-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
    except OSError as exc:
        raise RuntimeError(f"Ray runtime directory is not writable: {root}") from exc
    return {"root": root, "temp_dir": temp_dir, "working_dir": working_dir}


def build_admission_runtime_env() -> dict[str, object]:
    """Ray env for weight-free actor admission.

    ``H3T4Worker.join_process_group`` imports torch and
    ``xfuser.core.distributed``. It does not import ``comfy`` or
    ``folder_paths``. Those are only needed later, when ``load_unet`` runs.
    """

    package_dir = Path(__file__).resolve().parent
    runtime_workdir = ray_runtime_paths()["working_dir"]
    python_entries = []
    existing = os.environ.get("PYTHONPATH")
    if existing:
        python_entries.extend(part for part in existing.split(os.pathsep) if part)
    env = {
        "PYTHONPATH": os.pathsep.join(dict.fromkeys(python_entries)),
        "MASTER_ADDR": "127.0.0.1",
    }
    evidence_dir = os.environ.get("H3_T4_EVIDENCE_DIR")
    if evidence_dir:
        env["H3_T4_EVIDENCE_DIR"] = evidence_dir
    return {
        "py_modules": [str(package_dir)],
        "working_dir": str(runtime_workdir),
        "env_vars": env,
    }


def build_unet_runtime_env(comfy_root: Path) -> dict[str, object]:
    """Runtime env used by H3T4Worker.load_unet.

    The Comfy checkout is not installed as a site package. A driver
    ``sys.path`` insert does not reach a Ray worker. The worker only sees
    this ``PYTHONPATH``.
    """

    root = Path(comfy_root).resolve()
    if not (root / "comfy" / "model_detection.py").is_file() or not (root / "folder_paths.py").is_file():
        raise RuntimeError(f"ComfyUI checkout is incomplete: {root}")
    package_dir = Path(__file__).resolve().parent
    runtime_workdir = ray_runtime_paths()["working_dir"]
    python_entries = [str(root)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        python_entries.extend(part for part in existing.split(os.pathsep) if part)
    return {
        "py_modules": [str(package_dir)],
        "working_dir": str(runtime_workdir),
        "env_vars": build_runtime_env_vars(
            pythonpath=os.pathsep.join(dict.fromkeys(python_entries)),
            comfy_root=str(root),
            evidence_dir=os.environ.get("H3_T4_EVIDENCE_DIR"),
        ),
    }


def _build_runtime_env() -> dict[str, object]:
    import folder_paths

    comfy_file = Path(folder_paths.__file__).resolve()
    comfy_root = next(
        (parent for parent in comfy_file.parents if (parent / "main.py").exists() and (parent / "execution.py").exists()),
        None,
    )
    if comfy_root is None:
        raise RuntimeError("Unable to locate the ComfyUI repository root for Ray workers")
    return build_unet_runtime_env(comfy_root)


LOAD_WORKER_IMPORTS = (
    "comfy",
    "folder_paths",
    "comfy.model_detection",
    "comfy.model_management",
    "comfy.model_patcher",
    "comfy.model_base",
    "comfy.utils",
    "xfuser.core.distributed",
    "yunchang",
    "torch",
    "ray",
)


def probe_worker_imports(
    *,
    runtime_env: dict[str, object],
    checkpoint_dir: Path,
    checkpoint_file: str | None = None,
    module_names: tuple[str, ...] = LOAD_WORKER_IMPORTS,
) -> dict[str, Any]:
    """Import the load stack inside a Ray worker, not in the driver.

    The worker process gets ``runtime_env`` and Ray's working directory as its
    cwd. The driver's ``sys.path`` is not copied.
    """

    import ray

    from .runtime.worker import worker_import_snapshot

    ray.shutdown()
    paths = ray_runtime_paths()
    try:
        ray.init(
            address="local",
            namespace="h3-unet-import-probe",
            include_dashboard=False,
            num_cpus=1,
            object_store_memory=80 * 1024 * 1024,
            _temp_dir=str(paths["temp_dir"]),
            runtime_env=runtime_env,
        )
        remote = ray.remote(worker_import_snapshot)
        snapshot = ray.get(
            remote.remote(list(module_names), str(checkpoint_dir), checkpoint_file),
            timeout=90,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "message": str(exc)[:500],
            "imports": [],
            "cuda_count": None,
            "checkpoint_dir_writable": False,
            "download": False,
        }
    finally:
        ray.shutdown()
    if not isinstance(snapshot, dict):
        return {"ok": False, "error_type": "bad_probe", "message": "worker probe returned no report", "imports": [], "download": False}
    snapshot["download"] = False
    snapshot["runtime_pythonpath"] = (runtime_env.get("env_vars") or {}).get("PYTHONPATH")
    return snapshot


def _resolve_unet(name: str) -> str:
    import folder_paths

    path = folder_paths.get_full_path("diffusion_models", name)
    if path is None:
        path = folder_paths.get_full_path_or_raise("checkpoints", name)
    return path


def current_evidence() -> EvidenceStore | None:
    from .runtime.evidence import current

    return current()


def _require_smoke_before_weights() -> None:
    if os.environ.get("H3_T4_REQUIRE_ACTOR_ADMISSION", "0") == "1":
        from .runtime.actor_admission import require_actor_admission_file

        require_actor_admission_file(os.environ.get("H3_T4_ACTOR_ADMISSION_FILE", ""))
    if os.environ.get("H3_T4_REQUIRE_SMOKE", "0") != "1":
        return
    raw = os.environ.get("H3_T4_SMOKE_FILE", "")
    path = Path(raw) if raw else None
    if path is None or not path.is_file():
        raise RuntimeError("model stage blocked: H3_T4_REQUIRE_SMOKE=1 but no passing smoke_result.json was provided")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("model stage blocked: smoke result is not valid JSON") from exc
    if not isinstance(payload, dict) or not model_stage_allowed(payload):
        raise RuntimeError(
            "model stage blocked until a weight-free cuda-nccl two-GPU collective passes; "
            f"mode={payload.get('mode') if isinstance(payload, dict) else None!r} "
            f"collective_ok={payload.get('collective_ok') if isinstance(payload, dict) else None!r}"
        )


def _release_conditioning(model_management: Any) -> None:
    model_management.unload_all_models()
    model_management.soft_empty_cache()
    gc.collect()


class H3T4Initializer:
    """Create the fixed local two-worker MiniMax-H3 execution group."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        return {
            "required": {
                "ray_cluster_namespace": ("STRING", {"default": "minimax-h3-t4"}),
                "GPU_SELECT": ("STRING", {"default": "0,1"}),
                "ray_object_store_gb": (
                    "FLOAT",
                    {"default": 4.0, "min": 4.0, "max": 16.0, "step": 0.5},
                ),
                "load_after": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("H3T4_ACTOR_GROUP",)
    RETURN_NAMES = ("actor_group",)
    FUNCTION = "initialize"
    CATEGORY = "MiniMax H3 T4"

    def __init__(
        self,
        *,
        ray_module: Any | None = None,
        model_management: Any | None = None,
        worker_factory: Callable[[int, dict[str, object]], Any] | None = None,
        runtime_env_builder: Callable[[], dict[str, object]] | None = None,
    ) -> None:
        self._ray = ray_module
        self._model_management = model_management
        self._worker_factory = worker_factory
        self._runtime_env_builder = runtime_env_builder or _build_runtime_env
        self.last_placement: dict[str, Any] | None = None
        self.last_ready: list[Any] = []
        self.last_join: list[Any] = []
        self.evidence: EvidenceStore | None = None

    def _create_worker(
        self,
        rank: int,
        config: dict[str, object],
        ray_module: Any,
        *,
        memory_bytes: int,
        master_port: int,
    ) -> Any:
        if self._worker_factory is not None:
            return self._worker_factory(rank, config)
        from .runtime.worker import H3T4Worker

        # Pass ray.get into worker config so it can resolve shared state dict refs.
        # Master address stays in the config rather than an actor runtime_env:
        # an actor env can replace the job env and would also be the wrong place
        # to set CUDA_VISIBLE_DEVICES.
        config = {
            **config,
            "_ray_get": ray_module.get,
            "_master_addr": "127.0.0.1",
            "_master_port": int(master_port),
        }

        # ``memory`` is a Ray admission reservation, not an RSS cap. The value
        # is the fair share of logical memory, clamped by H3_T4_MAX_RAM_GB
        # (default 14 GiB). Requesting 14 GiB twice does not fit a 32 GiB host
        # after the object store, and the second actor then stays pending.
        actor_class = self._ray.remote(num_cpus=1, num_gpus=1, memory=memory_bytes)(H3T4Worker)
        return actor_class.options(name=f"H3T4Worker:{rank}", memory=memory_bytes).remote(rank, config)

    def initialize(
        self,
        ray_cluster_namespace: str = "minimax-h3-t4",
        GPU_SELECT: str = "0,1",
        ray_object_store_gb: float = 4.0,
        load_after: Any | None = None,
        *,
        manage_parent_memory: bool = True,
        interrupt_checker: Any | None = None,
        force_on_failure: bool = False,
        for_admission: bool = False,
    ) -> tuple[H3T4ActorGroup]:
        if not 4.0 <= float(ray_object_store_gb) <= 16.0:
            raise ValueError("ray_object_store_gb must be between 4.0 and 16.0 GiB")
        selected = [part.strip() for part in GPU_SELECT.split(",") if part.strip()]
        if selected != ["0", "1"]:
            raise ValueError("The exact MiniMax-H3 T4 runtime requires GPU_SELECT='0,1'")
        ray_module = self._ray or _ray_module()
        self._ray = ray_module
        if load_after is None:
            raise ValueError("MiniMax-H3 initialization requires conditioning connected to load_after")
        load_after = None
        if manage_parent_memory:
            _release_conditioning(self._model_management or _model_management())
        evidence = EvidenceStore.open_default()
        self.evidence = evidence
        os.environ["H3_T4_EVIDENCE_DIR"] = str(evidence.run_dir)
        ray_module.shutdown()
        topology = ExactH3T4Topology()
        config = topology.as_worker_config()
        workers: list[Any] = []
        object_store_bytes = int(float(ray_object_store_gb) * 1024**3)
        try:
            paths = ray_runtime_paths()
            evidence.event("ray_runtime_dir", temp_dir=str(paths["temp_dir"]), working_dir=str(paths["working_dir"]))
            runtime_env = (build_admission_runtime_env if for_admission and self._runtime_env_builder is _build_runtime_env else self._runtime_env_builder)()
            ray_module.init(
                address="local",
                namespace=ray_cluster_namespace,
                object_store_memory=object_store_bytes,
                include_dashboard=False,
                _temp_dir=str(paths["temp_dir"]),
                runtime_env=runtime_env,
            )
            memory_bytes = self._admit_workers(ray_module, topology.world_size, object_store_bytes, evidence)
            master_port = choose_master_port()
            evidence.event("master_addr", host="127.0.0.1", port=master_port)
            for rank in range(topology.world_size):
                if interrupt_checker is not None:
                    interrupt_checker()
                workers.append(
                    self._create_worker(
                        rank,
                        config,
                        ray_module,
                        memory_bytes=memory_bytes,
                        master_port=master_port,
                    )
                )
            if interrupt_checker is not None:
                interrupt_checker()
            ready_timeout = env_seconds("H3_T4_ACTOR_READY_TIMEOUT_S", 120)
            join_timeout = env_seconds("H3_T4_COLLECTIVE_TIMEOUT_S", 180)
            self.last_ready = run_stage(workers, ray_module, "ready", ready_timeout, evidence, interrupt_checker=interrupt_checker)
            evidence.snapshot_processes()
            self.last_join = run_stage(
                workers,
                ray_module,
                "join_process_group",
                join_timeout,
                evidence,
                interrupt_checker=interrupt_checker,
            )
            evidence.event("startup_ready", ranks=len(workers))
            evidence.quick_save()
        except BaseException as exc:
            note_failure(exc)
            try:
                evidence.quick_save()
            except Exception:
                pass
            force = force_on_failure or isinstance(exc, (StartupTimeout, PlacementError))
            try:
                self._actor_group(workers, ray_module, topology).close(timeout_seconds=env_seconds("H3_T4_CLEANUP_TIMEOUT_S", 5), force=force)
            except BaseException as close_exc:
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(f"Secondary worker teardown failure: {type(close_exc).__name__}: {close_exc}")
            raise
        return (self._actor_group(workers, ray_module, topology),)

    def _actor_group(self, workers: list[Any], ray_module: Any, topology: ExactH3T4Topology) -> H3T4ActorGroup:
        group = H3T4ActorGroup(workers, ray_module, topology)
        group.note_ready(self.last_ready)
        return group

    def _admit_workers(self, ray_module: Any, world_size: int, object_store_bytes: int, evidence: EvidenceStore) -> int:
        reader = getattr(ray_module, "available_resources", None)
        cap = reservation_cap_bytes()
        if reader is None:
            evidence.event("placement_unmeasured", reservation_cap_bytes=cap)
            self.last_placement = {"fits": True, "per_actor_memory_bytes": cap, "reason": "scheduler resources unavailable"}
            return cap
        plan = plan_from_scheduler_resources(
            reader(),
            world_size=world_size,
            object_store_bytes=object_store_bytes,
            reservation_cap_bytes=cap,
        )
        self.last_placement = plan
        evidence.event("placement", **plan)
        if not plan["fits"]:
            raise PlacementError(
                "refusing to start workers: "
                f"{plan['reason']}; logical_memory={plan['logical_memory_bytes']} "
                f"cap={plan['reservation_cap_bytes']} gpus={plan['gpu_count']}",
                plan,
            )
        return int(plan["per_actor_memory_bytes"])


class H3T4UNETLoader:
    """Load an INT8 MiniMax-H3 checkpoint on both FSDP workers via shared state dict."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        try:
            import folder_paths

            names = folder_paths.get_filename_list("diffusion_models") + folder_paths.get_filename_list("checkpoints")
        except ImportError:
            names = []
        return {
            "required": {
                "actor_group": ("H3T4_ACTOR_GROUP",),
                "unet_name": (names,),
                "load_after": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("H3T4_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "MiniMax H3 T4"

    def __init__(
        self,
        *,
        path_resolver: Callable[[str], str] | None = None,
        model_management: Any | None = None,
    ) -> None:
        self._path_resolver = path_resolver or _resolve_unet
        self._model_management = model_management

    def load(
        self,
        actor_group: H3T4ActorGroup,
        unet_name: str,
        load_after: Any | None,
        *,
        manage_parent_memory: bool = True,
        interrupt_checker: Any | None = None,
        force_on_failure: bool = False,
        timeout_seconds: float | None = None,
    ) -> tuple[H3T4ActorGroup]:
        actor_group.require_alive()
        if load_after is None:
            raise ValueError("MiniMax-H3 INT8 loading requires conditioning connected to load_after")
        load_after = None
        if manage_parent_memory:
            _release_conditioning(self._model_management or _model_management())
        ray_module = actor_group.ray_module
        evidence = current_evidence()
        timeout = env_seconds("H3_T4_MODEL_LOAD_TIMEOUT_S", 900) if timeout_seconds is None else timeout_seconds
        try:
            _require_smoke_before_weights()
            path = self._path_resolver(unet_name)
            # Broadcast checkpoint path to workers one rank at a time.
            # Each worker mmaps the same file independently — no main-process RAM peak,
            # no Ray Object Store copy. Peak RAM is ~20GB per worker (mmap-backed).
            for index, worker in enumerate(actor_group.workers):
                if evidence is not None:
                    evidence.event("model_load_start", rank=index, progress=f"{index}/{len(actor_group.workers)}")
                ref = remote(worker, "load_unet", path, "int8")
                resolve_one(
                    ray_module,
                    ref,
                    timeout,
                    label=f"load_unet rank {index}",
                    interrupt_checker=interrupt_checker,
                )
                if evidence is not None:
                    evidence.event("model_load_done", rank=index, progress=f"{index + 1}/{len(actor_group.workers)}")
                    evidence.quick_save()
        except BaseException as exc:
            note_failure(exc)
            if evidence is not None:
                try:
                    evidence.quick_save()
                except Exception:
                    pass
            try:
                actor_group.close(
                    timeout_seconds=env_seconds("H3_T4_CLEANUP_TIMEOUT_S", 5),
                    force=force_on_failure or isinstance(exc, StartupTimeout),
                )
            except BaseException as close_exc:
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(f"Secondary worker teardown failure: {type(close_exc).__name__}: {close_exc}")
            raise

        actor_group.checkpoint = path
        return (actor_group,)
