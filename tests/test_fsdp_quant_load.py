from __future__ import annotations

import json
import os
import traceback
from multiprocessing import get_context
from pathlib import Path

import pytest

COMFY = Path("/tmp/ComfyUI-h3")
SRC = Path("/workspace/src")


def _rank_worker(rank: int, world: int, path: str, port: int, missing_scale: bool, out: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    import sys

    sys.path[:0] = [str(COMFY), str(SRC)]
    import comfy.options

    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0], "--cpu"]
    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from comfy.ops import disable_weight_init, mixed_precision_ops
    from minimax_h3_t4.runtime.checkpoint_budget import LazyCheckpoint
    from minimax_h3_t4.runtime.fsdp import load_from_full_model_state_dict, register_absent_weight_parameters
    from minimax_h3_t4.runtime.unet_load import bounded_weight_check, load_integrity, summarize_module_storage

    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        if rank == 0:
            weight = torch.arange(1024, dtype=torch.int16).remainder(127).to(torch.int8).reshape(4, 256)
            plain = torch.arange(1024, dtype=torch.bfloat16).reshape(4, 256)
            conf = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode()
            payload = {
                "quant.weight": weight,
                "quant.comfy_quant": torch.tensor(list(conf), dtype=torch.uint8),
                "plain.weight": plain,
            }
            if not missing_scale:
                payload["quant.weight_scale"] = torch.tensor(0.25)
            save_file(payload, path)
        dist.barrier()
        host = torch.nn.Module()
        host.quant = mixed_precision_ops({}, torch.bfloat16).Linear(256, 4, bias=False, device=torch.device("cpu"))
        host.plain = disable_weight_init.Linear(256, 4, bias=False, device=torch.device("meta"))
        register_absent_weight_parameters(host)
        host.to("meta")
        fully_shard(host.quant, mesh=mesh)
        fully_shard(host.plain, mesh=mesh)
        fully_shard(host, mesh=mesh)
        before = host.quant.weight
        assert before.device.type == "meta"

        def getter(key: str) -> torch.Tensor:
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as handle:
                return handle.get_tensor(key)

        keys = ["quant.weight", "quant.comfy_quant", "plain.weight"]
        if not missing_scale:
            keys.append("quant.weight_scale")
        try:
            report = load_from_full_model_state_dict(host, LazyCheckpoint(getter, keys), torch.device("cpu"), strict=True, cpu_offload=True)
        except ValueError as exc:
            Path(out).write_text(json.dumps({"rank": rank, "error": type(exc).__name__, "message": str(exc)}), encoding="utf-8")
            return
        quant = host.quant.weight
        quant_local = getattr(quant, "_local_tensor", quant)
        qdata = getattr(quant_local, "_qdata", None)
        plain_local = getattr(host.plain.weight, "_local_tensor", host.plain.weight)
        activation = torch.ones(1, 256, dtype=torch.bfloat16)
        plain_mro_before = [item.__name__ for item in type(host.plain.weight).mro()[:4]]
        quant_mro_before = [item.__name__ for item in type(quant).mro()[:4]]
        reference = torch.arange(1024, dtype=torch.int16).remainder(127).to(torch.int8).reshape(4, 256)
        plain_reference = torch.arange(1024, dtype=torch.bfloat16).reshape(4, 256)
        expect_q = torch.tensor_split(reference, world, dim=0)[rank]
        expect_p = torch.tensor_split(plain_reference, world, dim=0)[rank]
        outer = torch.nn.Module()
        outer.diffusion_model = host
        check = bounded_weight_check(outer, path, rank=rank, world_size=world)
        wrong = bounded_weight_check(outer, path, rank=1 - rank, world_size=world)
        storage = summarize_module_storage(outer)
        plain_forward = host.plain(activation).detach().float().flatten()[:4].tolist()
        quant_forward = None
        quant_forward_error = None
        try:
            quant_forward = host.quant(activation).detach().float().flatten()[:4].tolist()
        except Exception as exc:
            quant_forward_error = {"error_type": type(exc).__name__, "message": str(exc)[:400]}
        Path(out).write_text(
            json.dumps(
                {
                    "rank": rank,
                    "still_meta": report["still_meta"],
                    "resident": report["resident"],
                    "quant_device": str(qdata.device),
                    "quant_dtype": str(qdata.dtype),
                    "quant_match": bool(torch.equal(qdata.view(torch.int8).cpu(), expect_q)),
                    "plain_device": str(plain_local.device),
                    "plain_dtype": str(plain_local.dtype),
                    "plain_match": bool(torch.equal(plain_local.detach().cpu(), expect_p)),
                    "quant_type": type(quant).__name__,
                    "plain_mro_before": plain_mro_before,
                    "quant_mro_before": quant_mro_before,
                    "plain_shape_before": list(plain_local.shape),
                    "plain_forward": plain_forward,
                    "quant_forward": quant_forward,
                    "quant_forward_error": quant_forward_error,
                    "check_status": check["status"],
                    "compared": check["compared"],
                    "matched": check["matched"],
                    "mismatched": check["mismatched"],
                    "samples": check["samples"],
                    "integrity": load_integrity(storage, check, report)["status"],
                    "wrong_integrity": load_integrity(storage, wrong, report)["status"],
                }
            ),
            encoding="utf-8",
        )
    except Exception:
        Path(out).write_text(json.dumps({"rank": rank, "error": "traceback", "message": traceback.format_exc()}), encoding="utf-8")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_pair(tmp: Path, *, missing_scale: bool, port: int) -> list[dict]:
    path = str(tmp / "weights.safetensors")
    procs = []
    outs = []
    context = get_context("spawn")
    for rank in range(2):
        out = str(tmp / f"rank{rank}.json")
        outs.append(out)
        procs.append(context.Process(target=_rank_worker, args=(rank, 2, path, port, missing_scale, out)))
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    assert [proc.exitcode for proc in procs] == [0, 0]
    return [json.loads(Path(out).read_text(encoding="utf-8")) for out in outs]


def test_integer_storage_is_not_expanded_into_the_float_placeholder() -> None:
    import torch

    from minimax_h3_t4.runtime.fsdp import load_from_full_model_state_dict

    layer = torch.nn.Linear(4, 4, bias=False)
    layer.weight = torch.nn.Parameter(torch.empty(4, 4, device="meta"))

    class Checkpoint(dict):
        def get(self, key, default=None):
            if key == "weight":
                return torch.ones(4, 4, dtype=torch.int8)
            return default

    with pytest.raises(RuntimeError, match="refusing to expand"):
        load_from_full_model_state_dict(layer, Checkpoint(), torch.device("cpu"), strict=True)


@pytest.mark.skipif(not COMFY.is_dir(), reason="pinned Comfy checkout is absent")
def test_two_rank_fsdp_replaces_meta_with_int8_shard(tmp_path: Path) -> None:
    rows = _run_pair(tmp_path, missing_scale=False, port=29541)
    for row in rows:
        assert row.get("error") is None, row
        assert row["still_meta"] == 0
        assert row["resident"] == 2
        assert row["quant_device"] == "cpu"
        assert row["quant_dtype"] == "torch.int8"
        assert row["quant_match"] is True
        assert row["plain_device"] == "cpu"
        assert row["plain_dtype"] == "torch.bfloat16"
        assert row["plain_match"] is True
        assert row["quant_mro_before"][0] == "DTensor", row
        assert row["plain_mro_before"][0] == "DTensor", row
        assert row["plain_shape_before"] == [2, 256]
        assert row["quant_forward_error"] is None, row
        assert row["compared"] >= 2
        assert row["matched"] == row["compared"]
        assert row["mismatched"] == 0
        assert any(sample.get("quantized") is True and sample.get("storage") == "quantized_qdata" for sample in row["samples"])
        assert row["integrity"] == "COMPLETE"
        assert row["wrong_integrity"] == "INCOMPLETE"
    assert rows[0]["plain_forward"] == rows[1]["plain_forward"]
    assert rows[0]["quant_forward"] == rows[1]["quant_forward"]
    assert rows[0]["quant_forward"] is not None


@pytest.mark.skipif(not COMFY.is_dir(), reason="pinned Comfy checkout is absent")
def test_two_rank_fsdp_rejects_a_missing_scale(tmp_path: Path) -> None:
    rows = _run_pair(tmp_path, missing_scale=True, port=29551)
    assert rows
    assert all(row.get("error") == "ValueError" for row in rows)
    assert all("scale" in row["message"].lower() for row in rows)
