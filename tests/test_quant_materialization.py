from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

COMFY = Path("/tmp/ComfyUI-h3")


def test_absent_quant_weight_is_loaded_and_counted() -> None:
    script = r"""
import json, sys
comfy_root, src_root, weight_path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path[:0] = [comfy_root, src_root]
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--cpu"]
import torch
from comfy.model_patcher import ModelPatcher
from comfy.ops import disable_weight_init, mixed_precision_ops
from safetensors.torch import save_file
from minimax_h3_t4.runtime.checkpoint_budget import LazyCheckpoint
from minimax_h3_t4.runtime.fsdp import load_from_full_model_state_dict, register_absent_weight_parameters
from minimax_h3_t4.runtime.unet_load import bounded_weight_check, load_integrity, summarize_module_storage

plain = disable_weight_init.Linear(8, 4, bias=False, device=torch.device("meta"))
quant_ops = mixed_precision_ops({}, torch.bfloat16)
quant = quant_ops.Linear(8, 4, bias=True, device=torch.device("cpu"))
host = torch.nn.Module()
host.plain = plain
host.quant = quant
patcher = ModelPatcher(host, torch.device("cpu"), torch.device("cpu"), size=1)
before = summarize_module_storage(patcher)
registered = register_absent_weight_parameters(host)
weight = torch.arange(32, dtype=torch.int8).reshape(4, 8)
conf = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode()
path = weight_path
save_file({
    "quant.weight": weight,
    "quant.weight_scale": torch.tensor(0.25),
    "quant.comfy_quant": torch.tensor(list(conf), dtype=torch.uint8),
    "quant.bias": torch.ones(4),
    "plain.weight": torch.arange(32, dtype=torch.float32).reshape(4, 8),
}, path)

def getter(key):
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)

keys = ["quant.weight", "quant.weight_scale", "quant.comfy_quant", "quant.bias", "plain.weight"]
report = load_from_full_model_state_dict(host, LazyCheckpoint(getter, keys), torch.device("cpu"), strict=False, cpu_offload=True)
after = summarize_module_storage(patcher)
outer = torch.nn.Module()
outer.diffusion_model = host
check = bounded_weight_check(outer, path, rank=0, world_size=1)
qdata = quant.weight._qdata
print(json.dumps({
    "patcher_parameters": callable(getattr(patcher, "parameters", None)),
    "before_materialized": before["materialized_bytes"],
    "registered": registered,
    "report": report,
    "after_parameter_count": after["parameter_count"],
    "after_materialized": after["materialized_bytes"],
    "after_uninitialized": after["uninitialized_weights"],
    "counted_type": after["counted_type"],
    "qdata": qdata.view(torch.int8).flatten().tolist(),
    "plain_sum": float(plain.weight.detach().sum()),
    "meta_bytes": after["meta_bytes"],
    "integrity": load_integrity(after, check, report)["status"],
    "check": check,
    "false_pass": load_integrity(
        {"materialized_bytes": 21794304, "meta_bytes": 80423927808, "uninitialized_weights": 0, "logical_bytes": 80445722112},
        {"status": "NOT_COMPARABLE", "compared": 0, "matched": 0},
        {"matched": 532, "missing": 0, "still_meta": 264, "requested": 532},
    )["status"],
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(COMFY), "/workspace/src", "/tmp/h3-tiny-quant-load.safetensors"],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["patcher_parameters"] is False
    assert report["before_materialized"] < report["after_materialized"]
    assert report["registered"] == 1
    assert report["report"]["matched"] >= 2
    assert report["report"]["missing"] == 0
    assert report["after_materialized"] >= 32
    assert report["after_parameter_count"] >= 2
    assert report["after_uninitialized"] == 0
    assert report["counted_type"] != "ModelPatcher"
    assert report["qdata"] == list(range(32))
    assert report["plain_sum"] == float(sum(range(32)))
    assert report["meta_bytes"] == 0
    assert report["integrity"] == "COMPLETE"
    assert report["false_pass"] == "INCOMPLETE"


def test_missing_quant_scale_is_rejected() -> None:
    script = r"""
import json, sys
comfy_root, src_root, weight_path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path[:0] = [comfy_root, src_root]
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--cpu"]
import torch
from comfy.ops import mixed_precision_ops
from safetensors.torch import save_file
from minimax_h3_t4.runtime.checkpoint_budget import LazyCheckpoint
from minimax_h3_t4.runtime.fsdp import load_from_full_model_state_dict, register_absent_weight_parameters
ops = mixed_precision_ops({}, torch.bfloat16)
quant = ops.Linear(8, 4, bias=False, device=torch.device("cpu"))
register_absent_weight_parameters(quant)
weight = torch.arange(32, dtype=torch.int8).reshape(4, 8)
conf = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode()
save_file({"weight": weight, "comfy_quant": torch.tensor(list(conf), dtype=torch.uint8)}, weight_path)

def getter(key):
    from safetensors import safe_open
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)
try:
    load_from_full_model_state_dict(quant, LazyCheckpoint(getter, ["weight", "comfy_quant"]), torch.device("cpu"), strict=True)
except ValueError as exc:
    print(type(exc).__name__)
    raise SystemExit(0)
raise SystemExit(2)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(COMFY), "/workspace/src", "/tmp/h3-missing-scale.safetensors"],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
