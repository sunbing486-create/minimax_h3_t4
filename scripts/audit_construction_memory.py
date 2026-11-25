#!/usr/bin/env python3
"""Isolated RSS measurements for MiniMax-H3 construction.

Each case is a fresh process. The full architecture is constructed only on
the meta device. Meta logical bytes are not RSS.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def memory_now() -> dict[str, int | None]:
    rss = None
    pss = None
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
    rollup = Path("/proc/self/smaps_rollup")
    if rollup.is_file():
        for line in rollup.read_text(encoding="utf-8").splitlines():
            if line.startswith("Pss:"):
                pss = int(line.split()[1]) * 1024
    return {"rss_bytes": rss, "pss_bytes": pss}


def parameter_audit(module) -> dict[str, object]:
    logical = 0
    meta = 0
    by_device: dict[str, int] = {}
    dtypes: dict[str, int] = {}
    for parameter in module.parameters():
        if parameter is None:
            continue
        nbytes = int(parameter.numel()) * int(parameter.element_size())
        logical += nbytes
        device = str(parameter.device)
        by_device[device] = by_device.get(device, 0) + nbytes
        dtype = str(parameter.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + nbytes
        if parameter.device.type == "meta":
            meta += nbytes
    return {
        "logical_parameter_bytes": logical,
        "meta_parameter_bytes": meta,
        "bytes_by_device": by_device,
        "bytes_by_dtype": dtypes,
        "meta_is_not_rss": True,
    }


def _boot_comfy():
    import comfy.options

    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0], "--cpu"]
    import torch
    from comfy.ldm.minimax.model import MiniMaxH3Model
    import comfy.memory_management
    import comfy.ops

    return torch, MiniMaxH3Model, comfy.memory_management, comfy.ops


def tiny_kwargs(torch) -> dict[str, object]:
    return dict(
        hidden_size=32,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=8,
        ffn_hidden_size=64,
        latents_dim=4,
        audio_latents_dim=4,
        text_dim=16,
        timestep_input_dim=8,
        time_embed_hidden_size=32,
        time_embed_dim=16,
        rope_inv_freq_len=4,
        dtype=torch.float32,
    )


def medium_kwargs(torch) -> dict[str, object]:
    return dict(
        hidden_size=128,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=32,
        ffn_hidden_size=256,
        latents_dim=8,
        audio_latents_dim=8,
        text_dim=32,
        timestep_input_dim=16,
        time_embed_hidden_size=64,
        time_embed_dim=32,
        rope_inv_freq_len=8,
        dtype=torch.float32,
    )


def run_case(case: str) -> dict[str, object]:
    torch, MiniMaxH3Model, memory_management, ops = _boot_comfy()
    before = memory_now()
    audit: dict[str, object] | None = None
    extra: dict[str, object] = {}
    if case == "tiny-cpu":
        model = MiniMaxH3Model(**tiny_kwargs(torch), device=torch.device("cpu"), operations=torch.nn)
        audit = parameter_audit(model)
    elif case == "tiny-meta":
        model = MiniMaxH3Model(**tiny_kwargs(torch), device=torch.device("meta"), operations=torch.nn)
        audit = parameter_audit(model)
    elif case == "tiny-cpu-then-meta":
        model = MiniMaxH3Model(**tiny_kwargs(torch), device=torch.device("cpu"), operations=torch.nn)
        occupied = memory_now()
        model.to("meta")
        gc.collect()
        released = memory_now()
        extra = {
            "rss_after_construct_bytes": occupied["rss_bytes"],
            "rss_after_to_meta_bytes": released["rss_bytes"],
            "rss_change_on_to_meta_bytes": None
            if occupied["rss_bytes"] is None or released["rss_bytes"] is None
            else released["rss_bytes"] - occupied["rss_bytes"],
        }
        audit = parameter_audit(model)
    elif case == "medium-cpu":
        model = MiniMaxH3Model(**medium_kwargs(torch), device=torch.device("cpu"), operations=torch.nn)
        audit = parameter_audit(model)
    elif case == "medium-meta":
        model = MiniMaxH3Model(**medium_kwargs(torch), device=torch.device("meta"), operations=torch.nn)
        audit = parameter_audit(model)
    elif case == "medium-cpu-then-meta":
        model = MiniMaxH3Model(**medium_kwargs(torch), device=torch.device("cpu"), operations=torch.nn)
        occupied = memory_now()
        model.to("meta")
        gc.collect()
        released = memory_now()
        extra = {
            "rss_after_construct_bytes": occupied["rss_bytes"],
            "pss_after_construct_bytes": occupied["pss_bytes"],
            "rss_after_to_meta_bytes": released["rss_bytes"],
            "pss_after_to_meta_bytes": released["pss_bytes"],
        }
        audit = parameter_audit(model)
    elif case == "tiny-aimdo-off":
        memory_management.aimdo_enabled = False
        model = MiniMaxH3Model(**tiny_kwargs(torch), device=None, operations=ops.disable_weight_init)
        audit = parameter_audit(model)
        extra = {"weight_is_none": model.video_patch_proj.weight is None}
    elif case == "tiny-aimdo-on":
        memory_management.aimdo_enabled = True
        model = MiniMaxH3Model(**tiny_kwargs(torch), device=None, operations=ops.disable_weight_init)
        audit = parameter_audit(model)
        extra = {"weight_is_none": model.video_patch_proj.weight is None}
    elif case == "real-meta-default-dtype":
        model = MiniMaxH3Model(device=torch.device("meta"), operations=torch.nn)
        audit = parameter_audit(model)
    elif case == "real-meta-bfloat16":
        model = MiniMaxH3Model(dtype=torch.bfloat16, device=torch.device("meta"), operations=torch.nn)
        audit = parameter_audit(model)
    elif case == "shard-dict-committed":
        start = memory_now()
        held = {str(index): torch.zeros(8 * 1024 * 1024, dtype=torch.float32) for index in range(4)}
        occupied = memory_now()
        del held
        gc.collect()
        released = memory_now()
        return {
            "case": case,
            "torch": torch.__version__,
            "before": start,
            "while_holding_128MiB": occupied,
            "after_delete": released,
            "logical_held_bytes": 4 * 8 * 1024 * 1024 * 4,
        }
    elif case == "one-tensor-reuse":
        start = memory_now()
        peaks = []
        for _index in range(4):
            tensor = torch.zeros(8 * 1024 * 1024, dtype=torch.float32)
            peaks.append(memory_now()["rss_bytes"])
            del tensor
            gc.collect()
        return {"case": case, "torch": torch.__version__, "before": start, "rss_each_tensor": peaks, "after": memory_now(), "logical_one_tensor_bytes": 8 * 1024 * 1024 * 4}
    else:
        raise SystemExit(f"unknown case {case}")
    after = memory_now()
    return {
        "case": case,
        "torch": torch.__version__,
        "before": before,
        "after": after,
        "rss_delta_bytes": None if before["rss_bytes"] is None or after["rss_bytes"] is None else after["rss_bytes"] - before["rss_bytes"],
        "pss_delta_bytes": None if before["pss_bytes"] is None or after["pss_bytes"] is None else after["pss_bytes"] - before["pss_bytes"],
        "audit": audit,
        **extra,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="")
    args = parser.parse_args(argv)
    if args.case:
        print(json.dumps(run_case(args.case)))
        return 0
    cases = [
        "tiny-cpu",
        "tiny-meta",
        "tiny-cpu-then-meta",
        "medium-cpu",
        "medium-meta",
        "medium-cpu-then-meta",
        "tiny-aimdo-off",
        "tiny-aimdo-on",
        "real-meta-default-dtype",
        "real-meta-bfloat16",
        "shard-dict-committed",
        "one-tensor-reuse",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(Path("/tmp/ComfyUI-h3")), str(SRC), env.get("PYTHONPATH", "")])
    rows = []
    for case in cases:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--case", case],
            check=False,
            text=True,
            capture_output=True,
            env=env,
        )
        if completed.returncode != 0:
            rows.append({"case": case, "ok": False, "stderr": (completed.stderr or "")[-800:]})
            continue
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        payload["ok"] = True
        rows.append(payload)
    from minimax_h3_t4.runtime.model_facts import header_summary

    summary = header_summary()
    report = {
        "torch_under_test": rows[0].get("torch") if rows else None,
        "torch_note": "This CPU environment is not Kaggle torch 2.11.0+cu130.",
        "comfy_revision": "2fcb6cef9ab1b6dc556b7cff1765a0724ad5e884",
        "model_stage_ready": False,
        "video_inference_claimed": False,
        "cases": rows,
        "header": {
            "unet_declared_payload_bytes": summary["minimax_h3_fl2va_pruned_int8_convrot.safetensors"]["declared_payload_bytes"],
            "unet_max_tensor_bytes": summary["minimax_h3_fl2va_pruned_int8_convrot.safetensors"]["largest"][0]["bytes"],
            "text_declared_payload_bytes": summary["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"]["declared_payload_bytes"],
            "text_max_tensor_bytes": summary["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"]["largest"][0]["bytes"],
        },
    }
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
