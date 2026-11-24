from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

HELPER = r"""
import json, os, sys
from pathlib import Path

import comfy.options
comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--cpu"]

import torch
from safetensors.torch import save_file
from comfy.model_base import BaseModel
from comfy.model_detection import detect_unet_config

sys.path.insert(0, os.environ["H3_SRC"])
from minimax_h3_t4.runtime.checkpoint_budget import LazyCheckpoint

def rss():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0

path = Path(sys.argv[1])
weight = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
other = torch.tensor([9.0, 8.0])
save_file({
    "weight": weight,
    "bias": other,
    "video_patch_proj.weight": torch.zeros(4, 2),
    "condition_proj.weight": torch.zeros(2, 3),
    "audio_patch_proj.weight": torch.zeros(4, 2),
    "final_layer.video_out.weight": torch.zeros(4, 2),
    "final_layer.audio_out.weight": torch.zeros(1, 2),
    "blocks.0.attn.q_norm.weight": torch.zeros(2),
    "blocks.0.attn.qkv_proj.weight": torch.zeros(6, 4),
    "blocks.0.mlp.fc1.weight": torch.zeros(4, 4),
    "time_embedder.proj_in.weight": torch.zeros(4, 2),
    "time_embedder.proj_out.weight": torch.zeros(3, 4),
    "rope.inv_freq": torch.zeros(5),
}, str(path))
reads = []

def load_key(key):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        value = handle.get_tensor(key)
    reads.append({"key": key, "sum": float(value.sum()), "rss": rss()})
    return value

lazy = LazyCheckpoint(load_key, list({
    "weight": 0, "bias": 0, "video_patch_proj.weight": 0, "condition_proj.weight": 0,
    "audio_patch_proj.weight": 0, "final_layer.video_out.weight": 0, "final_layer.audio_out.weight": 0,
    "blocks.0.attn.q_norm.weight": 0, "blocks.0.attn.qkv_proj.weight": 0, "blocks.0.mlp.fc1.weight": 0,
    "time_embedder.proj_in.weight": 0, "time_embedder.proj_out.weight": 0, "rope.inv_freq": 0,
}))
assert lazy.resident_bytes == 0
detected = detect_unet_config(lazy, "")
lazy.release_materialized()
host = type("Host", (), {})()
host.model_config = type("Config", (), {"process_unet_state_dict": staticmethod(lambda sd: sd)})()
module = torch.nn.Linear(4, 2)
with torch.no_grad():
    module.weight.copy_(torch.zeros_like(module.weight))
    module.bias.copy_(torch.zeros_like(module.bias))
host.diffusion_model = module
before = rss()
BaseModel.load_model_weights(host, lazy, "", assign=True)
after = rss()
report = {
    "detected_image_model": None if not isinstance(detected, dict) else detected.get("image_model"),
    "reads": reads,
    "pop_resident_after": lazy.resident_bytes,
    "keys_left": sorted(lazy.keys()),
    "weight_sum": float(module.weight.detach().sum()),
    "bias_sum": float(module.bias.detach().sum()),
    "rss_before": before,
    "rss_after": after,
    "used_load_file": False,
}
print(json.dumps(report))
"""


def test_real_comfy_load_model_weights_pops_tensors(tmp_path: Path) -> None:
    script = tmp_path / "load_once.py"
    script.write_text(textwrap.dedent(HELPER), encoding="utf-8")
    env = os_env()
    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "tiny.safetensors")],
        check=False,
        text=True,
        capture_output=True,
        env=env,
        cwd="/workspace",
    )
    assert completed.returncode == 0, completed.stderr[-1500:]
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["detected_image_model"] == "minimax_h3"
    assert report["weight_sum"] == 36.0
    assert report["bias_sum"] == 17.0
    assert report["pop_resident_after"] == 0
    assert report["keys_left"] == []
    read_keys = [item["key"] for item in report["reads"]]
    assert read_keys[0] == "video_patch_proj.weight"
    assert "weight" in read_keys
    assert "bias" in read_keys
    assert report["used_load_file"] is False


def test_one_key_assign_releases_before_the_next_key(tmp_path: Path) -> None:
    script = tmp_path / "stream.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, os, sys
            from pathlib import Path
            import torch
            from safetensors.torch import save_file
            sys.path.insert(0, os.environ["H3_SRC"])
            from minimax_h3_t4.runtime.checkpoint_budget import LazyCheckpoint
            path = Path(sys.argv[1])
            save_file({"weight": torch.arange(8, dtype=torch.float32).reshape(2, 4), "bias": torch.tensor([1.0, 2.0])}, str(path))
            peaks = []
            def load_key(key):
                from safetensors import safe_open
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    return handle.get_tensor(key)
            lazy = LazyCheckpoint(load_key, ["weight", "bias"])
            module = torch.nn.Linear(4, 2)
            for key in list(lazy.keys()):
                tensor = lazy.pop(key)
                peaks.append(lazy.resident_bytes)
                module.load_state_dict({key: tensor}, strict=False, assign=True)
                del tensor
            print(json.dumps({"peaks_after_pop": peaks, "weight0": float(module.weight.detach().view(-1)[0]), "resident": lazy.resident_bytes}))
            """
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "stream.safetensors")],
        check=False,
        text=True,
        capture_output=True,
        env=os_env(),
        cwd="/workspace",
    )
    assert completed.returncode == 0, completed.stderr[-1000:]
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["peaks_after_pop"] == [0, 0]
    assert report["weight0"] == 0.0
    assert report["resident"] == 0


def os_env() -> dict[str, str]:
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = "/tmp/ComfyUI-h3:/workspace/src"
    env["H3_SRC"] = "/workspace/src"
    return env
