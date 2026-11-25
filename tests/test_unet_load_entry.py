from __future__ import annotations

import ast
import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from minimax_h3_t4.runtime.official_transfer import transfer_file
from minimax_h3_t4.runtime.unet_load import (
    UNET_SPEC,
    bounded_weight_check,
    choose_weight_dir,
    decide_unet_load,
    required_files,
    run_unet_load_only,
    summarize_module_storage,
)
from scripts.prepare_model_stage import download_official


class _Disk:
    def __init__(self, total: int, free: int) -> None:
        self.total = total
        self.free = free


def test_unet_notebook_compiles_and_does_not_enable_transfer_by_default() -> None:
    notebook = json.loads(Path("deploy/kaggle_actor_admission/kaggle_unet_load.ipynb").read_text(encoding="utf-8"))
    source = "".join(next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")["source"])
    ast.parse(source)
    compile(source, "<notebook>", "exec")
    assert 'os.environ.get("H3_T4_ENABLE_TRANSFER", "0")' in source


def test_unet_load_requires_only_the_official_unet_file() -> None:
    files = required_files()
    assert [item["name"] for item in files] == ["unet"]
    assert files[0]["bytes"] == 20970379616
    assert 20957446144 < files[0]["bytes"]


def test_working_disk_is_not_selected_when_it_is_smaller_than_the_file(tmp_path: Path) -> None:
    working = tmp_path / "working"
    temporary = tmp_path / "tmp"
    working.mkdir()
    temporary.mkdir()

    def usage(path: Path) -> _Disk:
        if path == working:
            return _Disk(20957446144, 20957446144)
        return _Disk(80 * 1024**3, 80 * 1024**3)

    assert choose_weight_dir([working, temporary], int(UNET_SPEC["bytes"]), usage) == temporary
    assert choose_weight_dir([working], int(UNET_SPEC["bytes"]), usage) is None


def test_load_only_does_not_start_without_cuda_or_the_file(tmp_path: Path) -> None:
    decision = decide_unet_load(unet_reusable=False, cuda_count=0, allow=False, available_bytes=30 * 1024**3)
    assert decision["action"] == "NOT_RUN"
    assert decision["unet_loaded"] is False
    assert decision["video_inference_claimed"] is False
    report = run_unet_load_only(unet_path=None, cuda_count=0, allow=True, available_bytes=30 * 1024**3)
    assert report["action"] == "NOT_RUN"
    assert not (tmp_path / "should-not-exist").exists()


def test_meta_parameters_do_not_count_as_loaded() -> None:
    layer = torch.nn.Linear(4, 2, device="meta")
    summary = summarize_module_storage(layer)
    assert summary["meta_bytes"] == summary["logical_bytes"]
    assert summary["materialized_bytes"] == 0
    assert summary["meta_counts_as_loaded"] is False


def test_bounded_check_matches_a_real_safetensors_value(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    path = tmp_path / "tiny.safetensors"
    save_file({"weight": weight}, str(path))
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.copy_(weight)
    report = bounded_weight_check(module, str(path))
    assert report["status"] == "CHECKED"
    assert report["matched"] == 1


def test_enabled_transfer_resumes_and_checks_the_hash(tmp_path: Path) -> None:
    payload = os.urandom(128 * 1024)
    digest = hashlib.sha256(payload).hexdigest()
    server, url = _serve(payload)
    dest = tmp_path / "file.bin"
    partial = tmp_path / "file.bin.partial"
    partial.write_bytes(payload[:1000])
    try:
        result = transfer_file(url=url, dest=dest, expected_bytes=len(payload), expected_sha256=digest, deadline=time_plus(10))
    finally:
        server.shutdown()
    assert result["ok"] is True
    assert result["started_transfer"] is True
    assert result["action"] == "transferred"
    assert dest.read_bytes() == payload
    assert not partial.exists()


def test_confirm_without_enable_does_not_create_a_file(tmp_path: Path) -> None:
    report = download_official(
        confirm=True,
        enable_transfer=False,
        only="minimax_h3_audio_vae_fp32.safetensors",
        dest_dir=str(tmp_path),
        deadline=time_plus(5),
    )
    assert report["started_transfer"] is False
    assert list(tmp_path.iterdir()) == []


def _serve(payload: bytes):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            start = 0
            end = len(payload) - 1
            header = self.headers.get("Range")
            status = 200
            if header and header.startswith("bytes="):
                start_text, end_text = header.removeprefix("bytes=").split("-", 1)
                start = int(start_text or "0")
                end = int(end_text) if end_text else len(payload) - 1
                status = 206
            body = payload[start : end + 1]
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/file"


def time_plus(seconds: float) -> float:
    import time

    return time.monotonic() + seconds
