"""How much host RAM a two-rank UNet load holds, without reading tensor payloads.

``safetensors.torch.load_file`` materializes every tensor. The old worker did
that on each rank and then copied the dict again via ``model_state_dict``.
``release_sd`` drops full tensors only after that copy exists. A streaming
reader keeps one tensor, shards it, then drops it.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


def host_available_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    """Read the JSON header only. Tensor bytes stay on disk."""

    file_path = Path(path)
    with file_path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{file_path} is not a safetensors file")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = json.loads(handle.read(header_len))
    tensors = []
    total = 0
    largest = 0
    for key, meta in header.items():
        if key == "__metadata__" or not isinstance(meta, dict):
            continue
        shape = [int(dim) for dim in meta.get("shape", [])]
        count = 1
        for dim in shape or [1]:
            count *= dim
        nbytes = count * _DTYPE_BYTES.get(str(meta.get("dtype", "")), 0)
        total += nbytes
        largest = max(largest, nbytes)
        tensors.append({"key": key, "shape": shape, "dtype": meta.get("dtype"), "bytes": nbytes})
    return {"tensors": tensors, "total_bytes": total, "max_tensor_bytes": largest, "header_bytes_read": 8 + header_len}


def sequential_resident_plan(
    *,
    file_bytes: int,
    max_tensor_bytes: int,
    host_available_bytes: int,
    world_size: int = 2,
) -> dict[str, Any]:
    """Predict host RAM for the old full-file load and for streaming shards.

    VRAM is not this number. A file larger than one T4 can still be sharded.
    """

    shard_bytes = int(file_bytes) // world_size
    old_second_rank = shard_bytes + int(file_bytes)
    old_with_second_copy = shard_bytes + int(file_bytes) * 2
    streaming_second_rank = shard_bytes + int(max_tensor_bytes)
    available = int(host_available_bytes)
    return {
        "file_bytes": int(file_bytes),
        "max_tensor_bytes": int(max_tensor_bytes),
        "world_size": world_size,
        "shard_lower_bound_bytes": shard_bytes,
        "old_mechanism": "each rank load_file materializes the whole file; model_state_dict can hold a second copy; release_sd runs after that",
        "old_second_rank_peak_bytes": old_second_rank,
        "old_second_rank_fits": old_second_rank < available,
        "old_second_copy_peak_bytes": old_with_second_copy,
        "old_second_copy_fits": old_with_second_copy < available,
        "streaming_second_rank_peak_bytes": streaming_second_rank,
        "streaming_second_rank_fits": streaming_second_rank < available and int(max_tensor_bytes) < available,
        "host_available_bytes": available,
        "inference_memory_proven": False,
    }


class LazyCheckpoint(dict):
    """Materialize a checkpoint key only when that key is read.

    Assigning None, which ``release_sd`` does, drops that tensor before the
    next key is read. ``prefix`` is stripped without copying tensors.
    """

    def __init__(self, loader: Any, keys: list[str], prefix: str = "") -> None:
        mapped = [key[len(prefix) :] for key in keys if key.startswith(prefix)] if prefix else list(keys)
        super().__init__({key: None for key in mapped})
        self._loader = loader
        self._prefix = prefix
        self._file_keys = {key[len(prefix) :] if prefix and key.startswith(prefix) else key: key for key in keys}
        self._resident = 0
        self._peak = 0
        self.materialized_keys: list[str] = []

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        if value is None and key in self:
            value = self._loader(self._file_keys.get(key, key))
            super().__setitem__(key, value)
            self.materialized_keys.append(key)
            self._resident += _nbytes(value)
            self._peak = max(self._peak, self._resident)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        previous = super().get(key)
        if previous is not None and value is None:
            self._resident = max(0, self._resident - _nbytes(previous))
        super().__setitem__(key, value)

    @property
    def peak_bytes(self) -> int:
        return self._peak

    @property
    def resident_bytes(self) -> int:
        return self._resident


def load_row_shard(tensor: Any, rank: int, world_size: int) -> Any:
    """CPU stand-in for taking one rank's rows without keeping the other rows."""

    if world_size < 1:
        raise ValueError("world_size must be positive")
    rows = int(tensor.shape[0])
    if rows < world_size:
        raise ValueError("tensor does not have a row for every rank")
    start = rank * (rows // world_size)
    end = rows if rank == world_size - 1 else (rank + 1) * (rows // world_size)
    shard = tensor.narrow(0, start, end - start).contiguous()
    return shard


def _nbytes(value: Any) -> int:
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    return 0
