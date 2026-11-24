#!/usr/bin/env python3
"""Run the weight-free two-rank probe.

CPU and mock modes are control-flow checks. They are not a dual-T4 result and
they do not generate video. ``--mode cuda-nccl`` refuses to start unless two
CUDA devices are already visible, performs one allreduce, and does not download
weights. On failure the process exits immediately after writing
``evidence.zip`` and ``evidence.zip.sha256`` under ``--evidence-dir``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimax_h3_t4.runtime.dual_gpu_probe import main


if __name__ == "__main__":
    raise SystemExit(main())
