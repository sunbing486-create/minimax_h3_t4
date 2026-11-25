from __future__ import annotations

import ast
import json
from pathlib import Path

from scripts.run_actor_session import ACTOR_BUDGET_S, CHECK_TIMEOUT_S, decide_actor, decide_check


def test_check_timeout_is_at_least_60_and_29_seconds_does_not_run() -> None:
    assert CHECK_TIMEOUT_S >= 60
    assert ACTOR_BUDGET_S == 45 + 60 + 20
    assert decide_check(29) == "NOT_RUN_BUDGET"
    assert decide_check(CHECK_TIMEOUT_S) == "RUN"


def test_actor_starts_only_with_gpu_gate_and_full_budget() -> None:
    assert decide_actor(ACTOR_BUDGET_S - 1, gpu_permitted=True) == "NOT_RUN_BUDGET"
    assert decide_actor(ACTOR_BUDGET_S, gpu_permitted=False) == "NOT_RUN_GPU_GATE"
    assert decide_actor(ACTOR_BUDGET_S, gpu_permitted=True) == "RUN"


def test_notebook_cells_compile() -> None:
    notebook = Path("deploy/kaggle_actor_admission/kaggle_gpu_session.ipynb")
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert cells
    for source in cells:
        ast.parse(source)
        compile(source, "<notebook>", "exec")
        assert "except:" not in source
        assert "raise SystemExit(code)" not in source
        assert "raise SystemExit(subprocess.call" not in source
        assert "timeout" not in source or "29" not in source
        assert "show_session_evidence" in source
