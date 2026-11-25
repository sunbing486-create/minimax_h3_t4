from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from scripts.find_package_root import find_package_root


def _package(root: Path) -> None:
    script_dir = root / "scripts"
    script_dir.mkdir(parents=True)
    (script_dir / "run_actor_session.py").write_text("print('entry')\n", encoding="utf-8")
    (script_dir / "bootstrap_actor_admission.py").write_text("print('bootstrap')\n", encoding="utf-8")
    (root / "COMMIT.txt").write_text("test\n", encoding="utf-8")


def test_modern_kaggle_dataset_mount_is_found(tmp_path: Path) -> None:
    package = tmp_path / "datasets" / "rubbi486" / "h3-actor-admission-code" / "h3-actor-admission"
    _package(package)
    assert find_package_root([tmp_path], environ={}) == package.resolve()


def test_old_flat_input_layout_is_found(tmp_path: Path) -> None:
    package = tmp_path / "h3-actor-admission"
    _package(package)
    assert find_package_root([tmp_path], environ={}) == package.resolve()


def test_multiple_packages_require_an_explicit_root(tmp_path: Path) -> None:
    first = tmp_path / "datasets" / "owner" / "new-dataset" / "h3-actor-admission"
    second = tmp_path / "datasets" / "owner" / "old-dataset" / "h3-actor-admission"
    _package(first)
    _package(second)
    with pytest.raises(SystemExit, match="multiple admission packages"):
        find_package_root([tmp_path], environ={})
    assert find_package_root([tmp_path], environ={"H3_T4_PACKAGE_ROOT": str(first)}) == first.resolve()


def test_pinned_root_is_not_replaced_by_a_search_hit(tmp_path: Path) -> None:
    package = tmp_path / "datasets" / "owner" / "dataset" / "h3-actor-admission"
    _package(package)
    with pytest.raises(SystemExit, match="not the admission package"):
        find_package_root([tmp_path], environ={"H3_T4_PACKAGE_ROOT": str(tmp_path / "missing")})


def test_notebook_finder_matches_the_module_and_compiles() -> None:
    notebook = json.loads(Path("deploy/kaggle_actor_admission/kaggle_gpu_session.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 1
    ast.parse(cells[0])
    compile(cells[0], "<notebook>", "exec")
    module_tree = ast.parse(Path("scripts/find_package_root.py").read_text(encoding="utf-8"))
    notebook_tree = ast.parse(cells[0])
    module_fn = next(node for node in module_tree.body if isinstance(node, ast.FunctionDef) and node.name == "find_package_root")
    notebook_fn = next(node for node in notebook_tree.body if isinstance(node, ast.FunctionDef) and node.name == "find_package_root")
    assert ast.dump(module_fn) == ast.dump(notebook_fn)
    assert "seed.glob" not in cells[0]
    assert "MAX_DEPTH = 6" in cells[0]
