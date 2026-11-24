"""Locate this admission package without guessing among older copies."""

from __future__ import annotations

import os
from pathlib import Path

ENTRY_SCRIPT = Path("scripts") / "run_actor_session.py"
MARKER_SCRIPT = Path("scripts") / "bootstrap_actor_admission.py"
MAX_DEPTH = 6


def default_seeds() -> list[Path]:
    return [Path.cwd(), Path("/kaggle/input"), Path("/kaggle/working")]


def find_package_root(seeds: list[Path] | None = None, environ: dict[str, str] | None = None) -> Path:
    """Return the only directory that contains the current admission entry.

    ``H3_T4_PACKAGE_ROOT`` selects one directory and disables the search.
    A modern Kaggle mount is ``/kaggle/input/datasets/<owner>/<dataset>/<package>``.
    Older sessions mounted the package directly under ``/kaggle/input`` or
    ``/kaggle/working``. More than one match is an error.
    """

    env = os.environ if environ is None else environ
    pinned = env.get("H3_T4_PACKAGE_ROOT", "").strip()
    if pinned:
        root = Path(pinned)
        if not _is_admission_package(root):
            raise SystemExit(f"H3_T4_PACKAGE_ROOT is not the admission package: {root}")
        return root.resolve()
    found: list[Path] = []
    seen: set[str] = set()
    for seed in seeds if seeds is not None else default_seeds():
        if not seed.exists():
            continue
        for match in _scan(seed.resolve(), MAX_DEPTH):
            key = str(match.resolve())
            if key not in seen:
                seen.add(key)
                found.append(match.resolve())
    if not found:
        raise SystemExit(
            "admission package not found. Upload the folder that contains "
            "scripts/run_actor_session.py or set H3_T4_PACKAGE_ROOT."
        )
    if len(found) > 1:
        lines = "\n".join(str(path) for path in found)
        raise SystemExit("multiple admission packages found; set H3_T4_PACKAGE_ROOT to one:\n" + lines)
    return found[0]


def _is_admission_package(root: Path) -> bool:
    return (root / ENTRY_SCRIPT).is_file() and (root / MARKER_SCRIPT).is_file()


def _scan(seed: Path, max_depth: int) -> list[Path]:
    matches: list[Path] = []
    for current, dirnames, _filenames in os.walk(seed):
        current_path = Path(current)
        try:
            depth = len(current_path.relative_to(seed).parts)
        except ValueError:
            depth = max_depth + 1
        if depth > max_depth:
            dirnames.clear()
            continue
        if _is_admission_package(current_path):
            matches.append(current_path)
            dirnames.clear()
    return matches
