"""Path helpers for flexible local data layouts."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


DATA_ROOT_ENV_VARS = ("XIANZE_DATA_ROOT", "DATA_ROOT")


def candidate_data_roots() -> list[Path]:
    """Return data roots in priority order.

    The course files have appeared in both of these layouts:
    - data/student_release/data_ds15/train.pkl
    - data/data_ds15/train.pkl

    Environment variables are checked first so teammates can keep data wherever
    they want without editing config files.
    """
    roots: list[Path] = []
    for name in DATA_ROOT_ENV_VARS:
        value = os.environ.get(name)
        if value:
            roots.append(Path(value))
    roots.extend([Path("data/student_release"), Path("data")])

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def path_candidates(path: str | Path) -> list[Path]:
    """Build likely alternatives for a path without touching the filesystem."""
    original = Path(path)
    candidates = [original]
    if original.is_absolute():
        return candidates

    parts = original.parts
    suffixes: list[Path] = []

    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "student_release":
        suffixes.append(Path(*parts[2:]))
    elif len(parts) >= 1 and parts[0] == "data":
        suffixes.append(Path(*parts[1:]))
    else:
        suffixes.append(original)

    for suffix in suffixes:
        if str(suffix) == ".":
            continue
        for root in candidate_data_roots():
            candidates.append(root / suffix)

    # Explicit cross-layout aliases for the most common hard-coded forms.
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "student_release":
        candidates.append(Path("data") / Path(*parts[2:]))
    elif len(parts) >= 1 and parts[0] == "data":
        candidates.append(Path("data/student_release") / Path(*parts[1:]))

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def resolve_existing_path(path: str | Path, *, required: bool = True) -> Path | None:
    """Resolve a file path across supported local data layouts."""
    candidates = path_candidates(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        formatted = "\n  - ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"File not found. Tried:\n  - {formatted}")
    return None


def first_existing_path(paths: Iterable[str | Path]) -> Path | None:
    """Return the first existing path after applying layout aliases."""
    for path in paths:
        resolved = resolve_existing_path(path, required=False)
        if resolved is not None:
            return resolved
    return None
