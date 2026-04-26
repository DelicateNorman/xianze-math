"""Pickle I/O utilities."""
import pickle
from pathlib import Path
from typing import Any


def load_pkl(path: str | Path) -> Any:
    """Load a pickle file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj: Any, path: str | Path) -> None:
    """Save object to pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def validate_task_a_output(predictions: list[dict]) -> list[str]:
    """Check Task A submission format. Returns list of error messages."""
    import numpy as np
    errors = []
    for i, item in enumerate(predictions):
        if "traj_id" not in item:
            errors.append(f"[{i}] missing traj_id")
        if "coords" not in item:
            errors.append(f"[{i}] missing coords")
            continue
        coords = np.array(item["coords"])
        if coords.ndim != 2 or coords.shape[1] != 2:
            errors.append(f"[{i}] coords shape {coords.shape} invalid")
        if np.any(np.isnan(coords)):
            errors.append(f"[{i}] coords contains NaN")
    return errors


def validate_task_b_output(predictions: list[dict]) -> list[str]:
    """Check Task B submission format. Returns list of error messages."""
    import numpy as np
    errors = []
    for i, item in enumerate(predictions):
        if "traj_id" not in item:
            errors.append(f"[{i}] missing traj_id")
        if "travel_time" not in item:
            errors.append(f"[{i}] missing travel_time")
            continue
        tt = item["travel_time"]
        if np.isnan(tt) or np.isinf(tt):
            errors.append(f"[{i}] travel_time is nan/inf")
        if tt <= 0:
            errors.append(f"[{i}] travel_time={tt} not positive")
    return errors
