"""Task B dataset utilities."""
from __future__ import annotations
import numpy as np
from src.common.io import load_pkl


def load_task_b_input(path: str) -> list[dict]:
    data = load_pkl(path)
    for item in data:
        item["coords"] = np.array(item["coords"], dtype=np.float64)
    return data


def load_task_b_gt(path: str) -> list[dict]:
    return load_pkl(path)


def load_train_data(path: str) -> list[dict]:
    """Load ds15 training data and compute travel_time from timestamps."""
    data = load_pkl(path)
    processed = []
    for item in data:
        coords = np.array(item["coords"], dtype=np.float64)
        ts = np.array(item["timestamps"], dtype=np.int64)
        travel_time = int(ts[-1] - ts[0])
        if travel_time <= 0 or len(coords) < 2:
            continue
        processed.append({
            "coords": coords,
            "departure_timestamp": int(ts[0]),
            "travel_time": travel_time,
            "order_id": item.get("order_id", ""),
        })
    return processed
