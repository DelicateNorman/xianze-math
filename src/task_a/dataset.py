"""Task A dataset utilities."""
from __future__ import annotations
import numpy as np
from src.common.io import load_pkl


def load_task_a_input(path: str) -> list[dict]:
    """Load Task A input file. Each item has traj_id, timestamps, coords, mask."""
    data = load_pkl(path)
    for item in data:
        item["coords"] = np.array(item["coords"], dtype=np.float64)
        item["timestamps"] = np.array(item["timestamps"], dtype=np.int64)
        item["mask"] = np.array(item["mask"], dtype=bool)
    return data


def load_task_a_gt(path: str) -> list[dict]:
    """Load Task A ground truth."""
    data = load_pkl(path)
    for item in data:
        item["coords"] = np.array(item["coords"], dtype=np.float64)
    return data


def align_gt(inputs: list[dict], gts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Align ground truth to input by traj_id."""
    gt_map = {g["traj_id"]: g for g in gts}
    aligned_gts = [gt_map[inp["traj_id"]] for inp in inputs]
    return inputs, aligned_gts
