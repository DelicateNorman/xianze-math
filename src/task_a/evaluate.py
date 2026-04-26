"""Task A evaluation: Haversine MAE and RMSE on masked-out points."""
from __future__ import annotations
import numpy as np
from src.common.geo import haversine_batch


def evaluate_task_a(predictions: list[dict], ground_truths: list[dict],
                    inputs: list[dict]) -> dict:
    """
    Compute MAE and RMSE over all missing (mask=False) points.

    predictions: list of {'traj_id': ..., 'coords': np.ndarray}
    ground_truths: list of {'traj_id': ..., 'coords': np.ndarray}
    inputs: list of Task A input dicts with 'mask' field
    """
    gt_map = {g["traj_id"]: g for g in ground_truths}
    inp_map = {i["traj_id"]: i for i in inputs}

    all_errors = []

    for pred in predictions:
        tid = pred["traj_id"]
        gt = gt_map[tid]
        inp = inp_map[tid]

        pred_coords = np.array(pred["coords"], dtype=np.float64)
        gt_coords = np.array(gt["coords"], dtype=np.float64)
        mask = np.array(inp["mask"], dtype=bool)

        missing = ~mask
        if not np.any(missing):
            continue

        dists = haversine_batch(pred_coords[missing], gt_coords[missing])
        all_errors.extend(dists.tolist())

    errors = np.array(all_errors)
    return {
        "mae_meter": float(np.mean(errors)),
        "rmse_meter": float(np.sqrt(np.mean(errors ** 2))),
        "n_points": len(errors),
    }


def check_known_points_preserved(predictions: list[dict], inputs: list[dict],
                                  tol: float = 1e-4) -> list[str]:
    """Verify that known (mask=True) points are unchanged."""
    inp_map = {i["traj_id"]: i for i in inputs}
    warnings = []
    for pred in predictions:
        tid = pred["traj_id"]
        inp = inp_map[tid]
        mask = np.array(inp["mask"], dtype=bool)
        orig = np.array(inp["coords"], dtype=np.float64)[mask]
        pred_known = np.array(pred["coords"], dtype=np.float64)[mask]
        diff = np.abs(orig - pred_known).max() if len(orig) > 0 else 0
        if diff > tol:
            warnings.append(f"traj_id={tid}: known point deviation {diff:.6f}")
    return warnings
