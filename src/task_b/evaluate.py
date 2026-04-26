"""Task B evaluation: MAE, RMSE, MAPE."""
from __future__ import annotations
import numpy as np


def evaluate_task_b(predictions: list[dict], ground_truths: list[dict]) -> dict:
    """
    predictions: list of {'traj_id': ..., 'travel_time': float}
    ground_truths: list of {'traj_id': ..., 'travel_time': int/float}
    """
    gt_map = {g["traj_id"]: g["travel_time"] for g in ground_truths}

    pred_vals, true_vals = [], []
    for pred in predictions:
        tid = pred["traj_id"]
        if tid not in gt_map:
            continue
        pred_vals.append(pred["travel_time"])
        true_vals.append(gt_map[tid])

    pred_arr = np.array(pred_vals, dtype=np.float64)
    true_arr = np.array(true_vals, dtype=np.float64)

    abs_err = np.abs(pred_arr - true_arr)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    # MAPE: avoid division by zero
    nonzero = true_arr > 0
    mape = float(np.mean(abs_err[nonzero] / true_arr[nonzero]) * 100)

    return {
        "mae_second": mae,
        "rmse_second": rmse,
        "mape_percent": mape,
        "n_samples": len(pred_vals),
    }
