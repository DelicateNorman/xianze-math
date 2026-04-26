"""Task A prediction pipeline."""
from __future__ import annotations
import numpy as np
from tqdm import tqdm
from src.task_a.methods import (
    linear_time_interpolation,
    linear_with_speed_smoothing,
    knn_template_refinement,
)


def predict_task_a(inputs: list[dict], method: str = "linear_with_speed_smoothing",
                   config: dict | None = None,
                   train_trajectories: list[np.ndarray] | None = None) -> list[dict]:
    """
    Run Task A prediction for all input trajectories.
    Returns list of {'traj_id': ..., 'coords': np.ndarray}.
    """
    config = config or {}
    predictions = []

    for item in tqdm(inputs, desc=f"Task A [{method}]"):
        if method == "linear_time_interpolation":
            coords = linear_time_interpolation(item)

        elif method == "linear_with_speed_smoothing":
            sp_cfg = config.get("speed_smoothing", {})
            coords = linear_with_speed_smoothing(
                item,
                max_speed_kmh=sp_cfg.get("max_speed_kmh", 120.0),
                smoothing_window=sp_cfg.get("smoothing_window", 3),
                iterations=sp_cfg.get("iterations", 2),
            )

        elif method == "knn_template_refinement":
            knn_cfg = config.get("knn_template", {})
            n_known = int(np.sum(item["mask"]))
            n_total = len(item["mask"])
            keep_rate = n_known / n_total
            # Choose alpha based on keep rate
            alpha = knn_cfg.get("alpha_8", 0.3) if keep_rate > 0.1 else knn_cfg.get("alpha_16", 0.2)
            coords = knn_template_refinement(
                item,
                train_trajectories=train_trajectories or [],
                alpha=alpha,
                start_thresh_km=knn_cfg.get("start_dist_thresh_km", 1.0),
                end_thresh_km=knn_cfg.get("end_dist_thresh_km", 1.0),
                top_k=knn_cfg.get("top_k", 5),
                max_candidates=knn_cfg.get("max_candidates", 200),
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        predictions.append({"traj_id": item["traj_id"], "coords": coords})

    return predictions
