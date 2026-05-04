"""Task A prediction pipeline."""
from __future__ import annotations
import numpy as np
from tqdm import tqdm
from src.task_a.methods import (
    linear_time_interpolation,
    linear_with_speed_smoothing,
    catmull_rom_interpolation,
    pchip_time_interpolation,
    knn_template_refinement,
    build_template_index,
    build_local_segment_index,
    local_segment_template_interpolation,
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

    # Build KDTree index once for KNN method
    _knn_index = None
    if method == "knn_template_refinement" and train_trajectories:
        _knn_index = build_template_index(train_trajectories)

    _segment_index = None
    if method == "local_segment_template_interpolation" and train_trajectories:
        segment_cfg = config.get("local_segment_template", {})
        configured_spans = segment_cfg.get("spans")
        if configured_spans:
            input_spans = {
                int(right - left)
                for item in inputs
                for left, right in zip(np.where(item["mask"])[0][:-1], np.where(item["mask"])[0][1:])
            }
            spans = sorted({int(span) for span in configured_spans if int(span) in input_spans})
        else:
            spans = sorted({
                int(right - left)
                for item in inputs
                for left, right in zip(np.where(item["mask"])[0][:-1], np.where(item["mask"])[0][1:])
                if int(right - left) >= 2
            })
        _segment_index = build_local_segment_index(
            train_trajectories,
            spans=spans,
            max_segments_per_span=segment_cfg.get("max_segments_per_span", 250_000),
            samples_per_traj_span=segment_cfg.get("samples_per_traj_span", 3),
            min_displacement_m=segment_cfg.get("min_displacement_m", 20.0),
            seed=config.get("seed", 42),
        )

    for item in tqdm(inputs, desc=f"Task A [{method}]"):
        if method == "linear_time_interpolation":
            coords = linear_time_interpolation(item)

        elif method == "catmull_rom_interpolation":
            coords = catmull_rom_interpolation(item)

        elif method == "pchip_time_interpolation":
            coords = pchip_time_interpolation(item)

        elif method == "local_segment_template_interpolation":
            segment_cfg = config.get("local_segment_template", {})
            coords = local_segment_template_interpolation(
                item,
                segment_index=_segment_index,
                alpha=segment_cfg.get("alpha", 1.0),
                top_k=segment_cfg.get("top_k", 20),
                max_feature_distance=segment_cfg.get("max_feature_distance", 2.5),
                confidence_blend=segment_cfg.get("confidence_blend", "none"),
                confidence_threshold=segment_cfg.get("confidence_threshold", 1.2),
                confidence_scale=segment_cfg.get("confidence_scale", 1.0),
            )

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
            alpha = knn_cfg.get("alpha_8", 0.3) if keep_rate > 0.1 else knn_cfg.get("alpha_16", 0.2)
            coords = knn_template_refinement(
                item,
                train_trajectories=train_trajectories or [],
                alpha=alpha,
                start_thresh_km=knn_cfg.get("start_dist_thresh_km", 1.0),
                end_thresh_km=knn_cfg.get("end_dist_thresh_km", 1.0),
                top_k=knn_cfg.get("top_k", 5),
                max_candidates=knn_cfg.get("max_candidates", 200),
                index=_knn_index,
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        predictions.append({"traj_id": item["traj_id"], "coords": coords})

    return predictions
