"""Task B prediction pipeline."""
from __future__ import annotations
import numpy as np
from src.task_b.models import GlobalSpeedModel, TimeBucketSpeedModel, EnsembleModel


def predict_task_b(items: list[dict], model) -> list[dict]:
    """Run Task B prediction. model must have a .predict(items) method."""
    preds_arr = model.predict(items)
    return [
        {"traj_id": item["traj_id"], "travel_time": float(p)}
        for item, p in zip(items, preds_arr)
    ]
