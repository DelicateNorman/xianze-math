"""Task B experiment: blend committed ensemble with KNN residual correction."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from src.common.io import load_pkl
from src.task_b.dataset import load_task_b_gt, load_task_b_input, load_train_data
from src.task_b.features import build_enhanced_feature_matrix


TRAIN_PATH = "data/data_ds15/train.pkl"
VAL_INPUT_PATH = "data/task_B_tte/val_input.pkl"
VAL_GT_PATH = "data/task_B_tte/val_gt.pkl"
BEST_PRED_PATH = "outputs/submissions/task_b_val_sampling_residual_ensemble.pkl"


@dataclass
class Result:
    name: str
    mae: float
    rmse: float


def metric(name: str, pred: np.ndarray, y: np.ndarray) -> Result:
    return Result(
        name,
        float(mean_absolute_error(y, pred)),
        float(mean_squared_error(y, pred) ** 0.5),
    )


def fit_count_baseline(train_data: list[dict]) -> tuple[dict[int, float], float]:
    n_points = np.array([len(item["coords"]) for item in train_data], dtype=np.int64)
    targets = np.array([item["travel_time"] for item in train_data], dtype=np.float64)
    interval = float(np.median(targets / np.maximum(n_points - 1, 1)))
    table = {
        int(n): float(np.median(targets[n_points == n]))
        for n in sorted(set(n_points.tolist()))
    }
    return table, interval


def count_baseline(items: list[dict], table: dict[int, float], interval: float) -> np.ndarray:
    return np.array([
        table.get(len(item["coords"]), (len(item["coords"]) - 1) * interval)
        for item in items
    ], dtype=np.float64)


def idw(dist: np.ndarray, residual: np.ndarray, power: float) -> np.ndarray:
    weights = 1.0 / np.maximum(dist, 1e-6) ** power
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return np.sum(weights * residual, axis=1)


def print_result(result: Result) -> None:
    print(f"{result.name:34s} MAE={result.mae:8.4f} RMSE={result.rmse:8.4f}", flush=True)


def main() -> None:
    started = time.time()
    print("Loading data and current best predictions...", flush=True)
    train_data = load_train_data(TRAIN_PATH)
    val_items = load_task_b_input(VAL_INPUT_PATH)
    val_gt = load_task_b_gt(VAL_GT_PATH)
    best_pred_items = load_pkl(BEST_PRED_PATH)
    y_train = np.array([item["travel_time"] for item in train_data], dtype=np.float64)
    y_map = {item["traj_id"]: item["travel_time"] for item in val_gt}
    best_map = {item["traj_id"]: item["travel_time"] for item in best_pred_items}
    y_val = np.array([y_map[item["traj_id"]] for item in val_items], dtype=np.float64)
    best_pred = np.array([best_map[item["traj_id"]] for item in val_items], dtype=np.float64)

    table, interval = fit_count_baseline(train_data)
    base_train = count_baseline(train_data, table, interval)
    base_val = count_baseline(val_items, table, interval)
    residual_train = y_train - base_train

    print("Building enhanced features and nearest neighbors...", flush=True)
    X_train, _ = build_enhanced_feature_matrix(train_data, base_train)
    X_val, _ = build_enhanced_feature_matrix(val_items, base_val)
    scaler = make_pipeline(RobustScaler())
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    nn = NearestNeighbors(n_neighbors=233, algorithm="auto", n_jobs=-1)
    nn.fit(Xtr)
    dist, idx = nn.kneighbors(Xva, return_distance=True)

    results = [metric("current_best", best_pred, y_val)]
    for k in (34, 55, 89, 144, 233):
        residual = idw(dist[:, :k], residual_train[idx[:, :k]], power=2.0)
        knn_pred = np.maximum(base_val + residual, 30.0)
        results.append(metric(f"knn_k{k}_idw2", knn_pred, y_val))
        for w in np.linspace(0.0, 0.30, 16):
            blended = (1.0 - w) * best_pred + w * knn_pred
            results.append(metric(f"blend_knn{k}_w{w:.2f}", blended, y_val))

    print("\nResults sorted by MAE:", flush=True)
    for result in sorted(results, key=lambda item: item.mae)[:30]:
        print_result(result)
    print(f"Elapsed: {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
