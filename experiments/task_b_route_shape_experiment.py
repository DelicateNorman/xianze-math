"""Task B experiment: fixed route-shape features.

The production feature set has strong aggregate geometry but little direct
representation of the whole route shape. This script adds resampled coordinate
landmarks and checks whether tree residual models can use route-specific shape
information better than coarse start/end/grid features.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.task_b.dataset import load_task_b_gt, load_task_b_input, load_train_data
from src.task_b.features import build_enhanced_feature_matrix


TRAIN_PATH = "data/data_ds15/train.pkl"
VAL_INPUT_PATH = "data/task_B_tte/val_input.pkl"
VAL_GT_PATH = "data/task_B_tte/val_gt.pkl"


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


def resample_by_index(coords: np.ndarray, points: int) -> np.ndarray:
    if len(coords) == points:
        return coords.astype(np.float64, copy=False)
    source = np.arange(len(coords), dtype=np.float64)
    target = np.linspace(0.0, len(coords) - 1.0, points)
    lon = np.interp(target, source, coords[:, 0])
    lat = np.interp(target, source, coords[:, 1])
    return np.column_stack([lon, lat])


def route_shape_matrix(items: list[dict], points: int) -> tuple[np.ndarray, list[str]]:
    rows = []
    for item in items:
        coords = np.asarray(item["coords"], dtype=np.float64)
        sampled = resample_by_index(coords, points)
        start = coords[0]
        end = coords[-1]
        line = start + np.linspace(0.0, 1.0, points)[:, None] * (end - start)
        rel_start = sampled - start
        rel_line = sampled - line
        rows.append(np.concatenate([
            sampled.reshape(-1),
            rel_start.reshape(-1),
            rel_line.reshape(-1),
        ]))
    names = []
    for prefix in ("shape_abs", "shape_rel_start", "shape_rel_line"):
        for idx in range(points):
            names.extend([f"{prefix}_{idx}_lon", f"{prefix}_{idx}_lat"])
    return np.asarray(rows, dtype=np.float64), names


def hgb(seed: int, depth: int, leaves: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=1000,
        max_leaf_nodes=leaves,
        max_depth=depth,
        learning_rate=0.03,
        l2_regularization=0.02,
        random_state=seed,
    )


def main() -> None:
    started = time.time()
    print("Loading data...", flush=True)
    train_data = load_train_data(TRAIN_PATH)
    val_items = load_task_b_input(VAL_INPUT_PATH)
    val_gt = load_task_b_gt(VAL_GT_PATH)
    y_train = np.array([item["travel_time"] for item in train_data], dtype=np.float64)
    y_map = {item["traj_id"]: item["travel_time"] for item in val_gt}
    y_val = np.array([y_map[item["traj_id"]] for item in val_items], dtype=np.float64)

    table, interval = fit_count_baseline(train_data)
    base_train = count_baseline(train_data, table, interval)
    base_val = count_baseline(val_items, table, interval)
    residual_train = y_train - base_train

    print("Building enhanced features...", flush=True)
    X_base_train, base_names = build_enhanced_feature_matrix(train_data, base_train)
    X_base_val, _ = build_enhanced_feature_matrix(val_items, base_val)
    results = [metric("count_median_baseline", base_val, y_val)]

    for points in (8, 16, 24, 32):
        print(f"Building route shape features points={points}...", flush=True)
        X_shape_train, shape_names = route_shape_matrix(train_data, points)
        X_shape_val, _ = route_shape_matrix(val_items, points)
        X_train = np.column_stack([X_base_train, X_shape_train])
        X_val = np.column_stack([X_base_val, X_shape_val])
        print(
            f"points={points}, X_train={X_train.shape}, features={len(base_names) + len(shape_names)}",
            flush=True,
        )
        for name, seed, depth, leaves in (
            ("hgb_d8", 42, 8, 63),
            ("hgb_d6", 7, 6, 31),
        ):
            model_name = f"shape{points}_{name}"
            print(f"Fitting {model_name}...", flush=True)
            model = hgb(seed=seed, depth=depth, leaves=leaves)
            model.fit(X_train, residual_train)
            pred = np.maximum(base_val + model.predict(X_val), 30.0)
            result = metric(model_name, pred, y_val)
            results.append(result)
            print(f"{result.name:20s} MAE={result.mae:8.4f} RMSE={result.rmse:8.4f}", flush=True)

    print("\nResults sorted by MAE:", flush=True)
    for result in sorted(results, key=lambda item: item.mae):
        print(f"{result.name:20s} MAE={result.mae:8.4f} RMSE={result.rmse:8.4f}", flush=True)
    print(f"Elapsed: {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
