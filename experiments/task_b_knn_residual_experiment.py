"""Task B experiment: nearest-neighbor residual correction.

Uses the current point-count median baseline, then predicts the residual from
nearby training samples in an enhanced feature space. This tests whether local
memorization of sampling phase and route geometry beats tree residual models.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

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
    notes: str = ""


def metric(name: str, pred: np.ndarray, y: np.ndarray, notes: str = "") -> Result:
    return Result(
        name,
        float(mean_absolute_error(y, pred)),
        float(mean_squared_error(y, pred) ** 0.5),
        notes,
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


def weighted_residual(dist: np.ndarray, residual: np.ndarray, power: float) -> np.ndarray:
    weights = 1.0 / np.maximum(dist, 1e-6) ** power
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return np.sum(weights * residual, axis=1)


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
    X_train, names = build_enhanced_feature_matrix(train_data, base_train)
    X_val, _ = build_enhanced_feature_matrix(val_items, base_val)
    print(f"X_train={X_train.shape}, X_val={X_val.shape}, features={len(names)}", flush=True)

    feature_sets = {
        "all": np.arange(X_train.shape[1]),
        "sampling_tail": np.arange(max(0, X_train.shape[1] - 61), X_train.shape[1]),
        "geo_plus_sampling": np.r_[
            0:13,
            14:29,
            max(0, X_train.shape[1] - 61):X_train.shape[1],
        ],
    }
    results = [metric("count_median_baseline", base_val, y_val)]

    for scaler_name, scaler in {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
    }.items():
        for feature_name, columns in feature_sets.items():
            print(f"Scaling {scaler_name}/{feature_name}...", flush=True)
            pipe = make_pipeline(scaler)
            Xtr = pipe.fit_transform(X_train[:, columns])
            Xva = pipe.transform(X_val[:, columns])
            max_k = 256
            nn = NearestNeighbors(n_neighbors=max_k, algorithm="auto", n_jobs=-1)
            print(f"Fitting/querying NN {scaler_name}/{feature_name}...", flush=True)
            nn.fit(Xtr)
            dist, idx = nn.kneighbors(Xva, return_distance=True)
            for k in (3, 5, 8, 13, 21, 34, 55, 89, 144, 233):
                d = dist[:, :k]
                r = residual_train[idx[:, :k]]
                for reducer in ("median", "mean"):
                    if reducer == "median":
                        residual = np.median(r, axis=1)
                    else:
                        residual = np.mean(r, axis=1)
                    result = metric(
                        f"{scaler_name}_{feature_name}_k{k}_{reducer}",
                        base_val + residual,
                        y_val,
                    )
                    results.append(result)
                    print(
                        f"{result.name:44s} MAE={result.mae:8.4f} RMSE={result.rmse:8.4f}",
                        flush=True,
                    )
                for power in (0.5, 1.0, 2.0):
                    residual = weighted_residual(d, r, power)
                    result = metric(
                        f"{scaler_name}_{feature_name}_k{k}_idw{power}",
                        base_val + residual,
                        y_val,
                    )
                    results.append(result)
                    print(
                        f"{result.name:44s} MAE={result.mae:8.4f} RMSE={result.rmse:8.4f}",
                        flush=True,
                    )

    print("\nResults sorted by MAE:", flush=True)
    for result in sorted(results, key=lambda item: item.mae)[:25]:
        print(
            f"{result.name:44s} MAE={result.mae:8.4f} RMSE={result.rmse:8.4f} {result.notes}",
            flush=True,
        )
    print(f"Elapsed: {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
