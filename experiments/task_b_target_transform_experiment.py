"""Task B experiment: alternative residual target transforms.

The committed model predicts absolute residual seconds after a point-count
median baseline. This script tests whether the residual is easier to learn as
per-segment interval, relative interval, or square-root normalized residual.
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
    nseg_train = np.maximum(np.array([len(item["coords"]) - 1 for item in train_data], dtype=np.float64), 1.0)
    nseg_val = np.maximum(np.array([len(item["coords"]) - 1 for item in val_items], dtype=np.float64), 1.0)

    table, interval = fit_count_baseline(train_data)
    base_train = count_baseline(train_data, table, interval)
    base_val = count_baseline(val_items, table, interval)
    residual_train = y_train - base_train

    print("Building enhanced features...", flush=True)
    X_train, names = build_enhanced_feature_matrix(train_data, base_train)
    X_val, _ = build_enhanced_feature_matrix(val_items, base_val)
    print(f"X_train={X_train.shape}, X_val={X_val.shape}, features={len(names)}", flush=True)

    transforms = {
        "absolute_residual": (
            residual_train,
            lambda pred: base_val + pred,
        ),
        "residual_per_segment": (
            residual_train / nseg_train,
            lambda pred: base_val + pred * nseg_val,
        ),
        "residual_per_sqrt_segment": (
            residual_train / np.sqrt(nseg_train),
            lambda pred: base_val + pred * np.sqrt(nseg_val),
        ),
        "interval_direct": (
            y_train / nseg_train,
            lambda pred: pred * nseg_val,
        ),
        "log_interval_direct": (
            np.log1p(y_train / nseg_train),
            lambda pred: np.expm1(pred) * nseg_val,
        ),
        "relative_to_baseline": (
            y_train / np.maximum(base_train, 1.0) - 1.0,
            lambda pred: base_val * (1.0 + pred),
        ),
    }

    results: list[Result] = [metric("count_median_baseline", base_val, y_val)]
    preds: dict[str, np.ndarray] = {}
    for name, (target, inverse) in transforms.items():
        for seed, depth, leaves in ((42, 8, 63), (7, 6, 31)):
            model_name = f"{name}_hgb_d{depth}_s{seed}"
            print(f"Fitting {model_name}...", flush=True)
            model = hgb(seed=seed, depth=depth, leaves=leaves)
            model.fit(X_train, target)
            pred = np.maximum(inverse(model.predict(X_val)), 30.0)
            preds[model_name] = pred
            result = metric(model_name, pred, y_val)
            results.append(result)
            print(f"{result.name:42s} MAE={result.mae:8.4f} RMSE={result.rmse:8.4f}", flush=True)

    names_sorted = [item.name for item in sorted(results, key=lambda item: item.mae) if item.name in preds]
    print("Searching simple blends among top transforms...", flush=True)
    top = names_sorted[:8]
    for i, left in enumerate(top):
        for right in top[i + 1:]:
            for w in (0.25, 0.5, 0.75):
                pred = w * preds[left] + (1.0 - w) * preds[right]
                results.append(metric(f"blend_{w:.2f}_{left}__{right}", pred, y_val))

    print("\nResults sorted by MAE:", flush=True)
    for result in sorted(results, key=lambda item: item.mae)[:30]:
        print(
            f"{result.name:90s} MAE={result.mae:8.4f} RMSE={result.rmse:8.4f}",
            flush=True,
        )
    print(f"Elapsed: {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
