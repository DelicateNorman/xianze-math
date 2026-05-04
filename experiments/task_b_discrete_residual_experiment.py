"""Task B experiment: discrete residual classification.

This is an exploratory script, not part of the submission pipeline. It tests
whether the residual left after the point-count median baseline is better
treated as a discrete offset class than as a continuous regression target.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
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
    notes: str


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


def evaluate(name: str, pred: np.ndarray, y: np.ndarray, notes: str = "") -> Result:
    return Result(
        name=name,
        mae=float(mean_absolute_error(y, pred)),
        rmse=float(mean_squared_error(y, pred) ** 0.5),
        notes=notes,
    )


def print_result(result: Result) -> None:
    print(
        f"{result.name:34s} MAE={result.mae:8.4f} RMSE={result.rmse:8.4f} {result.notes}",
        flush=True,
    )


def main() -> None:
    started = time.time()
    print("Loading data...", flush=True)
    train_data = load_train_data(TRAIN_PATH)
    val_items = load_task_b_input(VAL_INPUT_PATH)
    val_gt = load_task_b_gt(VAL_GT_PATH)
    y_train = np.array([item["travel_time"] for item in train_data], dtype=np.float64)
    y_val_map = {item["traj_id"]: item["travel_time"] for item in val_gt}
    y_val = np.array([y_val_map[item["traj_id"]] for item in val_items], dtype=np.float64)

    table, interval = fit_count_baseline(train_data)
    base_train = count_baseline(train_data, table, interval)
    base_val = count_baseline(val_items, table, interval)
    residual_train = y_train - base_train

    print("Building enhanced features...", flush=True)
    X_train, feature_names = build_enhanced_feature_matrix(train_data, base_train)
    X_val, _ = build_enhanced_feature_matrix(val_items, base_val)
    print(f"X_train={X_train.shape}, X_val={X_val.shape}, features={len(feature_names)}", flush=True)

    results: list[Result] = []
    results.append(evaluate("count_median_baseline", base_val, y_val))

    print("Fitting continuous HGB residual baseline...", flush=True)
    reg = HistGradientBoostingRegressor(
        max_iter=1000,
        max_leaf_nodes=63,
        max_depth=8,
        learning_rate=0.03,
        l2_regularization=0.02,
        random_state=42,
    )
    reg.fit(X_train, residual_train)
    results.append(evaluate("hgb_residual_regression", base_val + reg.predict(X_val), y_val))

    for step in (3, 5, 10, 15, 30, 60):
        labels = np.rint(residual_train / step).astype(np.int16)
        classes = np.unique(labels)
        print(f"Fitting classifier step={step}, classes={len(classes)}...", flush=True)
        clf = HistGradientBoostingClassifier(
            max_iter=500,
            max_leaf_nodes=31,
            max_depth=6,
            learning_rate=0.04,
            l2_regularization=0.03,
            early_stopping=False,
            random_state=42,
        )
        clf.fit(X_train, labels)

        hard_residual = clf.predict(X_val).astype(np.float64) * step
        results.append(evaluate(
            f"discrete_hard_step_{step}",
            base_val + hard_residual,
            y_val,
            notes=f"classes={len(classes)}",
        ))

        prob = clf.predict_proba(X_val)
        expected_residual = prob @ clf.classes_.astype(np.float64) * step
        results.append(evaluate(
            f"discrete_expected_step_{step}",
            base_val + expected_residual,
            y_val,
            notes=f"classes={len(classes)}",
        ))

        blend = 0.7 * reg.predict(X_val) + 0.3 * expected_residual
        results.append(evaluate(
            f"reg_expected_blend_step_{step}",
            base_val + blend,
            y_val,
            notes="0.7 reg / 0.3 expected",
        ))

    print("\nResults sorted by MAE:", flush=True)
    for result in sorted(results, key=lambda item: item.mae):
        print_result(result)
    print(f"Elapsed: {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
