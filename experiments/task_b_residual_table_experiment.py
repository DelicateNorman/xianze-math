"""Task B experiment: residual calibration tables.

Tests whether the point-count median baseline can be improved with compact
median residual tables keyed by timestamp phase and simple route buckets.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.task_b.dataset import load_task_b_gt, load_task_b_input, load_train_data
from src.task_b.features import _grid_xy


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


def key_for(item: dict, spec: tuple[str, ...]) -> tuple:
    ts = int(item["departure_timestamp"])
    coords = item["coords"]
    start = coords[0]
    end = coords[-1]
    fields = []
    for part in spec:
        if part == "n":
            fields.append(len(coords))
        elif part.startswith("mod"):
            mod = int(part[3:])
            fields.append(ts % mod)
        elif part == "minute":
            fields.append((ts % 3600) // 60)
        elif part == "hour":
            fields.append((ts % 86400) // 3600)
        elif part == "start_grid":
            fields.append(_grid_xy(float(start[0]), float(start[1])))
        elif part == "end_grid":
            fields.append(_grid_xy(float(end[0]), float(end[1])))
        elif part == "route_grid":
            fields.append((_grid_xy(float(start[0]), float(start[1])), _grid_xy(float(end[0]), float(end[1]))))
        else:
            raise ValueError(part)
    return tuple(fields)


def fit_residual_table(
    items: list[dict],
    residuals: np.ndarray,
    spec: tuple[str, ...],
    min_samples: int,
) -> dict[tuple, float]:
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for item, residual in zip(items, residuals):
        buckets[key_for(item, spec)].append(float(residual))
    return {
        key: float(np.median(values))
        for key, values in buckets.items()
        if len(values) >= min_samples
    }


def apply_table(
    items: list[dict],
    table: dict[tuple, float],
    spec: tuple[str, ...],
) -> np.ndarray:
    return np.array([table.get(key_for(item, spec), 0.0) for item in items], dtype=np.float64)


def main() -> None:
    started = time.time()
    print("Loading data...", flush=True)
    train_data = load_train_data(TRAIN_PATH)
    val_items = load_task_b_input(VAL_INPUT_PATH)
    val_gt = load_task_b_gt(VAL_GT_PATH)
    y_train = np.array([item["travel_time"] for item in train_data], dtype=np.float64)
    y_map = {item["traj_id"]: item["travel_time"] for item in val_gt}
    y_val = np.array([y_map[item["traj_id"]] for item in val_items], dtype=np.float64)

    count_table, interval = fit_count_baseline(train_data)
    base_train = count_baseline(train_data, count_table, interval)
    base_val = count_baseline(val_items, count_table, interval)
    residual_train = y_train - base_train

    results = [metric("count_median_baseline", base_val, y_val)]
    specs = [
        ("n", "mod2"),
        ("n", "mod3"),
        ("n", "mod5"),
        ("n", "mod10"),
        ("n", "mod15"),
        ("n", "mod30"),
        ("n", "minute"),
        ("n", "hour"),
        ("n", "mod15", "hour"),
        ("n", "mod30", "hour"),
        ("n", "start_grid"),
        ("n", "end_grid"),
        ("n", "route_grid"),
        ("n", "mod15", "route_grid"),
    ]
    for spec in specs:
        for min_samples in (3, 5, 10, 20, 50):
            table = fit_residual_table(train_data, residual_train, spec, min_samples)
            corr = apply_table(val_items, table, spec)
            covered = float(np.mean(corr != 0.0))
            name = "+".join(spec) + f"_m{min_samples}"
            result = metric(name, base_val + corr, y_val, notes=f"buckets={len(table)} covered={covered:.3f}")
            results.append(result)
            print(
                f"{result.name:38s} MAE={result.mae:8.4f} RMSE={result.rmse:8.4f} {result.notes}",
                flush=True,
            )

    print("\nResults sorted by MAE:", flush=True)
    for result in sorted(results, key=lambda item: item.mae)[:20]:
        print(
            f"{result.name:38s} MAE={result.mae:8.4f} RMSE={result.rmse:8.4f} {result.notes}",
            flush=True,
        )
    print(f"Elapsed: {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
