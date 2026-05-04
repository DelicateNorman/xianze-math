"""Task B experiment: train-only historical speed priors.

Builds median speed tables from training trajectories with timestamps, then
uses only Task B input coords + departure_timestamp at prediction time to
produce route-level travel-time priors. These priors are added as features for
residual models and also tested as a light blend with the committed best output.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.common.geo import bearing_batch, haversine_batch
from src.common.io import load_pkl
from src.task_b.dataset import load_task_b_gt, load_task_b_input, load_train_data
from src.task_b.features import (
    GRID_LAT_MAX,
    GRID_LAT_MIN,
    GRID_LON_MAX,
    GRID_LON_MIN,
    build_enhanced_feature_matrix,
)


TRAIN_PATH = "data/data_ds15/train.pkl"
VAL_INPUT_PATH = "data/task_B_tte/val_input.pkl"
VAL_GT_PATH = "data/task_B_tte/val_gt.pkl"
BEST_PRED_PATH = "outputs/submissions/task_b_val_sampling_residual_ensemble.pkl"

SPEED_GRID = 30
DIR_BINS = 8


@dataclass
class Result:
    name: str
    mae: float
    rmse: float
    notes: str = ""


class SpeedPrior:
    def __init__(self) -> None:
        self.global_speed = 10.0
        self.hour_speed: dict[int, float] = {}
        self.cell_speed: dict[int, float] = {}
        self.cell_hour_speed: dict[tuple[int, int], float] = {}
        self.cell_hour_dir_speed: dict[tuple[int, int, int], float] = {}

    @staticmethod
    def cell_id(lon: float, lat: float) -> int:
        gx = int((lon - GRID_LON_MIN) / (GRID_LON_MAX - GRID_LON_MIN) * SPEED_GRID)
        gy = int((lat - GRID_LAT_MIN) / (GRID_LAT_MAX - GRID_LAT_MIN) * SPEED_GRID)
        gx = max(0, min(SPEED_GRID - 1, gx))
        gy = max(0, min(SPEED_GRID - 1, gy))
        return gy * SPEED_GRID + gx

    @staticmethod
    def dir_id(bearing: float) -> int:
        return int(((bearing + 360.0) % 360.0) / 360.0 * DIR_BINS) % DIR_BINS

    def fit(self, raw_train: list[dict]) -> "SpeedPrior":
        global_values: list[float] = []
        hour_values: dict[int, list[float]] = defaultdict(list)
        cell_values: dict[int, list[float]] = defaultdict(list)
        cell_hour_values: dict[tuple[int, int], list[float]] = defaultdict(list)
        cell_hour_dir_values: dict[tuple[int, int, int], list[float]] = defaultdict(list)

        for idx, item in enumerate(raw_train):
            if idx % 20_000 == 0:
                print(f"  speed prior fit rows={idx}/{len(raw_train)}", flush=True)
            coords = np.asarray(item["coords"], dtype=np.float64)
            ts = np.asarray(item["timestamps"], dtype=np.int64)
            if len(coords) < 2 or len(ts) != len(coords):
                continue
            dist = haversine_batch(coords[:-1], coords[1:])
            dt = np.diff(ts).astype(np.float64)
            valid = (dt > 0) & (dist > 1.0)
            if not np.any(valid):
                continue
            mids = (coords[:-1] + coords[1:]) * 0.5
            bearings = bearing_batch(coords)
            speeds = dist / np.maximum(dt, 1.0)
            for mid, t0, bearing, speed, ok in zip(mids, ts[:-1], bearings, speeds, valid):
                if not ok or speed < 0.5 or speed > 45.0:
                    continue
                speed_f = float(speed)
                hour = int((int(t0) % 86_400) // 3600)
                cell = self.cell_id(float(mid[0]), float(mid[1]))
                direction = self.dir_id(float(bearing))
                global_values.append(speed_f)
                hour_values[hour].append(speed_f)
                cell_values[cell].append(speed_f)
                cell_hour_values[(cell, hour)].append(speed_f)
                cell_hour_dir_values[(cell, hour, direction)].append(speed_f)

        self.global_speed = float(np.median(global_values))
        self.hour_speed = {
            key: float(np.median(values))
            for key, values in hour_values.items()
            if len(values) >= 200
        }
        self.cell_speed = {
            key: float(np.median(values))
            for key, values in cell_values.items()
            if len(values) >= 100
        }
        self.cell_hour_speed = {
            key: float(np.median(values))
            for key, values in cell_hour_values.items()
            if len(values) >= 50
        }
        self.cell_hour_dir_speed = {
            key: float(np.median(values))
            for key, values in cell_hour_dir_values.items()
            if len(values) >= 25
        }
        print(
            "  tables:"
            f" global={self.global_speed:.3f}"
            f" hour={len(self.hour_speed)}"
            f" cell={len(self.cell_speed)}"
            f" cell_hour={len(self.cell_hour_speed)}"
            f" cell_hour_dir={len(self.cell_hour_dir_speed)}",
            flush=True,
        )
        return self

    def predict_one(self, item: dict, baseline_time: float) -> tuple[float, float, float, float, float]:
        coords = np.asarray(item["coords"], dtype=np.float64)
        if len(coords) < 2:
            return baseline_time, 0.0, 0.0, 0.0, self.global_speed
        dist = haversine_batch(coords[:-1], coords[1:])
        mids = (coords[:-1] + coords[1:]) * 0.5
        bearings = bearing_batch(coords)
        cum_frac = np.r_[0.0, np.cumsum(dist[:-1]) / max(float(np.sum(dist)), 1.0)]
        departure = int(item["departure_timestamp"])

        pred = 0.0
        used_dir = 0
        used_cell_hour = 0
        used_cell = 0
        speeds = []
        for seg_dist, mid, bearing, frac in zip(dist, mids, bearings, cum_frac):
            approx_ts = departure + int(frac * baseline_time)
            hour = int((approx_ts % 86_400) // 3600)
            cell = self.cell_id(float(mid[0]), float(mid[1]))
            direction = self.dir_id(float(bearing))
            speed = self.cell_hour_dir_speed.get((cell, hour, direction))
            if speed is not None:
                used_dir += 1
            else:
                speed = self.cell_hour_speed.get((cell, hour))
                if speed is not None:
                    used_cell_hour += 1
                else:
                    speed = self.cell_speed.get(cell)
                    if speed is not None:
                        used_cell += 1
                    else:
                        speed = self.hour_speed.get(hour, self.global_speed)
            speeds.append(speed)
            pred += float(seg_dist) / max(float(speed), 0.5)

        nseg = max(len(dist), 1)
        return (
            pred,
            used_dir / nseg,
            used_cell_hour / nseg,
            used_cell / nseg,
            float(np.mean(speeds)),
        )


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


def speed_feature_matrix(items: list[dict], baseline: np.ndarray, prior: SpeedPrior) -> tuple[np.ndarray, list[str]]:
    rows = []
    for idx, (item, base) in enumerate(zip(items, baseline)):
        if idx % 20_000 == 0:
            print(f"  speed features rows={idx}/{len(items)}", flush=True)
        prior_time, dir_cov, cell_hour_cov, cell_cov, mean_speed = prior.predict_one(item, float(base))
        rows.append([
            prior_time,
            prior_time - float(base),
            prior_time / max(float(base), 1.0),
            dir_cov,
            cell_hour_cov,
            cell_cov,
            mean_speed,
        ])
    names = [
        "speed_prior_time",
        "speed_prior_minus_baseline",
        "speed_prior_ratio",
        "speed_prior_dir_coverage",
        "speed_prior_cell_hour_coverage",
        "speed_prior_cell_coverage",
        "speed_prior_mean_speed",
    ]
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


def print_result(result: Result) -> None:
    print(f"{result.name:42s} MAE={result.mae:8.4f} RMSE={result.rmse:8.4f} {result.notes}", flush=True)


def main() -> None:
    started = time.time()
    print("Loading data...", flush=True)
    with open(TRAIN_PATH, "rb") as file:
        raw_train = load_pkl(TRAIN_PATH)
    train_data = load_train_data(TRAIN_PATH)
    val_items = load_task_b_input(VAL_INPUT_PATH)
    val_gt = load_task_b_gt(VAL_GT_PATH)
    best_items = load_pkl(BEST_PRED_PATH)
    y_train = np.array([item["travel_time"] for item in train_data], dtype=np.float64)
    y_map = {item["traj_id"]: item["travel_time"] for item in val_gt}
    best_map = {item["traj_id"]: item["travel_time"] for item in best_items}
    y_val = np.array([y_map[item["traj_id"]] for item in val_items], dtype=np.float64)
    best_pred = np.array([best_map[item["traj_id"]] for item in val_items], dtype=np.float64)

    table, interval = fit_count_baseline(train_data)
    base_train = count_baseline(train_data, table, interval)
    base_val = count_baseline(val_items, table, interval)
    residual_train = y_train - base_train

    print("Fitting train-only speed prior...", flush=True)
    prior = SpeedPrior().fit(raw_train)
    print("Building base and speed-prior features...", flush=True)
    X_base_train, base_names = build_enhanced_feature_matrix(train_data, base_train)
    X_base_val, _ = build_enhanced_feature_matrix(val_items, base_val)
    X_speed_train, speed_names = speed_feature_matrix(train_data, base_train, prior)
    X_speed_val, _ = speed_feature_matrix(val_items, base_val, prior)
    X_train = np.column_stack([X_base_train, X_speed_train])
    X_val = np.column_stack([X_base_val, X_speed_val])
    print(f"X_train={X_train.shape}, features={len(base_names) + len(speed_names)}", flush=True)

    results = [
        metric("current_best_knn_blend", best_pred, y_val),
        metric("count_median_baseline", base_val, y_val),
        metric("speed_prior_direct", X_speed_val[:, 0], y_val),
    ]

    preds: dict[str, np.ndarray] = {}
    for name, seed, depth, leaves in (
        ("speed_prior_hgb_d8", 42, 8, 63),
        ("speed_prior_hgb_d6", 7, 6, 31),
    ):
        print(f"Fitting {name}...", flush=True)
        model = hgb(seed=seed, depth=depth, leaves=leaves)
        model.fit(X_train, residual_train)
        pred = np.maximum(base_val + model.predict(X_val), 30.0)
        preds[name] = pred
        result = metric(name, pred, y_val)
        results.append(result)
        print_result(result)

    print("Searching blends with current best...", flush=True)
    for name, pred in preds.items():
        for w in np.linspace(0.05, 0.50, 10):
            results.append(metric(f"blend_best_{name}_w{w:.2f}", (1.0 - w) * best_pred + w * pred, y_val))

    print("\nResults sorted by MAE:", flush=True)
    for result in sorted(results, key=lambda item: item.mae)[:30]:
        print_result(result)
    print(f"Elapsed: {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
