"""Feature extraction for Task B."""
from __future__ import annotations
import numpy as np
from src.common.geo import (
    haversine_batch, trajectory_length, straight_line_distance,
    bearing_batch, bearing_change, speed_ms
)
from src.common.time_features import extract_time_features

# Xi'an bounding box: lon [108.7, 109.2], lat [33.9, 34.5]
GRID_LON_MIN, GRID_LON_MAX = 108.7, 109.2
GRID_LAT_MIN, GRID_LAT_MAX = 33.9, 34.5
GRID_SIZE = 20  # 20x20 grid


def _grid_xy(lon: float, lat: float) -> tuple[int, int]:
    gx = int((lon - GRID_LON_MIN) / (GRID_LON_MAX - GRID_LON_MIN) * GRID_SIZE)
    gy = int((lat - GRID_LAT_MIN) / (GRID_LAT_MAX - GRID_LAT_MIN) * GRID_SIZE)
    gx = max(0, min(GRID_SIZE - 1, gx))
    gy = max(0, min(GRID_SIZE - 1, gy))
    return gx, gy


def extract_features(item: dict) -> dict:
    """Extract all features from a trajectory item."""
    coords = np.array(item["coords"], dtype=np.float64)
    ts = item["departure_timestamp"]

    n = len(coords)
    start = coords[0]
    end = coords[-1]

    total_dist = trajectory_length(coords)
    straight_dist = straight_line_distance(coords)
    tortuosity = total_dist / max(straight_dist, 1.0)

    segs = haversine_batch(coords[:-1], coords[1:]) if n >= 2 else np.array([0.0])
    mean_seg = float(np.mean(segs))
    max_seg = float(np.max(segs))
    std_seg = float(np.std(segs))
    # Segment distance percentiles (capture distribution shape)
    p25_seg = float(np.percentile(segs, 25))
    p50_seg = float(np.percentile(segs, 50))
    p75_seg = float(np.percentile(segs, 75))
    p90_seg = float(np.percentile(segs, 90))
    # Fraction of "slow" segments (< 50m in 15s ≈ 12 km/h)
    slow_ratio = float(np.mean(segs < 50.0))

    bearings = bearing_batch(coords) if n >= 2 else np.array([0.0])
    bc = bearing_change(bearings) if len(bearings) >= 2 else np.array([0.0])
    bc_mean = float(np.mean(bc))
    bc_std = float(np.std(bc))

    bbox_w = float(np.max(coords[:, 0]) - np.min(coords[:, 0]))
    bbox_h = float(np.max(coords[:, 1]) - np.min(coords[:, 1]))

    sgx, sgy = _grid_xy(start[0], start[1])
    egx, egy = _grid_xy(end[0], end[1])

    time_feats = extract_time_features(ts)

    return {
        # Distance features
        "total_distance": total_dist,
        "straight_distance": straight_dist,
        "tortuosity": tortuosity,
        "mean_segment_distance": mean_seg,
        "max_segment_distance": max_seg,
        "std_segment_distance": std_seg,
        "p25_segment_distance": p25_seg,
        "p50_segment_distance": p50_seg,
        "p75_segment_distance": p75_seg,
        "p90_segment_distance": p90_seg,
        "slow_segment_ratio": slow_ratio,
        # Estimated travel time from sampling structure (num_points * ~15.64s)
        "est_tt_from_npts": (n - 1) * 15.64,
        # Shape features
        "num_points": n,
        "bearing_change_mean": bc_mean,
        "bearing_change_std": bc_std,
        "bbox_width": bbox_w,
        "bbox_height": bbox_h,
        "bbox_area": bbox_w * bbox_h,
        # Spatial features
        "start_lon": float(start[0]),
        "start_lat": float(start[1]),
        "end_lon": float(end[0]),
        "end_lat": float(end[1]),
        "delta_lon": float(end[0] - start[0]),
        "delta_lat": float(end[1] - start[1]),
        "start_grid_x": sgx,
        "start_grid_y": sgy,
        "end_grid_x": egx,
        "end_grid_y": egy,
        # Time features
        **time_feats,
    }


def build_feature_matrix(items: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix X for a list of items."""
    rows = [extract_features(item) for item in items]
    if not rows:
        return np.empty((0, 0)), []
    cols = list(rows[0].keys())
    X = np.array([[row[c] for c in cols] for row in rows], dtype=np.float64)
    return X, cols


def build_sampling_phase_matrix(
    items: list[dict],
    n_baseline: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build timestamp phase and sampling-structure features for Task B.

    The ds15 trajectories are sampled at a near-regular interval, so travel time
    is strongly tied to point count. Raw timestamp modulo features help capture
    the residual introduced by sampling phase and endpoint alignment.
    """
    if not items:
        return np.empty((0, 0)), []

    ts = np.array([int(item["departure_timestamp"]) for item in items], dtype=np.int64)
    n_points = np.array([len(item["coords"]) for item in items], dtype=np.float64)
    base = np.asarray(n_baseline, dtype=np.float64)
    day_second = (ts % 86_400).astype(np.float64)

    columns: list[tuple[str, np.ndarray]] = [
        ("unix_day_second", day_second),
        ("unix_day_fraction", day_second / 86_400.0),
        ("num_points_raw", n_points),
        ("num_segments", n_points - 1.0),
        ("num_segments_sq", (n_points - 1.0) ** 2),
        ("n_median_baseline", base),
        ("n_median_interval", base / np.maximum(n_points - 1.0, 1.0)),
    ]

    for mod in (2, 3, 4, 5, 6, 7, 10, 12, 15, 20, 30, 60, 120, 300, 600, 900, 1800, 3600):
        remainder = (ts % mod).astype(np.float64)
        columns.append((f"ts_mod_{mod}", remainder))
        if mod <= 120:
            columns.append((f"ts_mod_{mod}_sin", np.sin(2 * np.pi * remainder / mod)))
            columns.append((f"ts_mod_{mod}_cos", np.cos(2 * np.pi * remainder / mod)))

    names = [name for name, _ in columns]
    matrix = np.column_stack([values for _, values in columns]).astype(np.float64)
    return matrix, names


def build_enhanced_feature_matrix(
    items: list[dict],
    n_baseline: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build base geometry/time features plus sampling phase features."""
    base_matrix, base_names = build_feature_matrix(items)
    phase_matrix, phase_names = build_sampling_phase_matrix(items, n_baseline)
    if base_matrix.size == 0:
        return phase_matrix, phase_names
    return np.column_stack([base_matrix, phase_matrix]), base_names + phase_names
