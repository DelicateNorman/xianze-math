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
