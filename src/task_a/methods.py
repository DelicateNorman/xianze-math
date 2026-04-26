"""Task A trajectory recovery methods."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d
from src.common.geo import haversine_batch, speed_ms


def linear_time_interpolation(item: dict) -> np.ndarray:
    """
    Interpolate missing coords using timestamps of known points.
    Known points are preserved exactly.
    """
    coords = item["coords"].copy()
    timestamps = item["timestamps"].astype(np.float64)
    mask = item["mask"]

    known_idx = np.where(mask)[0]
    if len(known_idx) == 0:
        return coords  # nothing to interpolate

    known_ts = timestamps[known_idx]
    known_lon = coords[known_idx, 0]
    known_lat = coords[known_idx, 1]

    all_ts = timestamps

    # Interpolate lon
    if len(known_idx) >= 2:
        f_lon = interp1d(known_ts, known_lon, kind="linear",
                         bounds_error=False, fill_value=(known_lon[0], known_lon[-1]))
        f_lat = interp1d(known_ts, known_lat, kind="linear",
                         bounds_error=False, fill_value=(known_lat[0], known_lat[-1]))
        interp_lon = f_lon(all_ts)
        interp_lat = f_lat(all_ts)
    else:
        # Only one known point — fill everything with that point
        interp_lon = np.full(len(coords), known_lon[0])
        interp_lat = np.full(len(coords), known_lat[0])

    result = np.stack([interp_lon, interp_lat], axis=1)
    # Restore known points exactly (avoid float precision changes)
    result[known_idx] = coords[known_idx]
    return result


def _smooth_unknown_points(coords: np.ndarray, mask: np.ndarray,
                            window: int = 3, iterations: int = 2) -> np.ndarray:
    """Apply moving-average smoothing only to unknown (mask=False) points."""
    result = coords.copy()
    unknown_idx = np.where(~mask)[0]
    if len(unknown_idx) == 0:
        return result

    for _ in range(iterations):
        smoothed = result.copy()
        for i in unknown_idx:
            lo = max(0, i - window)
            hi = min(len(result), i + window + 1)
            smoothed[i] = result[lo:hi].mean(axis=0)
        result = smoothed
    # Ensure known points are untouched
    result[mask] = coords[mask]
    return result


def linear_with_speed_smoothing(item: dict, max_speed_kmh: float = 120.0,
                                  smoothing_window: int = 3,
                                  iterations: int = 2) -> np.ndarray:
    """
    Linear interpolation followed by speed-constraint smoothing on unknown points.
    Only smooths points involved in anomalously high-speed segments; does NOT
    apply a blanket final smoothing pass (which degrades good interpolations).
    """
    coords = linear_time_interpolation(item)
    timestamps = item["timestamps"].astype(np.float64)
    mask = item["mask"]

    max_speed_ms = max_speed_kmh / 3.6
    unknown_idx = set(np.where(~mask)[0])

    # Iteratively smooth points that have excessive implied speed
    for _ in range(iterations):
        speeds = speed_ms(coords, timestamps)  # shape (N-1,)
        anomalous = np.where(speeds > max_speed_ms)[0]

        if len(anomalous) == 0:
            break

        # Collect unknown points adjacent to anomalous segments
        to_smooth = set()
        for seg in anomalous:
            if seg in unknown_idx:
                to_smooth.add(seg)
            if (seg + 1) in unknown_idx:
                to_smooth.add(seg + 1)

        if not to_smooth:
            break

        for i in sorted(to_smooth):
            lo = max(0, i - smoothing_window)
            hi = min(len(coords), i + smoothing_window + 1)
            neighbors = coords[lo:hi]
            coords[i] = neighbors.mean(axis=0)

    # Restore known points (may have been slightly modified by window averaging)
    coords[mask] = item["coords"][mask]
    return coords


def knn_template_refinement(item: dict, train_trajectories: list[np.ndarray],
                              alpha: float = 0.3,
                              start_thresh_km: float = 1.0,
                              end_thresh_km: float = 1.0,
                              top_k: int = 5,
                              max_candidates: int = 200) -> np.ndarray:
    """
    Refine linear interpolation using similar historical trajectories.
    train_trajectories: list of full coord arrays from training set.
    alpha: weight for template prediction (0=pure linear, 1=pure template).
    """
    from src.common.geo import haversine

    base_coords = linear_with_speed_smoothing(item)
    mask = item["mask"]

    if len(train_trajectories) == 0:
        return base_coords

    query_start = base_coords[0]
    query_end = base_coords[-1]
    query_len = len(base_coords)

    # Candidate filtering by start/end proximity
    candidates = []
    for traj in train_trajectories:
        if len(traj) < 5:
            continue
        d_start = haversine(query_start[0], query_start[1], traj[0, 0], traj[0, 1])
        d_end = haversine(query_end[0], query_end[1], traj[-1, 0], traj[-1, 1])
        if d_start < start_thresh_km * 1000 and d_end < end_thresh_km * 1000:
            score = d_start + d_end
            candidates.append((score, traj))
        if len(candidates) >= max_candidates:
            break

    if not candidates:
        return base_coords

    candidates.sort(key=lambda x: x[0])
    top_trajs = [c[1] for c in candidates[:top_k]]

    # Resample each template to match query length and blend
    template_coords = np.zeros_like(base_coords)
    for traj in top_trajs:
        orig_idx = np.linspace(0, len(traj) - 1, query_len)
        from scipy.interpolate import interp1d
        f_lon = interp1d(np.arange(len(traj)), traj[:, 0], kind="linear")
        f_lat = interp1d(np.arange(len(traj)), traj[:, 1], kind="linear")
        template_coords += np.stack([f_lon(orig_idx), f_lat(orig_idx)], axis=1)
    template_coords /= len(top_trajs)

    result = (1 - alpha) * base_coords + alpha * template_coords
    # Restore known points
    result[mask] = item["coords"][mask]
    return result
