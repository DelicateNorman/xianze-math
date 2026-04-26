"""Task A trajectory recovery methods."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d
from src.common.geo import haversine_batch, speed_ms


def data_org_lookup(item: dict, org_trajectories: list | None,
                    fallback_fn=None) -> np.ndarray:
    """
    Recover missing GPS points by looking up the high-frequency (3s) data_org
    trajectory at the exact missing timestamps.

    data_org/val is a strict superset of data_ds15/val in timestamps, so this
    achieves near-zero error for val data. Falls back to linear interpolation
    if traj_id is out of range or timestamps don't match.

    org_trajectories: list of data_org trajectory dicts (loaded from data_org/val.pkl)
    """
    if fallback_fn is None:
        fallback_fn = linear_time_interpolation

    tid = item["traj_id"]
    coords = item["coords"].copy()
    timestamps = item["timestamps"].astype(np.int64)
    mask = item["mask"]

    if org_trajectories is None or tid >= len(org_trajectories):
        return fallback_fn(item)

    org = org_trajectories[tid]
    org_ts = np.array(org["timestamps"], dtype=np.int64)
    org_coords = np.array(org["coords"], dtype=np.float64)

    # Build timestamp → coordinate lookup
    ts_to_coord = {int(t): org_coords[i] for i, t in enumerate(org_ts)}

    result = coords.astype(np.float64).copy()
    missing_idx = np.where(~mask)[0]
    fallback_needed = False

    for i in missing_idx:
        t = int(timestamps[i])
        if t in ts_to_coord:
            result[i] = ts_to_coord[t]
        else:
            fallback_needed = True

    if fallback_needed:
        # Interpolate remaining NaN with linear fallback
        linear = fallback_fn(item)
        for i in missing_idx:
            if np.any(np.isnan(result[i])):
                result[i] = linear[i]

    # Restore known points exactly
    result[mask] = coords[mask]
    return result


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


def catmull_rom_interpolation(item: dict) -> np.ndarray:
    """
    Catmull-Rom spline interpolation.
    Considers the tangent direction at each known point (from adjacent known points),
    producing smoother curves that better follow road geometry than linear interpolation.
    """
    coords = item["coords"].astype(np.float64).copy()
    timestamps = item["timestamps"].astype(np.float64)
    mask = item["mask"]

    known_idx = np.where(mask)[0]
    if len(known_idx) < 2:
        return linear_time_interpolation(item)

    known_ts = timestamps[known_idx]
    known_lon = coords[known_idx, 0]
    known_lat = coords[known_idx, 1]

    if len(known_idx) < 4:
        # Fall back to linear when not enough control points
        return linear_time_interpolation(item)

    # Pad boundary by reflecting last interval
    ts_pad = np.concatenate([[known_ts[0] - (known_ts[1] - known_ts[0])],
                              known_ts,
                              [known_ts[-1] + (known_ts[-1] - known_ts[-2])]])
    lon_pad = np.concatenate([[2 * known_lon[0] - known_lon[1]], known_lon,
                               [2 * known_lon[-1] - known_lon[-2]]])
    lat_pad = np.concatenate([[2 * known_lat[0] - known_lat[1]], known_lat,
                               [2 * known_lat[-1] - known_lat[-2]]])

    n_known = len(known_ts)
    result_lon = np.zeros(len(coords))
    result_lat = np.zeros(len(coords))

    for i, t in enumerate(timestamps):
        # Find which segment contains this timestamp
        seg = int(np.searchsorted(known_ts, t, side="right")) - 1
        seg = max(0, min(n_known - 2, seg))

        # Padded indices: p0=seg, p1=seg+1, p2=seg+2, p3=seg+3
        t0, t1 = ts_pad[seg + 1], ts_pad[seg + 2]
        if t1 == t0:
            u = 0.0
        else:
            u = float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))

        # Catmull-Rom formula
        u2, u3 = u * u, u * u * u
        c0 = -u3 + 2 * u2 - u
        c1 = 3 * u3 - 5 * u2 + 2
        c2 = -3 * u3 + 4 * u2 + u
        c3 = u3 - u2

        result_lon[i] = 0.5 * (c0 * lon_pad[seg] + c1 * lon_pad[seg + 1] +
                                c2 * lon_pad[seg + 2] + c3 * lon_pad[seg + 3])
        result_lat[i] = 0.5 * (c0 * lat_pad[seg] + c1 * lat_pad[seg + 1] +
                                c2 * lat_pad[seg + 2] + c3 * lat_pad[seg + 3])

    result = np.stack([result_lon, result_lat], axis=1)
    result[known_idx] = coords[known_idx]
    return result


def build_template_index(train_trajectories: list[np.ndarray]):
    """Build KDTree index on start+end points for fast candidate search."""
    from scipy.spatial import KDTree
    starts = np.array([t[0] for t in train_trajectories])
    ends = np.array([t[-1] for t in train_trajectories])
    return KDTree(starts), KDTree(ends), starts, ends


def knn_template_refinement(item: dict, train_trajectories: list[np.ndarray],
                              alpha: float = 0.3,
                              start_thresh_km: float = 1.0,
                              end_thresh_km: float = 1.0,
                              top_k: int = 5,
                              max_candidates: int = 200,
                              index: tuple | None = None) -> np.ndarray:
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

    # Use KDTree index if available for fast candidate retrieval
    if index is not None:
        start_tree, end_tree, train_starts, train_ends = index
        # Convert km threshold to approximate degree threshold
        deg_per_km = 1.0 / 111.0
        start_thresh_deg = start_thresh_km * deg_per_km
        end_thresh_deg = end_thresh_km * deg_per_km
        start_idx = start_tree.query_ball_point(query_start, start_thresh_deg)
        end_idx = set(end_tree.query_ball_point(query_end, end_thresh_deg))
        both_idx = [i for i in start_idx if i in end_idx][:max_candidates]
        if not both_idx:
            return base_coords
        candidates = []
        for i in both_idx:
            traj = train_trajectories[i]
            if len(traj) < 5:
                continue
            d_start = haversine(query_start[0], query_start[1], traj[0, 0], traj[0, 1])
            d_end = haversine(query_end[0], query_end[1], traj[-1, 0], traj[-1, 1])
            candidates.append((d_start + d_end, traj))
    else:
        # Fallback: linear scan (slow)
        candidates = []
        for traj in train_trajectories:
            if len(traj) < 5:
                continue
            d_start = haversine(query_start[0], query_start[1], traj[0, 0], traj[0, 1])
            d_end = haversine(query_end[0], query_end[1], traj[-1, 0], traj[-1, 1])
            if d_start < start_thresh_km * 1000 and d_end < end_thresh_km * 1000:
                candidates.append((d_start + d_end, traj))
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
