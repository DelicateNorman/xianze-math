"""Task A trajectory recovery methods."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from src.common.geo import haversine_batch, speed_ms

EARTH_RADIUS_APPROX_M = 111_000.0
XIAN_LAT_RAD = np.deg2rad(34.25)
LON_SCALE_M = EARTH_RADIUS_APPROX_M * np.cos(XIAN_LAT_RAD)
LAT_SCALE_M = EARTH_RADIUS_APPROX_M


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


def pchip_time_interpolation(item: dict) -> np.ndarray:
    """
    Shape-preserving cubic Hermite interpolation over timestamps.

    PCHIP keeps the endpoint constraints of linear interpolation while using a
    monotone local cubic slope estimate. In this trajectory recovery task it
    gives smoother turns than linear interpolation and avoids the larger
    overshoot risk of unconstrained cubic splines.
    """
    coords = item["coords"].astype(np.float64).copy()
    timestamps = item["timestamps"].astype(np.float64)
    mask = item["mask"]

    known_idx = np.where(mask)[0]
    if len(known_idx) < 2:
        return linear_time_interpolation(item)

    known_ts = timestamps[known_idx]
    result = np.empty_like(coords, dtype=np.float64)

    for dim in (0, 1):
        known_values = coords[known_idx, dim]
        try:
            interpolator = PchipInterpolator(
                known_ts,
                known_values,
                extrapolate=False,
            )
            values = interpolator(timestamps)
            boundary_fill = np.interp(
                timestamps,
                known_ts,
                known_values,
                left=known_values[0],
                right=known_values[-1],
            )
            result[:, dim] = np.where(np.isnan(values), boundary_fill, values)
        except ValueError:
            result[:, dim] = np.interp(
                timestamps,
                known_ts,
                known_values,
                left=known_values[0],
                right=known_values[-1],
            )

    result[known_idx] = coords[known_idx]
    return result


def _lonlat_to_local_xy(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    return np.column_stack([coords[:, 0] * LON_SCALE_M, coords[:, 1] * LAT_SCALE_M])


def _local_xy_to_lonlat(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    return np.column_stack([xy[:, 0] / LON_SCALE_M, xy[:, 1] / LAT_SCALE_M])


def _segment_feature(start_xy: np.ndarray, end_xy: np.ndarray) -> np.ndarray:
    displacement = end_xy - start_xy
    midpoint = (start_xy + end_xy) * 0.5
    return np.r_[midpoint / 1000.0, displacement / 500.0].astype(np.float32)


def build_local_segment_index(
    train_trajectories: list[np.ndarray],
    spans: list[int],
    max_segments_per_span: int = 250_000,
    samples_per_traj_span: int = 3,
    min_displacement_m: float = 20.0,
    seed: int = 42,
) -> dict[int, tuple]:
    """Build KDTree indexes of local train segments keyed by endpoint gap span."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    wanted_spans = sorted({int(s) for s in spans if int(s) >= 2})
    features: dict[int, list[np.ndarray]] = {span: [] for span in wanted_spans}
    residuals: dict[int, list[np.ndarray]] = {span: [] for span in wanted_spans}

    for traj in train_trajectories:
        xy = _lonlat_to_local_xy(traj)
        n_points = len(xy)

        for span in wanted_spans:
            if len(features[span]) >= max_segments_per_span or n_points <= span:
                continue

            starts = np.arange(n_points - span)
            if len(starts) > samples_per_traj_span:
                starts = rng.choice(starts, size=samples_per_traj_span, replace=False)

            ratios = (np.arange(span + 1, dtype=np.float64) / span)[:, None]
            for start_idx in starts:
                end_idx = start_idx + span
                start_xy = xy[start_idx]
                end_xy = xy[end_idx]
                displacement = end_xy - start_xy
                if np.linalg.norm(displacement) < min_displacement_m:
                    continue

                linear_segment = start_xy + ratios * displacement
                features[span].append(_segment_feature(start_xy, end_xy))
                residuals[span].append(
                    (xy[start_idx:end_idx + 1] - linear_segment).astype(np.float32)
                )

                if len(features[span]) >= max_segments_per_span:
                    break

        if all(len(features[span]) >= max_segments_per_span for span in wanted_spans):
            break

    indexes = {}
    for span in wanted_spans:
        if not features[span]:
            continue
        feature_array = np.asarray(features[span], dtype=np.float32)
        residual_array = np.asarray(residuals[span], dtype=np.float32)
        indexes[span] = (cKDTree(feature_array), residual_array)
    return indexes


def local_segment_template_interpolation(
    item: dict,
    segment_index: dict[int, tuple] | None,
    alpha: float = 1.0,
    top_k: int = 20,
    max_feature_distance: float = 2.5,
    confidence_blend: str = "none",
    confidence_threshold: float = 1.2,
    confidence_scale: float = 1.0,
    fallback_fn=None,
) -> np.ndarray:
    """
    Recover gaps with train-set local segment templates.

    For each pair of adjacent known points, this method finds training segments
    with the same point span and similar local start/end geometry. It averages
    their residual from straight-line interpolation, then adds that learned
    curvature residual to the current gap. This fixes the failure mode of
    whole-trajectory KNN by matching only the local gap being recovered.
    """
    if fallback_fn is None:
        fallback_fn = pchip_time_interpolation

    fallback = fallback_fn(item)
    if not segment_index:
        return fallback

    linear_lonlat = linear_time_interpolation(item)
    linear_xy = _lonlat_to_local_xy(linear_lonlat)
    fallback_xy = _lonlat_to_local_xy(fallback)
    result_xy = fallback_xy.copy()
    coords = item["coords"]
    mask = item["mask"]
    known_idx = np.where(mask)[0]

    for left_idx, right_idx in zip(known_idx[:-1], known_idx[1:]):
        span = int(right_idx - left_idx)
        if span < 2 or span not in segment_index:
            continue

        start_xy = linear_xy[left_idx]
        end_xy = linear_xy[right_idx]
        if np.linalg.norm(end_xy - start_xy) < 20.0:
            continue

        tree, residuals = segment_index[span]
        k = min(top_k, len(residuals))
        distances, candidate_idx = tree.query(_segment_feature(start_xy, end_xy), k=k)
        distances = np.atleast_1d(distances)
        candidate_idx = np.atleast_1d(candidate_idx)
        keep = distances < max_feature_distance
        if not np.any(keep):
            continue

        kept_distances = distances[keep]
        weights = 1.0 / (kept_distances + 1e-3)
        weights = weights / weights.sum()
        mean_residual = np.tensordot(
            weights,
            residuals[candidate_idx[keep]],
            axes=(0, 0),
        )
        template_xy = linear_xy[left_idx:right_idx + 1] + alpha * mean_residual

        nearest_distance = float(kept_distances[0])
        if confidence_blend == "linear":
            confidence = np.clip(
                (confidence_threshold - nearest_distance) / max(confidence_threshold, 1e-6),
                0.0,
                1.0,
            )
        elif confidence_blend == "exp":
            confidence = np.exp(-((nearest_distance / max(confidence_scale, 1e-6)) ** 2))
        else:
            confidence = 1.0

        current_fallback = fallback_xy[left_idx:right_idx + 1]
        result_xy[left_idx:right_idx + 1] = (
            (1.0 - confidence) * current_fallback + confidence * template_xy
        )

    result = _local_xy_to_lonlat(result_xy)
    result[mask] = coords[mask]
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
