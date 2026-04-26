"""Geographic utilities: Haversine distance, bearing, trajectory stats."""
import numpy as np

EARTH_RADIUS_M = 6_371_000.0
DEG2RAD = np.pi / 180.0


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine distance in meters between two (lon, lat) points."""
    dlon = (lon2 - lon1) * DEG2RAD
    dlat = (lat2 - lat1) * DEG2RAD
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * DEG2RAD) * np.cos(lat2 * DEG2RAD) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def haversine_batch(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """
    Vectorized Haversine distance in meters.
    coords1, coords2: (..., 2) arrays in [lon, lat] format.
    """
    lon1, lat1 = coords1[..., 0] * DEG2RAD, coords1[..., 1] * DEG2RAD
    lon2, lat2 = coords2[..., 0] * DEG2RAD, coords2[..., 1] * DEG2RAD
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def trajectory_length(coords: np.ndarray) -> float:
    """Total Haversine path length in meters. coords shape: (N, 2) [lon, lat]."""
    if len(coords) < 2:
        return 0.0
    dists = haversine_batch(coords[:-1], coords[1:])
    return float(np.nansum(dists))


def straight_line_distance(coords: np.ndarray) -> float:
    """Straight-line Haversine distance from start to end."""
    if len(coords) < 2:
        return 0.0
    return float(haversine_batch(coords[[0]], coords[[-1]])[0])


def bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Bearing angle in degrees [0, 360) from point 1 to point 2."""
    lat1r, lat2r = lat1 * DEG2RAD, lat2 * DEG2RAD
    dlon = (lon2 - lon1) * DEG2RAD
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def bearing_batch(coords: np.ndarray) -> np.ndarray:
    """Bearing between consecutive points. Shape (N-1,) for input (N,2)."""
    lon1, lat1 = coords[:-1, 0] * DEG2RAD, coords[:-1, 1] * DEG2RAD
    lon2, lat2 = coords[1:, 0] * DEG2RAD, coords[1:, 1] * DEG2RAD
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def bearing_change(bearings: np.ndarray) -> np.ndarray:
    """Absolute angle change between consecutive bearings. Shape (N-2,)."""
    diff = bearings[1:] - bearings[:-1]
    diff = (diff + 180) % 360 - 180
    return np.abs(diff)


def speed_ms(coords: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Speed in m/s between consecutive points. Shape (N-1,)."""
    dists = haversine_batch(coords[:-1], coords[1:])
    dt = np.diff(timestamps.astype(float))
    dt = np.where(dt == 0, 1e-6, dt)
    return dists / dt
