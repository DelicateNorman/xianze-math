"""Time feature extraction from Unix timestamps."""
import numpy as np
from datetime import datetime, timezone


def unix_to_datetime(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()


def departure_hour(ts: int) -> int:
    return unix_to_datetime(ts).hour


def day_of_week(ts: int) -> int:
    """0=Monday, 6=Sunday."""
    return unix_to_datetime(ts).weekday()


def is_weekend(ts: int) -> bool:
    return day_of_week(ts) >= 5


def is_peak_hour(ts: int) -> bool:
    """Simplified peak: 7-9 and 17-19."""
    h = departure_hour(ts)
    return (7 <= h <= 9) or (17 <= h <= 19)


def is_night(ts: int) -> bool:
    h = departure_hour(ts)
    return h >= 22 or h <= 5


def minute_of_day(ts: int) -> int:
    dt = unix_to_datetime(ts)
    return dt.hour * 60 + dt.minute


def extract_time_features(ts: int) -> dict:
    """Return all time features as a dict."""
    return {
        "departure_hour": departure_hour(ts),
        "day_of_week": day_of_week(ts),
        "is_weekend": int(is_weekend(ts)),
        "is_peak_hour": int(is_peak_hour(ts)),
        "is_night": int(is_night(ts)),
        "minute_of_day": minute_of_day(ts),
    }
