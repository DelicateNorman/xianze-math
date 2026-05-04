"""Exploratory data analysis for Xi'an taxi trajectories."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.io import load_pkl
from src.common.geo import trajectory_length, speed_ms
from src.common.paths import candidate_data_roots, resolve_existing_path
from src.common.time_features import departure_hour

DATA_ROOT = next((root for root in candidate_data_roots() if (root / "data_ds15" / "train.pkl").exists()),
                 Path("data/student_release"))
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_ds15_train():
    path = resolve_existing_path(DATA_ROOT / "data_ds15" / "train.pkl")
    data = load_pkl(path)
    result = []
    for item in data:
        coords = np.array(item["coords"], dtype=np.float64)
        ts = np.array(item["timestamps"], dtype=np.int64)
        if len(coords) < 2:
            continue
        tt = int(ts[-1] - ts[0])
        if tt <= 0:
            continue
        dist = trajectory_length(coords)
        spd = dist / tt  # m/s
        result.append({
            "coords": coords,
            "timestamps": ts,
            "travel_time": tt,
            "total_distance": dist,
            "speed_ms": spd,
            "n_points": len(coords),
            "hour": departure_hour(int(ts[0])),
        })
    return result


def plot_length_distribution(data):
    n_pts = [d["n_points"] for d in data]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(n_pts, bins=50, color="steelblue", edgecolor="white")
    ax.set_xlabel("Number of GPS points")
    ax.set_ylabel("Count")
    ax.set_title("Trajectory Length Distribution (ds15 train)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "traj_length_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  N points: mean={np.mean(n_pts):.1f}, median={np.median(n_pts):.0f}, "
          f"min={min(n_pts)}, max={max(n_pts)}")


def plot_travel_time_distribution(data):
    tt = [d["travel_time"] / 60 for d in data]  # minutes
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(tt, bins=60, range=(0, 80), color="coral", edgecolor="white")
    ax.set_xlabel("Travel time (minutes)")
    ax.set_ylabel("Count")
    ax.set_title("Travel Time Distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "travel_time_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Travel time (min): mean={np.mean(tt):.1f}, median={np.median(tt):.1f}")


def plot_distance_distribution(data):
    dists = [d["total_distance"] / 1000 for d in data]  # km
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dists, bins=60, range=(0, 30), color="mediumseagreen", edgecolor="white")
    ax.set_xlabel("Total path distance (km)")
    ax.set_ylabel("Count")
    ax.set_title("Path Distance Distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "distance_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Distance (km): mean={np.mean(dists):.2f}, median={np.median(dists):.2f}")


def plot_speed_distribution(data):
    spds = [d["speed_ms"] * 3.6 for d in data]  # km/h
    spds = [s for s in spds if s < 150]  # filter extreme outliers
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(spds, bins=60, color="mediumpurple", edgecolor="white")
    ax.set_xlabel("Average speed (km/h)")
    ax.set_ylabel("Count")
    ax.set_title("Average Speed Distribution")
    ax.axvline(np.median(spds), color="red", linestyle="--", label=f"Median={np.median(spds):.1f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "speed_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Speed (km/h): mean={np.mean(spds):.1f}, median={np.median(spds):.1f}")


def plot_hourly_speed(data):
    by_hour: dict[int, list] = {h: [] for h in range(24)}
    for d in data:
        by_hour[d["hour"]].append(d["speed_ms"] * 3.6)
    hours = list(range(24))
    mean_speeds = [np.median(by_hour[h]) if by_hour[h] else 0 for h in hours]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hours, mean_speeds, color="steelblue")
    ax.set_xlabel("Departure hour")
    ax.set_ylabel("Median speed (km/h)")
    ax.set_title("Hourly Average Speed (Xi'an Taxis)")
    ax.set_xticks(hours)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hourly_speed.png", dpi=150)
    plt.close(fig)
    print(f"  Peak slowdown hour: {hours[np.argmin(mean_speeds)]} h ({min(mean_speeds):.1f} km/h)")


def plot_spatial_distribution(data, sample=5000):
    np.random.seed(42)
    idx = np.random.choice(len(data), min(sample, len(data)), replace=False)
    starts = np.array([data[i]["coords"][0] for i in idx])
    ends = np.array([data[i]["coords"][-1] for i in idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, pts, title in [(axes[0], starts, "Start Points"),
                             (axes[1], ends, "End Points")]:
        ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.3, c="steelblue")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
    fig.suptitle("Spatial Distribution of Taxi Trips (sample)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "spatial_distribution.png", dpi=150)
    plt.close(fig)


def main():
    print("Loading training data...")
    data = load_ds15_train()
    print(f"Loaded {len(data)} valid trajectories")

    print("\nTrajectory length:")
    plot_length_distribution(data)

    print("\nTravel time:")
    plot_travel_time_distribution(data)

    print("\nPath distance:")
    plot_distance_distribution(data)

    print("\nAverage speed:")
    plot_speed_distribution(data)

    print("\nHourly speed pattern:")
    plot_hourly_speed(data)

    print("\nSpatial distribution:")
    plot_spatial_distribution(data)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
