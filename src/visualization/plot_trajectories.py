"""Visualize trajectory recovery comparisons."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_recovery_comparison(input_item: dict, pred_coords: np.ndarray,
                              gt_coords: np.ndarray | None = None,
                              title: str = "", save_path: str | None = None) -> None:
    """Plot sparse input, prediction, and optionally ground truth."""
    mask = np.array(input_item["mask"], dtype=bool)
    known = np.array(input_item["coords"], dtype=np.float64)[mask]

    fig, ax = plt.subplots(figsize=(8, 6))

    if gt_coords is not None:
        ax.plot(gt_coords[:, 0], gt_coords[:, 1], "g-", linewidth=1, label="Ground Truth", alpha=0.7)

    ax.plot(pred_coords[:, 0], pred_coords[:, 1], "b-", linewidth=1.5, label="Prediction")
    ax.scatter(known[:, 0], known[:, 1], c="red", s=30, zorder=5, label="Known Points")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or f"Traj {input_item['traj_id']}")
    ax.legend()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_error_distribution(errors_a1: list[float], errors_a2: list[float] | None = None,
                              labels: list[str] | None = None,
                              save_path: str | None = None) -> None:
    """Plot error distribution histogram for Task A methods."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(errors_a1, bins=60, alpha=0.6, label=labels[0] if labels else "Method 1",
            color="steelblue", density=True)
    if errors_a2 is not None:
        ax.hist(errors_a2, bins=60, alpha=0.6, label=labels[1] if labels and len(labels) > 1 else "Method 2",
                color="coral", density=True)
    ax.set_xlabel("Haversine error (m)")
    ax.set_ylabel("Density")
    ax.set_title("Task A Error Distribution")
    ax.legend()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_tte_scatter(pred_times: np.ndarray, true_times: np.ndarray,
                     title: str = "Predicted vs True Travel Time",
                     save_path: str | None = None) -> None:
    """Scatter plot of predicted vs true travel times."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true_times / 60, pred_times / 60, alpha=0.2, s=5, c="steelblue")
    lim = max(true_times.max(), pred_times.max()) / 60
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("True travel time (min)")
    ax.set_ylabel("Predicted travel time (min)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
