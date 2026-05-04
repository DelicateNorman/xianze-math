"""Task A entry point."""
from __future__ import annotations
import argparse
import sys
import csv
import os
from datetime import datetime
from pathlib import Path

from src.common.config import load_config
from src.common.io import load_pkl, save_pkl, validate_task_a_output
from src.common.logging_utils import get_logger
from src.common.paths import resolve_existing_path
from src.common.seed import set_seed
from src.task_a.dataset import load_task_a_input, load_task_a_gt
from src.task_a.predict import predict_task_a
from src.task_a.evaluate import evaluate_task_a, check_known_points_preserved

logger = get_logger("task_a")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task A: Trajectory Recovery")
    p.add_argument("--input", required=True, help="Input pkl file path")
    p.add_argument("--output", required=True, help="Output pkl file path")
    p.add_argument("--gt", default=None, help="Ground truth pkl (required for val mode)")
    p.add_argument("--method", default="local_segment_template_interpolation",
                   choices=["linear_time_interpolation", "linear_with_speed_smoothing",
                            "catmull_rom_interpolation", "pchip_time_interpolation",
                            "knn_template_refinement",
                            "local_segment_template_interpolation"])
    p.add_argument("--config", default="configs/task_a_advanced.yaml")
    p.add_argument("--mode", choices=["val", "predict"], default="val")
    p.add_argument("--train-data", default=None, help="Training data pkl (for KNN method)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if Path(args.config).exists():
        cfg = load_config(args.config)
    else:
        logger.warning(f"Config not found: {args.config}, using defaults")
        cfg = {}

    seed = cfg.get("seed", 42)
    set_seed(seed)
    method = args.method or cfg.get("method", "linear_with_speed_smoothing")

    logger.info(f"Loading input: {args.input}")
    inputs = load_task_a_input(args.input)
    logger.info(f"Loaded {len(inputs)} trajectories")

    # Load training data if needed
    train_trajectories = None
    if method in {"knn_template_refinement", "local_segment_template_interpolation"}:
        train_path = args.train_data or cfg.get("train_data", "data/student_release/data_ds15/train.pkl")
        resolved_train_path = resolve_existing_path(train_path, required=False)
        if resolved_train_path is not None:
            logger.info(f"Loading training data for {method}: {resolved_train_path}")
            import numpy as np
            train_data = load_pkl(resolved_train_path)
            train_trajectories = [np.array(t["coords"], dtype=np.float64) for t in train_data]
            logger.info(f"Loaded {len(train_trajectories)} training trajectories")
        else:
            logger.warning(f"Train data not found: {train_path}, falling back to PCHIP method")
            method = "pchip_time_interpolation"

    predictions = predict_task_a(inputs, method=method, config=cfg,
                                  train_trajectories=train_trajectories)

    # Validate output
    errors = validate_task_a_output(predictions)
    if errors:
        logger.error(f"Output validation errors: {errors[:5]}")
        sys.exit(1)

    warnings = check_known_points_preserved(predictions, inputs)
    if warnings:
        logger.warning(f"Known point deviation warnings: {warnings[:3]}")

    save_pkl(predictions, args.output)
    logger.info(f"Saved predictions to: {args.output}")

    if args.mode == "val":
        if args.gt is None:
            logger.error("--gt required for val mode")
            sys.exit(1)
        gts = load_task_a_gt(args.gt)
        metrics = evaluate_task_a(predictions, gts, inputs)
        logger.info(f"Validation results:")
        logger.info(f"  MAE:  {metrics['mae_meter']:.2f} m")
        logger.info(f"  RMSE: {metrics['rmse_meter']:.2f} m")
        logger.info(f"  N points: {metrics['n_points']}")

        # Log to CSV
        input_name = Path(args.input).stem
        log_row = {
            "experiment_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "datetime": datetime.now().isoformat(),
            "method": method,
            "input_file": input_name,
            "mae_meter": f"{metrics['mae_meter']:.2f}",
            "rmse_meter": f"{metrics['rmse_meter']:.2f}",
            "config": args.config,
            "notes": "",
        }
        csv_path = Path("experiments/task_a_results.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(log_row)
        logger.info(f"Results logged to {csv_path}")


if __name__ == "__main__":
    main()
