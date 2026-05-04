"""Task B entry point."""
from __future__ import annotations
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from src.common.config import load_config
from src.common.io import load_pkl, save_pkl, validate_task_b_output
from src.common.logging_utils import get_logger
from src.common.paths import first_existing_path, resolve_existing_path
from src.common.seed import set_seed
from src.task_b.dataset import load_task_b_input, load_task_b_gt, load_train_data
from src.task_b.models import GlobalSpeedModel, TimeBucketSpeedModel, EnsembleModel
from src.task_b.predict import predict_task_b
from src.task_b.evaluate import evaluate_task_b

logger = get_logger("task_b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task B: Travel Time Estimation")
    p.add_argument("--input", required=True, help="Input pkl file")
    p.add_argument("--output", required=True, help="Output pkl file")
    p.add_argument("--gt", default=None, help="Ground truth pkl (val mode)")
    p.add_argument("--model-path", default="outputs/task_b/best_model.pkl",
                   help="Path to save/load trained model")
    p.add_argument("--config", default="configs/task_b_advanced.yaml")
    p.add_argument("--mode", choices=["val", "predict", "train"], default="val")
    p.add_argument("--method", default=None,
                   choices=["global_speed_baseline", "time_bucket_speed_model",
                            "ensemble_with_speed_constraints"],
                   help="Override config method")
    return p.parse_args()


def build_model(method: str, cfg: dict, train_data: list[dict]):
    if method == "global_speed_baseline":
        model = GlobalSpeedModel().fit(train_data)
    elif method == "time_bucket_speed_model":
        model = TimeBucketSpeedModel(
            min_samples=cfg.get("speed_model", {}).get("min_samples_per_bucket", 5)
        ).fit(train_data)
    else:  # ensemble
        model = EnsembleModel(
            w_time_bucket=cfg.get("ensemble", {}).get("w_time_bucket", 0.2),
            w_regression=cfg.get("ensemble", {}).get("w_regression", 0.8),
            min_travel_time=cfg.get("constraints", {}).get("min_travel_time", 30.0),
        ).fit(train_data, cfg)
    return model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    method = args.method or cfg.get("method", "ensemble_with_speed_constraints")

    logger.info(f"Loading input: {args.input}")
    inputs = load_task_b_input(args.input)
    logger.info(f"Loaded {len(inputs)} trajectories")

    model_path = Path(args.model_path)

    resolved_model_path = resolve_existing_path(model_path, required=False)
    if resolved_model_path is None and model_path == Path("outputs/task_b/best_model.pkl"):
        resolved_model_path = first_existing_path(sorted(Path("outputs/task_b").glob("best_model*.pkl")))

    if args.mode == "predict" and method == "ensemble_with_speed_constraints" and resolved_model_path is not None:
        logger.info(f"Loading model from {resolved_model_path}")
        model = EnsembleModel.load(resolved_model_path)
    else:
        # Train model
        train_path = cfg.get("train_data", "data/student_release/data_ds15/train.pkl")
        resolved_train_path = resolve_existing_path(train_path, required=False)
        if resolved_train_path is None:
            logger.error(f"Training data not found: {train_path}")
            sys.exit(1)
        logger.info(f"Loading training data: {resolved_train_path}")
        train_data = load_train_data(resolved_train_path)
        logger.info(f"Loaded {len(train_data)} training samples")

        logger.info(f"Training model: {method}")
        model = build_model(method, cfg, train_data)

        # Save model
        if hasattr(model, "save"):
            model.save(model_path)
            logger.info(f"Model saved to {model_path}")

    predictions = predict_task_b(inputs, model)

    errors = validate_task_b_output(predictions)
    if errors:
        logger.error(f"Output validation errors: {errors[:5]}")
        sys.exit(1)

    save_pkl(predictions, args.output)
    logger.info(f"Saved predictions to: {args.output}")

    if args.mode == "val":
        if args.gt is None:
            logger.error("--gt required for val mode")
            sys.exit(1)
        gts = load_task_b_gt(args.gt)
        metrics = evaluate_task_b(predictions, gts)
        logger.info("Validation results:")
        logger.info(f"  MAE:  {metrics['mae_second']:.2f} s")
        logger.info(f"  RMSE: {metrics['rmse_second']:.2f} s")
        logger.info(f"  MAPE: {metrics['mape_percent']:.2f} %")

        log_row = {
            "experiment_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "datetime": datetime.now().isoformat(),
            "method": method,
            "mae_second": f"{metrics['mae_second']:.2f}",
            "rmse_second": f"{metrics['rmse_second']:.2f}",
            "mape_percent": f"{metrics['mape_percent']:.2f}",
            "config": args.config,
            "notes": "",
        }
        csv_path = Path("experiments/task_b_results.csv")
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
