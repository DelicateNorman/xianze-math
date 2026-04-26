"""Task B models: speed baselines + ML regression."""
from __future__ import annotations
import numpy as np
import pickle
from pathlib import Path
from src.common.geo import trajectory_length
from src.task_b.features import build_feature_matrix, GRID_SIZE, _grid_xy
from src.common.time_features import departure_hour


class GlobalSpeedModel:
    """Predict travel_time = total_distance / global_mean_speed."""

    def __init__(self) -> None:
        self.mean_speed_ms: float = 10.0  # fallback ~36 km/h

    def fit(self, train_data: list[dict]) -> "GlobalSpeedModel":
        speeds = []
        for item in train_data:
            dist = trajectory_length(item["coords"])
            tt = item["travel_time"]
            if tt > 0 and dist > 0:
                speeds.append(dist / tt)
        if speeds:
            # Use median to be robust to outliers
            self.mean_speed_ms = float(np.median(speeds))
        return self

    def predict(self, items: list[dict]) -> np.ndarray:
        preds = []
        for item in items:
            dist = trajectory_length(item["coords"])
            preds.append(dist / self.mean_speed_ms)
        return np.array(preds)


class TimeBucketSpeedModel:
    """Per-hour speed model with fallback to global speed."""

    def __init__(self, min_samples: int = 5) -> None:
        self.min_samples = min_samples
        self.hour_speed: dict[int, float] = {}
        self.global_speed: float = 10.0

    def fit(self, train_data: list[dict]) -> "TimeBucketSpeedModel":
        buckets: dict[int, list[float]] = {h: [] for h in range(24)}
        all_speeds = []
        for item in train_data:
            dist = trajectory_length(item["coords"])
            tt = item["travel_time"]
            if tt <= 0 or dist <= 0:
                continue
            spd = dist / tt
            all_speeds.append(spd)
            h = departure_hour(item["departure_timestamp"])
            buckets[h].append(spd)

        if all_speeds:
            self.global_speed = float(np.median(all_speeds))

        for h, speeds in buckets.items():
            if len(speeds) >= self.min_samples:
                self.hour_speed[h] = float(np.median(speeds))
            else:
                self.hour_speed[h] = self.global_speed

        return self

    def predict(self, items: list[dict]) -> np.ndarray:
        preds = []
        for item in items:
            dist = trajectory_length(item["coords"])
            h = departure_hour(item["departure_timestamp"])
            speed = self.hour_speed.get(h, self.global_speed)
            preds.append(dist / speed)
        return np.array(preds)


class RegressionModel:
    """Wrap sklearn regression models for TTE."""

    def __init__(self, model_type: str = "gradient_boosting", **kwargs) -> None:
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.feature_names: list[str] = []

    def _build_model(self):
        from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                                       HistGradientBoostingRegressor)
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        seed = self.kwargs.get("random_state", 42)
        if self.model_type == "hist_gradient_boosting":
            return HistGradientBoostingRegressor(
                max_iter=self.kwargs.get("n_estimators", 500),
                max_depth=self.kwargs.get("max_depth", 6),
                learning_rate=self.kwargs.get("learning_rate", 0.05),
                random_state=seed,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=self.kwargs.get("n_estimators", 200),
                max_depth=self.kwargs.get("max_depth", 5),
                learning_rate=self.kwargs.get("learning_rate", 0.05),
                random_state=seed,
            )
        elif self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.kwargs.get("n_estimators", 200),
                max_depth=self.kwargs.get("max_depth", None),
                random_state=seed,
                n_jobs=-1,
            )
        elif self.model_type == "ridge":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.kwargs.get("alpha", 1.0))),
            ])
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, items: list[dict], targets: np.ndarray) -> "RegressionModel":
        X, self.feature_names = build_feature_matrix(items)
        self.model = self._build_model()
        self.model.fit(X, targets)
        return self

    def predict(self, items: list[dict]) -> np.ndarray:
        X, _ = build_feature_matrix(items)
        return self.model.predict(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "RegressionModel":
        with open(path, "rb") as f:
            return pickle.load(f)

    def feature_importance(self) -> dict[str, float] | None:
        m = self.model
        # Unwrap pipeline if needed
        if hasattr(m, "named_steps"):
            m = list(m.named_steps.values())[-1]
        if hasattr(m, "feature_importances_"):
            return dict(zip(self.feature_names, m.feature_importances_))
        if hasattr(m, "coef_"):
            return dict(zip(self.feature_names, np.abs(m.coef_)))
        return None


class EnsembleModel:
    """Ensemble: w_time_bucket * TimeBucket + w_regression * Regression."""

    def __init__(self, w_time_bucket: float = 0.2, w_regression: float = 0.8,
                 min_travel_time: float = 30.0) -> None:
        self.w_time_bucket = w_time_bucket
        self.w_regression = w_regression
        self.min_travel_time = min_travel_time
        self.time_bucket_model: TimeBucketSpeedModel | None = None
        self.regression_model: RegressionModel | None = None

    def fit(self, train_data: list[dict], config: dict | None = None) -> "EnsembleModel":
        config = config or {}
        reg_cfg = config.get("regression", {})
        residual = config.get("residual_learning", {}).get("enabled", True)

        self.time_bucket_model = TimeBucketSpeedModel(
            min_samples=config.get("speed_model", {}).get("min_samples_per_bucket", 5)
        ).fit(train_data)

        targets = np.array([item["travel_time"] for item in train_data], dtype=np.float64)

        self.regression_model = RegressionModel(
            model_type=reg_cfg.get("model", "gradient_boosting"),
            n_estimators=reg_cfg.get("n_estimators", 200),
            max_depth=reg_cfg.get("max_depth", 5),
            learning_rate=reg_cfg.get("learning_rate", 0.05),
            random_state=reg_cfg.get("random_state", 42),
        )

        if residual:
            # Train on residuals from time_bucket
            baseline = self.time_bucket_model.predict(train_data)
            residuals = targets - baseline
            self.regression_model.fit(train_data, residuals)
            self._residual_mode = True
        else:
            self.regression_model.fit(train_data, targets)
            self._residual_mode = False

        return self

    def predict(self, items: list[dict]) -> np.ndarray:
        tb_preds = self.time_bucket_model.predict(items)
        reg_preds = self.regression_model.predict(items)

        if self._residual_mode:
            final = tb_preds + reg_preds
        else:
            final = self.w_time_bucket * tb_preds + self.w_regression * reg_preds

        final = np.maximum(final, self.min_travel_time)
        return final

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "EnsembleModel":
        with open(path, "rb") as f:
            return pickle.load(f)
