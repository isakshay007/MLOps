import datetime
import json
import platform

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from config import TrainConfig
from data import DATASET_NAME, FEATURE_NAMES, load_data, split_data

MODEL_FILE = "model.joblib"
METRICS_FILE = "metrics.json"


def regression_metrics(y_true, y_pred):
    """RMSE, MAE and R² rounded for the metrics file."""
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def train(config=None):
    """
    Train a Gradient Boosting regressor, evaluate it and save the model and metrics.
    Args:
        config (TrainConfig): Training settings; defaults to TrainConfig.from_env().
    Returns:
        dict: The metrics that were written to <output_dir>/metrics.json.
    """
    config = config or TrainConfig.from_env()
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, config.test_size, config.random_state)

    model = GradientBoostingRegressor(**config.model_params)
    model.fit(X_train, y_train)
    cv_r2 = cross_val_score(GradientBoostingRegressor(**config.model_params),
                            X_train, y_train, cv=5, scoring="r2")

    metrics = {
        "dataset": DATASET_NAME,
        "model": "GradientBoostingRegressor",
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "config": config.to_dict(),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "test": regression_metrics(y_test, model.predict(X_test)),
        # Always predicting the training mean; the model has to beat this to be useful
        "baseline_mean": regression_metrics(y_test, np.full_like(y_test, y_train.mean())),
        "cv5_r2_mean": round(float(cv_r2.mean()), 4),
        "cv5_r2_std": round(float(cv_r2.std()), 4),
        "top_features": sorted(
            ({"feature": f, "importance": round(float(i), 4)}
             for f, i in zip(FEATURE_NAMES, model.feature_importances_)),
            key=lambda d: d["importance"], reverse=True,
        )[:5],
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.output_dir / MODEL_FILE)
    (config.output_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2))
    return metrics
