import argparse
import datetime
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from data import DATASET_NAME, RANDOM_STATE, load_data, split_data

# Paths are relative to this file, so the script works from any working directory
LAB_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = LAB_DIR / "models"
MLRUNS_DIR = LAB_DIR / "mlruns"

MODEL_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
}


def model_path(version, models_dir=MODELS_DIR):
    """Return the file path of a model version, e.g. models/model_20261029120000_gb_model.joblib."""
    return Path(models_dir) / f"model_{version}_gb_model.joblib"


def fit_model(X_train, y_train):
    """
    Train a Gradient Boosting classifier.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    Returns:
        GradientBoostingClassifier: The fitted model.
    """
    model = GradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    return model


def save_model(model, version, models_dir=MODELS_DIR):
    """Save the model under its version name and return the path."""
    path = model_path(version, models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def log_to_mlflow(version, n_train, cv_scores):
    """Log the training run (params and 5-fold CV ROC-AUC) to the local MLflow store in mlruns/."""
    import mlflow

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("github_lab2_breast_cancer")
    with mlflow.start_run(run_name=f"gb_{version}"):
        mlflow.log_params({**MODEL_PARAMS, "dataset": DATASET_NAME, "train_rows": n_train})
        mlflow.log_metrics({"cv5_roc_auc_mean": cv_scores.mean(), "cv5_roc_auc_std": cv_scores.std()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str,
                        default=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S"),
                        help="Model version, passed in by GitHub Actions (defaults to the current UTC time)")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)
    path = save_model(model, args.timestamp)

    cv_scores = cross_val_score(GradientBoostingClassifier(**MODEL_PARAMS), X_train, y_train,
                                cv=5, scoring="roc_auc")
    if not args.no_mlflow:
        log_to_mlflow(args.timestamp, len(y_train), cv_scores)

    print(f"Trained model version {args.timestamp} on {len(y_train)} rows of {DATASET_NAME}")
    print(f"5-fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Saved model to {path}")
