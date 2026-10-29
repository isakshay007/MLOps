import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from data import load_data, split_data
from train_model import LAB_DIR, model_path

METRICS_DIR = LAB_DIR / "metrics"


def metrics_path(version, metrics_dir=METRICS_DIR):
    """Return the file path of a version's metrics, e.g. metrics/20261029120000_metrics.json."""
    return Path(metrics_dir) / f"{version}_metrics.json"


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the held-out test set. Malignant (1) is the positive class.
    Args:
        model: A fitted classifier with predict and predict_proba.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test target values.
    Returns:
        dict: Accuracy, precision, recall, F1, ROC-AUC and the confusion matrix.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "test_size": len(y_test),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        # Rows are the true class, columns the prediction: [[TN, FP], [FN, TP]]
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def save_metrics(metrics, version, metrics_dir=METRICS_DIR):
    """Save the metrics as JSON under the version name and return the path."""
    path = metrics_path(version, metrics_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=4))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Model version to evaluate")
    args = parser.parse_args()

    path = model_path(args.timestamp)
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run train_model.py --timestamp {args.timestamp} first.")
    model = joblib.load(path)

    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)
    metrics = evaluate_model(model, X_test, y_test)
    saved = save_metrics(metrics, args.timestamp)

    print(f"Test metrics for version {args.timestamp}: accuracy {metrics['accuracy']} | "
          f"precision {metrics['precision']} | recall {metrics['recall']} | "
          f"F1 {metrics['f1']} | ROC-AUC {metrics['roc_auc']}")
    print(f"Saved metrics to {saved}")
