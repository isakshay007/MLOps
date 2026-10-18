import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score

from data import CLASS_NAMES, FEATURE_NAMES, load_data, split_data

# Paths are relative to this file, so training works from any working directory
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "wine_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"


def fit_model(X_train, y_train):
    """
    Train a Random Forest Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    Returns:
        RandomForestClassifier: The fitted model.
    """
    rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=12)
    rf_classifier.fit(X_train, y_train)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(rf_classifier, MODEL_PATH)
    return rf_classifier


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the model on the held-out test set and save the metrics to a JSON file.
    Also reports 5-fold cross-validation accuracy on the training set, since a single
    54-row test set can be optimistic.
    Args:
        model (RandomForestClassifier): The fitted model.
        X_train, y_train (numpy.ndarray): Training data, used for cross-validation.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test target values.
    Returns:
        dict: Accuracy, macro F1, confusion matrix and per-class report.
    """
    y_pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    metrics = {
        "model": type(model).__name__,
        "params": {"n_estimators": model.n_estimators, "max_depth": model.max_depth},
        "test_size": len(y_test),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "cv5_accuracy_mean": round(cv_scores.mean(), 4),
        "cv5_accuracy_std": round(cv_scores.std(), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "per_class": classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True),
        "top_features": sorted(
            ({"feature": f, "importance": round(float(i), 4)}
             for f, i in zip(FEATURE_NAMES, model.feature_importances_)),
            key=lambda d: d["importance"], reverse=True,
        )[:5],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Test accuracy: {metrics['accuracy']}  |  Macro F1: {metrics['f1_macro']}  |  "
          f"5-fold CV accuracy: {metrics['cv5_accuracy_mean']} ± {metrics['cv5_accuracy_std']}")
