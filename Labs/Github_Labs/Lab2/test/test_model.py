import json

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from evaluate_model import evaluate_model, save_metrics
from train_model import model_path, save_model


def test_fit_model_returns_gradient_boosting(trained_model):
    assert isinstance(trained_model, GradientBoostingClassifier)
    assert trained_model.n_estimators_ == 200


def test_model_beats_quality_minimums(trained_model, splits):
    _, X_test, _, y_test = splits
    metrics = evaluate_model(trained_model, X_test, y_test)
    assert metrics["roc_auc"] >= 0.97
    assert metrics["f1"] >= 0.92


def test_evaluate_model_metrics(trained_model, splits):
    _, X_test, _, y_test = splits
    metrics = evaluate_model(trained_model, X_test, y_test)
    assert metrics["test_size"] == 114
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= metrics[name] <= 1.0
    assert np.array(metrics["confusion_matrix"]).sum() == 114


def test_save_and_reload_model(trained_model, splits, tmp_path):
    _, X_test, _, _ = splits
    path = save_model(trained_model, "20260101000000", tmp_path)
    assert path == model_path("20260101000000", tmp_path)
    assert path.name == "model_20260101000000_gb_model.joblib"
    reloaded = joblib.load(path)
    assert np.array_equal(reloaded.predict(X_test), trained_model.predict(X_test))


def test_save_metrics_writes_json(tmp_path):
    path = save_metrics({"f1": 0.95}, "20260101000000", tmp_path)
    assert path.name == "20260101000000_metrics.json"
    assert json.loads(path.read_text()) == {"f1": 0.95}
