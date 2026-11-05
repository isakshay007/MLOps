import json

from data import FEATURE_NAMES, load_data, split_data
from train import METRICS_FILE, MODEL_FILE


def test_dataset_uses_original_units():
    X, y = load_data()
    assert X.shape == (442, 10)
    assert len(FEATURE_NAMES) == 10
    ages = X[:, FEATURE_NAMES.index("age")]
    assert ages.min() == 19 and ages.max() == 79
    assert y.min() == 25 and y.max() == 346


def test_split_sizes_and_reproducible():
    X, y = load_data()
    X_train, X_test, _, _ = split_data(X, y)
    assert len(X_train) == 353 and len(X_test) == 89
    assert (split_data(X, y)[1] == X_test).all()


def test_train_writes_model_and_metrics(trained):
    config, metrics = trained
    assert (config.output_dir / MODEL_FILE).exists()
    saved = json.loads((config.output_dir / METRICS_FILE).read_text())
    assert saved == metrics
    assert saved["model"] == "GradientBoostingRegressor"
    assert saved["config"]["n_estimators"] == config.n_estimators


def test_model_beats_mean_baseline(trained):
    _, metrics = trained
    assert metrics["test"]["rmse"] < metrics["baseline_mean"]["rmse"]
    assert metrics["test"]["r2"] > 0.40
    assert metrics["baseline_mean"]["r2"] <= 0.0


def test_top_features_are_sorted(trained):
    _, metrics = trained
    importances = [f["importance"] for f in metrics["top_features"]]
    assert len(importances) == 5
    assert importances == sorted(importances, reverse=True)
