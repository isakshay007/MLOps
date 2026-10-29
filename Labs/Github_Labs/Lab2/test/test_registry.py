import pytest

from registry import latest_entry, load_registry, register_model

METRICS = {"accuracy": 0.96, "precision": 1.0, "recall": 0.9, "f1": 0.95, "roc_auc": 0.99,
           "test_size": 114, "confusion_matrix": [[72, 0], [4, 38]]}


def test_missing_registry_is_empty(tmp_path):
    registry = load_registry(tmp_path / "registry.json")
    assert registry == {"latest": None, "versions": []}
    assert latest_entry(registry) is None


def test_register_sets_latest(tmp_path):
    path = tmp_path / "registry.json"
    register_model("v1", "model_v1_gb_model.joblib", "v1_metrics.json", METRICS, path)
    register_model("v2", "model_v2_gb_model.joblib", "v2_metrics.json", METRICS, path)

    registry = load_registry(path)
    assert registry["latest"] == "v2"
    assert [e["version"] for e in registry["versions"]] == ["v1", "v2"]
    entry = latest_entry(registry)
    assert entry["model_file"] == "model_v2_gb_model.joblib"
    # Only the headline scores are copied; the full metrics stay in the metrics file
    assert set(entry["metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_duplicate_version_rejected(tmp_path):
    path = tmp_path / "registry.json"
    register_model("v1", "m.joblib", "m.json", METRICS, path)
    with pytest.raises(ValueError):
        register_model("v1", "m.joblib", "m.json", METRICS, path)
