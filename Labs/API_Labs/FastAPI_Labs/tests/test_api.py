import pytest
from fastapi.testclient import TestClient

from data import load_data, split_data
from main import EXAMPLE_WINE, app
from predict import load_metrics

client = TestClient(app)

# Real samples from the dataset, one per cultivar
CULTIVAR_2 = {**EXAMPLE_WINE, "alcohol": 12.37, "malic_acid": 0.94, "ash": 1.36, "alcalinity_of_ash": 10.6,
              "magnesium": 88.0, "total_phenols": 1.98, "flavanoids": 0.57, "nonflavanoid_phenols": 0.28,
              "proanthocyanins": 0.42, "color_intensity": 1.95, "hue": 1.05,
              "od280_od315_of_diluted_wines": 1.82, "proline": 520.0}
CULTIVAR_3 = {**EXAMPLE_WINE, "alcohol": 12.86, "malic_acid": 1.35, "ash": 2.32, "alcalinity_of_ash": 18.0,
              "magnesium": 122.0, "total_phenols": 1.51, "flavanoids": 1.25, "nonflavanoid_phenols": 0.21,
              "proanthocyanins": 0.94, "color_intensity": 4.1, "hue": 0.76,
              "od280_od315_of_diluted_wines": 1.29, "proline": 630.0}


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_model_info_reports_test_set_metrics():
    body = client.get("/model-info").json()
    assert body["model_type"] == "RandomForestClassifier"
    assert len(body["features"]) == 13
    assert body["classes"] == ["cultivar_1", "cultivar_2", "cultivar_3"]
    assert body["metrics"]["accuracy"] >= 0.9


@pytest.mark.parametrize("sample, expected", [
    (EXAMPLE_WINE, "cultivar_1"),
    (CULTIVAR_2, "cultivar_2"),
    (CULTIVAR_3, "cultivar_3"),
])
def test_predict_each_cultivar(sample, expected):
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    body = response.json()
    assert body["class_name"] == expected
    assert 0 < body["confidence"] <= 1
    assert body["confidence"] == max(body["probabilities"].values())
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-3)


def test_batch_predict():
    response = client.post("/predict/batch", json={"samples": [EXAMPLE_WINE, CULTIVAR_2, CULTIVAR_3]})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert [p["class_name"] for p in body["predictions"]] == ["cultivar_1", "cultivar_2", "cultivar_3"]


def test_batch_rejects_empty_list():
    assert client.post("/predict/batch", json={"samples": []}).status_code == 422


def test_rejects_negative_value():
    response = client.post("/predict", json={**EXAMPLE_WINE, "alcohol": -5})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "alcohol"]


def test_rejects_out_of_range_value():
    # 1400 instead of 14.0, a likely typo
    assert client.post("/predict", json={**EXAMPLE_WINE, "alcohol": 1400}).status_code == 422


def test_rejects_missing_field():
    sample = {k: v for k, v in EXAMPLE_WINE.items() if k != "proline"}
    assert client.post("/predict", json=sample).status_code == 422


def test_rejects_non_numeric_value():
    assert client.post("/predict", json={**EXAMPLE_WINE, "hue": "high"}).status_code == 422


def test_split_is_stratified_70_30():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_test) == 54
    assert sorted(set(y_test)) == [0, 1, 2]


def test_saved_metrics_match_api():
    assert client.get("/model-info").json()["metrics"] == load_metrics()
