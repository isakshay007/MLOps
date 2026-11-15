import json

import pandas as pd
import pytest

from src import penguins


def _write(tmp_path, df):
    path = tmp_path / "penguins.csv"
    df.to_csv(path, index=False)
    return path


def test_validate_real_data():
    result = penguins.validate_data()
    assert result["rows"] == 344
    assert result["missing"]["bill_length_mm"] == 2


def test_validate_rejects_missing_column(tmp_path):
    df = pd.read_csv(penguins.RAW_DATA).drop(columns="body_mass_g")
    with pytest.raises(ValueError, match="Missing columns: \\['body_mass_g'\\]"):
        penguins.validate_data(_write(tmp_path, df))


def test_validate_rejects_too_few_rows_and_bad_values(tmp_path):
    df = pd.read_csv(penguins.RAW_DATA).head(50).copy()
    df.loc[0, "body_mass_g"] = 60000  # a typo: 60 kg penguin
    with pytest.raises(ValueError) as err:
        penguins.validate_data(_write(tmp_path, df))
    assert "only 50 rows" in str(err.value)
    assert "1 body_mass_g value(s) outside 2500-6500" in str(err.value)


def test_validate_rejects_too_many_missing(tmp_path):
    df = pd.read_csv(penguins.RAW_DATA)
    df.loc[:40, "flipper_length_mm"] = None
    with pytest.raises(ValueError, match="flipper_length_mm is .* missing"):
        penguins.validate_data(_write(tmp_path, df))


def test_preprocess_drops_incomplete_rows(pipeline):
    prep = pipeline["prep"]
    assert prep["rows"] == 342 and prep["dropped"] == 2
    assert pd.read_csv(prep["clean_path"])[penguins.FEATURES].notna().all().all()


def test_run_dir_is_path_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(penguins, "OUTPUT_DIR", tmp_path)
    path = penguins.run_dir("manual__2026-11-15T20:00:00+00:00")
    assert path.name == "manual__2026-11-15T20_00_00_00_00"
    assert path.is_dir()


def test_select_k_bic_finds_three_species(pipeline):
    selection = pipeline["selection"]
    assert selection["best_k"] == 3
    # The original lab's methods disagree with the real number of species
    assert selection["kmeans_elbow_k"] != 3
    assert [s["k"] for s in selection["scores"]] == list(range(2, 11))


def test_select_k_rejects_bad_range(pipeline):
    with pytest.raises(ValueError, match="k_min"):
        penguins.select_k(pipeline["prep"]["clean_path"], k_min=5, k_max=3)


def test_model_beats_kmeans_baseline(pipeline):
    metrics = pipeline["metrics"]
    assert metrics["gmm_ari"] > 0.9
    assert metrics["gmm_ari"] > metrics["kmeans_ari"]
    assert metrics["purity"] > 0.95
    assert sorted(metrics["cluster_species"].values()) == ["Adelie", "Chinstrap", "Gentoo"]
    assert sum(metrics["cluster_sizes"].values()) == 342


def test_metrics_survive_json_round_trip(pipeline):
    # XCom stores task results as JSON; nothing may change on the way through
    metrics = pipeline["metrics"]
    assert json.loads(json.dumps(metrics)) == metrics
    assert json.loads((pipeline["out"] / "metrics.json").read_text()) == metrics


def test_predict_new_penguins(pipeline):
    predictions = pipeline["predictions"]
    assert [p["likely_species"] for p in predictions] == ["Adelie", "Chinstrap", "Gentoo"]
    assert all(0 <= p["confidence"] <= 1 for p in predictions)


def test_model_scales_raw_measurements(pipeline):
    # The saved pipeline includes the scaler, so raw grams and millimetres go straight in
    import joblib
    model = joblib.load(pipeline["model_path"])
    gentoo = pd.DataFrame([[47.6, 15.0, 217, 5100]], columns=penguins.FEATURES)
    cluster = str(model.predict(gentoo)[0])
    assert pipeline["metrics"]["cluster_species"][cluster] == "Gentoo"


@pytest.mark.parametrize("status, banner", [("passed", "PASSED quality check"), ("below", "BELOW quality threshold")])
def test_build_report(pipeline, status, banner):
    p = pipeline
    path = penguins.build_report(p["selection"], p["metrics"], p["predictions"], p["out"], status)
    html = open(path).read()
    assert banner in html
    assert "data:image/png;base64," in html
    assert "K-Means baseline" in html
