import json

import pytest

from conftest import PATIENT
from main import main
from predict import load_model, parse_patients, predict


def test_parse_single_patient_in_feature_order():
    shuffled = dict(reversed(list(PATIENT.items())))
    assert parse_patients(json.dumps(shuffled)) == [[float(v) for v in PATIENT.values()]]


def test_parse_list_of_patients():
    assert len(parse_patients(json.dumps([PATIENT, PATIENT]))) == 2


@pytest.mark.parametrize("bad, message", [
    ("not json", "not valid JSON"),
    ("[]", "empty"),
    ('"text"', "must be a JSON object"),
    (json.dumps({k: v for k, v in PATIENT.items() if k != "bmi"}), "missing \\['bmi'\\]"),
    (json.dumps({**PATIENT, "weight": 80}), "unknown \\['weight'\\]"),
    (json.dumps({**PATIENT, "age": "fifty"}), "'age' must be a number"),
    (json.dumps({**PATIENT, "sex": True}), "'sex' must be a number"),
])
def test_parse_rejects_bad_input(bad, message):
    with pytest.raises(ValueError, match=message):
        parse_patients(bad)


def test_predict_in_target_range(trained):
    config, _ = trained
    [prediction] = predict(load_model(config.output_dir), parse_patients(json.dumps(PATIENT)))
    assert 25 <= prediction <= 346


def test_missing_model_explains_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run the 'train' command first"):
        load_model(tmp_path)


def test_cli_train_then_predict(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("N_ESTIMATORS", "50")
    assert main(["train", "--min-r2", "0.0"]) == 0
    assert json.loads((tmp_path / "metrics.json").read_text())["config"]["n_estimators"] == 50

    capsys.readouterr()
    assert main(["predict", json.dumps([PATIENT, PATIENT])]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["predicted_progression"]) == 2


def test_cli_min_r2_gate_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("N_ESTIMATORS", "5")
    assert main(["train", "--min-r2", "0.99"]) == 1


def test_cli_bad_input_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    assert main(["predict", "{}"]) == 2
    assert "missing" in capsys.readouterr().err
