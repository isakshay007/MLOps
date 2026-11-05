import json
import math

import joblib

from data import FEATURE_NAMES
from train import MODEL_FILE


def load_model(output_dir):
    """Load the model saved by train(); explains how to fix it if there is none yet."""
    path = output_dir / MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run the 'train' command first "
                                f"(with the same output folder mounted).")
    return joblib.load(path)


def parse_patients(text):
    """
    Parse one patient (a JSON object) or several (a JSON list of objects) into feature rows.
    Every patient must have all 10 features as numbers; extra keys are rejected so typos are caught.
    Returns:
        list[list[float]]: One row per patient, in FEATURE_NAMES order.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Input is not valid JSON: {e.msg}") from None
    patients = data if isinstance(data, list) else [data]
    if not patients:
        raise ValueError("Input list is empty")

    rows = []
    for i, patient in enumerate(patients):
        where = f"patient {i}" if isinstance(data, list) else "patient"
        if not isinstance(patient, dict):
            raise ValueError(f"{where} must be a JSON object")
        missing = [f for f in FEATURE_NAMES if f not in patient]
        unknown = [k for k in patient if k not in FEATURE_NAMES]
        if missing or unknown:
            raise ValueError(f"{where}: missing {missing or 'none'}, unknown {unknown or 'none'}")
        row = []
        for f in FEATURE_NAMES:
            value = patient[f]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{where}: '{f}' must be a number, got {value!r}")
            row.append(float(value))
        rows.append(row)
    return rows


def predict(model, rows):
    """Predicted disease progression for each row, rounded to 1 decimal."""
    return [round(float(p), 1) for p in model.predict(rows)]
