import json
from functools import lru_cache
from pathlib import Path

import joblib

from data import CLASS_NAMES

# Paths are relative to this file, so the API works from any working directory
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "wine_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"


@lru_cache(maxsize=1)
def load_model():
    """
    Load the trained model once and reuse it (the original reloaded it on every request).
    Raises:
        FileNotFoundError: If train.py has not been run yet.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run `python train.py` first.")
    return joblib.load(MODEL_PATH)


def load_metrics():
    """Return the test-set metrics saved by train.py, or None if they don't exist."""
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


def predict_data(X):
    """
    Predict the cultivar and class probabilities for the input data.
    Args:
        X (list[list[float]]): One row of 13 features per wine sample.
    Returns:
        list[dict]: For each row, the class id, class name, confidence and all class probabilities.
    """
    model = load_model()
    results = []
    for probs in model.predict_proba(X):
        class_id = int(probs.argmax())
        results.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "confidence": round(float(probs[class_id]), 4),
            "probabilities": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)},
        })
    return results
