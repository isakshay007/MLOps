import sys
from pathlib import Path

import pytest

DAGS_DIR = Path(__file__).resolve().parent.parent / "dags"
# Airflow puts the dags folder on sys.path, which is how the DAG imports `src.penguins`
sys.path.insert(0, str(DAGS_DIR))

from src import penguins  # noqa: E402


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    """Run every step once on the real data, in the DAG's order; returns each step's result."""
    out = tmp_path_factory.mktemp("run")
    prep = penguins.preprocess(out)
    selection = penguins.select_k(prep["clean_path"])
    model_path = penguins.train_model(prep["clean_path"], selection["best_k"], out)
    metrics = penguins.evaluate_model(model_path, prep["clean_path"], out)
    predictions = penguins.predict_new(model_path, metrics["cluster_species"], out)
    return {"out": out, "prep": prep, "selection": selection, "model_path": model_path,
            "metrics": metrics, "predictions": predictions}
