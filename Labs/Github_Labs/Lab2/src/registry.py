import datetime
import json
from pathlib import Path

from data import DATASET_NAME
from train_model import LAB_DIR, MODEL_PARAMS

REGISTRY_PATH = LAB_DIR / "models" / "registry.json"


def load_registry(path=REGISTRY_PATH):
    """
    Load the model registry, or return an empty one if it doesn't exist yet.
    Returns:
        dict: {"latest": version or None, "versions": [entry, ...]} with the oldest entry first.
    """
    path = Path(path)
    if not path.exists():
        return {"latest": None, "versions": []}
    return json.loads(path.read_text())


def latest_entry(registry):
    """Return the registry entry of the latest promoted version, or None if nothing is promoted yet."""
    for entry in registry["versions"]:
        if entry["version"] == registry["latest"]:
            return entry
    return None


def register_model(version, model_file, metrics_file, metrics, path=REGISTRY_PATH):
    """
    Add a version to the registry and make it the latest.
    Args:
        version (str): The model version (timestamp).
        model_file (str): Model file name, relative to the models/ folder.
        metrics_file (str): Metrics file name, relative to the metrics/ folder.
        metrics (dict): Test metrics of this version.
        path (Path): Registry file to update.
    Returns:
        dict: The updated registry.
    """
    registry = load_registry(path)
    if any(entry["version"] == version for entry in registry["versions"]):
        raise ValueError(f"Version {version} is already registered")
    registry["versions"].append({
        "version": version,
        "registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "dataset": DATASET_NAME,
        "model": "GradientBoostingClassifier",
        "params": MODEL_PARAMS,
        "model_file": model_file,
        "metrics_file": metrics_file,
        "metrics": {k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
    })
    registry["latest"] = version
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=4))
    return registry
