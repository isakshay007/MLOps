import argparse
import json
import sys

from evaluate_model import metrics_path
from registry import latest_entry, load_registry, register_model
from train_model import model_path

# A new model must reach these test-set scores...
MIN_METRICS = {"f1": 0.92, "roc_auc": 0.97, "recall": 0.88}
# ...and may not score more than this below the latest promoted model on any of them.
MAX_DROP = 0.01


def check_quality(metrics, previous_metrics=None, min_metrics=MIN_METRICS, max_drop=MAX_DROP):
    """
    Decide whether a candidate model is good enough to promote.
    Args:
        metrics (dict): Test metrics of the candidate model.
        previous_metrics (dict or None): Test metrics of the latest promoted model, if any.
        min_metrics (dict): Minimum score for each metric.
        max_drop (float): Largest allowed drop compared with the previous model.
    Returns:
        list[str]: Reasons the model failed. Empty if it passed.
    """
    failures = []
    for name, minimum in min_metrics.items():
        if metrics[name] < minimum:
            failures.append(f"{name} {metrics[name]:.4f} is below the minimum {minimum:.4f}")
        if previous_metrics is not None and metrics[name] < previous_metrics[name] - max_drop:
            failures.append(f"{name} {metrics[name]:.4f} dropped more than {max_drop} "
                            f"from the latest model's {previous_metrics[name]:.4f}")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the quality gate and register the model if it passes")
    parser.add_argument("--timestamp", type=str, required=True, help="Model version to check")
    args = parser.parse_args()

    metrics = json.loads(metrics_path(args.timestamp).read_text())
    registry = load_registry()
    previous = latest_entry(registry)

    failures = check_quality(metrics, previous["metrics"] if previous else None)
    if failures:
        print(f"Quality gate FAILED for version {args.timestamp}; the model is not registered:")
        for reason in failures:
            print(f"  - {reason}")
        sys.exit(1)

    register_model(args.timestamp, model_path(args.timestamp).name,
                   metrics_path(args.timestamp).name, metrics)
    compared = f"compared with {previous['version']}" if previous else "first registered model"
    print(f"Quality gate passed ({compared}). Version {args.timestamp} is now the latest model.")
