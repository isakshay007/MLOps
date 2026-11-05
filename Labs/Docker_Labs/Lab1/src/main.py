"""
Command-line entry point, and the container's ENTRYPOINT.

    python src/main.py train [--min-r2 0.40]
    python src/main.py predict '{"age": 50, "sex": 1, "bmi": 27.5, ...}'
    python src/main.py predict - < patients.json
"""
import argparse
import json
import sys

from config import TrainConfig
from predict import load_model, parse_patients, predict
from train import train


def run_train(args):
    config = TrainConfig.from_env()
    metrics = train(config)
    test, base = metrics["test"], metrics["baseline_mean"]
    print(f"Trained GradientBoostingRegressor on {metrics['train_size']} patients "
          f"({', '.join(f'{k}={v}' for k, v in config.model_params.items())})")
    print(f"Test set ({metrics['test_size']} patients): RMSE {test['rmse']} | MAE {test['mae']} | "
          f"R² {test['r2']}   (baseline RMSE {base['rmse']})")
    print(f"5-fold CV R²: {metrics['cv5_r2_mean']} ± {metrics['cv5_r2_std']}")
    print(f"Saved model.joblib and metrics.json to {config.output_dir}")
    if args.min_r2 is not None and test["r2"] < args.min_r2:
        print(f"FAILED: test R² {test['r2']} is below the minimum {args.min_r2}", file=sys.stderr)
        return 1
    return 0


def run_predict(args):
    text = sys.stdin.read() if args.patients == "-" else args.patients
    rows = parse_patients(text)
    model = load_model(TrainConfig.from_env().output_dir)
    predictions = predict(model, rows)
    print(json.dumps({"predicted_progression": predictions[0] if len(predictions) == 1 else predictions}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diabetes progression model: train or predict")
    commands = parser.add_subparsers(dest="command", required=True)

    train_cmd = commands.add_parser("train", help="Train, evaluate and save the model")
    train_cmd.add_argument("--min-r2", type=float, default=None,
                           help="Exit with code 1 if the test R² is below this value")
    train_cmd.set_defaults(func=run_train)

    predict_cmd = commands.add_parser("predict", help="Predict disease progression for patients")
    predict_cmd.add_argument("patients", help="A JSON object or list of objects, or '-' to read from stdin")
    predict_cmd.set_defaults(func=run_predict)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
