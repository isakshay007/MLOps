# Github Lab 2: Breast Cancer Model Retraining, Quality Gate and Versioning with GitHub Actions

[![Github Lab2 - Breast Cancer Retrain, Gate & Version](https://github.com/isakshay007/MLOps/actions/workflows/github_lab2_breast_cancer_retrain.yml/badge.svg)](https://github.com/isakshay007/MLOps/actions/workflows/github_lab2_breast_cancer_retrain.yml)

Based on [Github Lab 2 from raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab2) ([tutorial video](https://youtu.be/cj5sXIMZUjQ)).

This lab uses GitHub Actions to retrain a model automatically. On every change to the lab it runs the tests, trains a **Gradient Boosting** classifier on the **Breast Cancer Wisconsin** dataset to tell malignant tumours from benign ones, and evaluates it on a held-out test set. It then runs a **quality gate**. Only a model that passes the gate is given a timestamped version, recorded in a **model registry**, and committed back to the repository.

---

## My modifications

| Area | Original lab | This version |
|---|---|---|
| **Dataset** | Synthetic `make_classification` data with a random size of 0–2000 rows (a 0-row run crashes) | **Breast Cancer Wisconsin**: 569 real tumours, 30 measurements, malignant (1) vs benign (0), stratified 80/20 split |
| **Model** | `RandomForestClassifier` with default settings | **`GradientBoostingClassifier`** (200 trees, learning rate 0.05, depth 3, subsample 0.8) |
| **Evaluation** | Regenerated the training data and scored on it, so F1 was meaningless | Held-out test set: accuracy, precision, recall, F1, ROC-AUC and confusion matrix, plus 5-fold CV ROC-AUC on the training set |
| **Quality gate** | None: every model was saved | `quality_gate.py` blocks a model whose F1 < 0.92, ROC-AUC < 0.97 or recall < 0.88, or that scores more than 0.01 below the latest model. A failed gate fails the workflow and nothing is committed |
| **Model registry** | Loose `.joblib` files | `models/registry.json` lists every promoted version with its params, metrics and files, and a `latest` pointer |
| **Experiment tracking** | MLflow run named after an unrelated dataset ("Reuters Corpus Volume") | MLflow experiment `github_lab2_breast_cancer` with params and CV scores (uploaded as a workflow artifact) |
| **Tests** | None (empty `test/` folder) | **16 pytest tests** for the data, model, metrics, quality gate and registry |
| **Workflow** | Two workflows (on push and daily). They commit as the course author, have no write permission, and the daily one has broken paths | One workflow: tests on Python 3.10–3.12, then train → evaluate → gate → commit as `github-actions[bot]`. Runs on lab changes, pull requests (no commit) and manual runs |
| **Code** | Scripts must run from the repo root; files written to the working directory and moved by the workflow | Paths built from each file's location, so the scripts run from any folder and write straight to `models/` and `metrics/` |

---

## Folder structure

```
Lab2/
├── src/
│   ├── data.py            # Loads Breast Cancer, relabels malignant = 1, stratified split
│   ├── train_model.py     # Trains Gradient Boosting, saves models/model_<version>_gb_model.joblib, logs to MLflow
│   ├── evaluate_model.py  # Test-set metrics -> metrics/<version>_metrics.json
│   ├── quality_gate.py    # Checks the metrics, registers the model if it passes (new)
│   └── registry.py        # Reads and updates models/registry.json (new)
├── test/                  # 16 pytest tests (new)
├── models/                # Versioned models + registry.json (written by the workflow)
├── metrics/               # Metrics of each version (written by the workflow)
├── workflows/             # The course's original workflow files, kept for reference
├── requirements.txt       # Pinned: scikit-learn, mlflow, pytest
└── README.md
```

The workflow that actually runs is at the repository root in [`.github/workflows/github_lab2_breast_cancer_retrain.yml`](../../../.github/workflows/github_lab2_breast_cancer_retrain.yml), because GitHub only runs workflows from there. The course's two original Lab 2 workflows are disabled in this fork.

---

## Running the lab locally

```bash
git clone https://github.com/isakshay007/MLOps.git
cd MLOps/Labs/Github_Labs/Lab2
python3 -m venv lab2_env
source lab2_env/bin/activate
pip install -r requirements.txt

pytest -v test                                   # 16 tests
python src/train_model.py --timestamp 20261029120000
python src/evaluate_model.py --timestamp 20261029120000
python src/quality_gate.py --timestamp 20261029120000
```

`--timestamp` is the model version. The workflow passes the current UTC time.

Example output:

```
5-fold CV ROC-AUC: 0.9912 ± 0.0102
Test metrics for version 20261029120000: accuracy 0.9649 | precision 1.0 | recall 0.9048 | F1 0.95 | ROC-AUC 0.9957
Quality gate passed (first registered model). Version 20261029120000 is now the latest model.
```

To see the MLflow runs: `mlflow ui --backend-store-uri mlruns` (from the `Lab2` folder).

---

## How the workflow runs

1. **Test** runs the pytest suite on Python 3.10, 3.11 and 3.12. Retraining only starts if every test passes.
2. **Train** generates a UTC timestamp as the version, trains the model and saves `models/model_<version>_gb_model.joblib`.
3. **Evaluate** scores the model on the 114-row test set and saves `metrics/<version>_metrics.json`.
4. **Quality gate** compares the metrics with the minimums and with the latest model in `models/registry.json`. If the model fails, the job fails and lists the reasons, and the model is not registered.
5. **Upload artifact** saves the model, metrics and MLflow run to the workflow run, even when the gate fails, so a rejected model can still be inspected.
6. **Commit** pushes the new model, its metrics and the updated registry back to `main` as `github-actions[bot]`. This step is skipped for pull requests.

The workflow is triggered by pushes to `main` that change this lab (the bot's own commits to `models/` and `metrics/` are ignored), by pull requests and by **Run workflow** in the Actions tab.

### Quality gate thresholds

| Metric | Minimum | Why |
|---|---|---|
| F1 (malignant) | 0.92 | Overall balance of precision and recall |
| ROC-AUC | 0.97 | How well the model ranks malignant above benign |
| Recall (malignant) | 0.88 | A missed malignant tumour is the costliest mistake |
| Drop vs latest model | ≤ 0.01 on each metric | A retrained model must not be worse than the one in use |

The thresholds are `MIN_METRICS` and `MAX_DROP` in [`src/quality_gate.py`](src/quality_gate.py).
