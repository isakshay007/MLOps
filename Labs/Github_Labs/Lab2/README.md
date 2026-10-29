# Github Lab 2: Breast Cancer Model Retraining, Quality Gate and Versioning with GitHub Actions

[![Github Lab2 - Breast Cancer Retrain, Gate & Version](https://github.com/isakshay007/MLOps/actions/workflows/github_lab2_breast_cancer_retrain.yml/badge.svg)](https://github.com/isakshay007/MLOps/actions/workflows/github_lab2_breast_cancer_retrain.yml)

Based on [Github Lab 2 from raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab2).

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
| **Tests** | None (empty `test/` folder) | **17 pytest tests** for the data, model, metrics, quality gate and registry |
| **Workflow** | Two workflows (on push and daily). They commit as the course author, have no write permission, and the daily one has broken paths | One workflow: tests on Python 3.10–3.12, then train → evaluate → gate → commit as `github-actions[bot]`. Runs on lab changes, pull requests (no commit) and manual runs |
| **Code** | Scripts must run from the repo root; files written to the working directory and moved by the workflow | Paths built from each file's location, so the scripts run from any folder and write straight to `models/` and `metrics/` |
| **Leftovers** | Broken scratch notebook (`src/test.ipynb`, Reuters data) and copies of the workflows in `workflows/` | Removed; the single workflow lives in `.github/workflows/` |

---

## Results

Tested end to end on Oct 29, 2026, both in GitHub Actions and in a fresh local clone (clean virtual environment, installed from `requirements.txt`).

### Model performance (held-out test set, 114 tumours)

| Metric | Score | Quality gate minimum |
|---|---|---|
| Accuracy | **0.9649** | – |
| Precision (malignant) | **1.0000** | – |
| Recall (malignant) | **0.9048** | 0.88 |
| F1 (malignant) | **0.9500** | 0.92 |
| ROC-AUC | **0.9964** | 0.97 |
| 5-fold CV ROC-AUC (training set) | **0.9904 ± 0.0114** | – |

Confusion matrix:

| | Predicted benign | Predicted malignant |
|---|---|---|
| **Actually benign (72)** | 72 | 0 |
| **Actually malignant (42)** | 4 | 38 |

The model raised **no false alarms**: every tumour it called malignant was malignant. It missed 4 of the 42 malignant tumours, which is why recall is the gated metric with the lowest minimum, and the one to improve next.

These scores come from the GitHub Actions runner (Linux). On macOS the predictions and confusion matrix are identical, but ROC-AUC is 0.9957 and CV ROC-AUC 0.9912 ± 0.0102. Small floating-point differences between platforms change the probability scores slightly.

### GitHub Actions runs

| Run | Trigger | Tests (3.10 / 3.11 / 3.12) | Quality gate | Version committed by `github-actions[bot]` |
|---|---|---|---|---|
| [40955988731](https://github.com/isakshay007/MLOps/actions/runs/40955988731) | Push of the lab | 16 / 16 / 16 passed | Passed (first registered model) | `20261029211551` (\*) |
| [40956479339](https://github.com/isakshay007/MLOps/actions/runs/40956479339) | Push of the `.gitignore` fix | 16 / 16 / 16 passed | Passed (compared with `20261029211551`) | `20261029211854` |
| [40957613380](https://github.com/isakshay007/MLOps/actions/runs/40957613380) | Push of the float fix and `*.md` filter | 17 / 17 / 17 passed | Passed (compared with `20261029211854`) | `20261029212704` (latest) |

(\*) In the first run the bot committed only the model file. The repo-wide `.gitignore` ignores `*.json`, so `registry.json` and the metrics file were left out. The fix was a `.gitignore` in this lab that lets them through, plus a workflow step that fails if any of the three files is missing from the commit. The missing files for this version were restored from the run's artifact. Since the fix, every version has its model, metrics and registry entry in the repo.

All three registered versions have the same scores. The data and random seed are fixed, so retraining reproduces the same model, and the gate accepts it because it is no worse than the latest version.

### Quality gate rejection test

To check that the gate blocks a bad model, a deliberately weak model (Gradient Boosting with 5 trees of depth 1) was evaluated and sent through `quality_gate.py`. The test ran locally, so the "latest model" is the macOS run with ROC-AUC 0.9957:

```
Quality gate FAILED for version weak; the model is not registered:
  - f1 0.8000 is below the minimum 0.9200
  - f1 0.8000 dropped more than 0.01 from the latest model's 0.9500
  - roc_auc 0.9636 is below the minimum 0.9700
  - roc_auc 0.9636 dropped more than 0.01 from the latest model's 0.9957
  - recall 0.6667 is below the minimum 0.8800
  - recall 0.6667 dropped more than 0.01 from the latest model's 0.9048
```

The script exited with code 1, which fails the workflow, and `registry.json` kept its previous `latest` version.

### Fresh-clone end-to-end check

| Step | Result |
|---|---|
| `git clone` + `pip install -r requirements.txt` (Python 3.12) | Installed cleanly |
| `pytest -v test` | 17 passed |
| `train_model.py` → `evaluate_model.py` → `quality_gate.py`, run from the **repo root** | Model, metrics and registry entry written inside the lab folder; gate passed against the committed latest version |
| Weak model through `quality_gate.py` | Rejected, exit code 1, registry unchanged |
| MLflow (`mlruns/`) | Run `gb_<version>` logged in experiment `github_lab2_breast_cancer` with the model params, dataset name and CV ROC-AUC |

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
├── test/                  # 17 pytest tests (new)
├── models/                # Versioned models + registry.json (written by the workflow)
├── metrics/               # Metrics of each version (written by the workflow)
├── .gitignore             # Lets registry.json and metrics/*.json past the repo-wide *.json ignore
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

pytest -v test                                   # 17 tests
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
