# Docker Lab 1: Diabetes Progression Model in a Container

[![Docker Lab1 - Build, Test, Train & Push](https://github.com/isakshay007/MLOps/actions/workflows/docker_lab1_diabetes.yml/badge.svg)](https://github.com/isakshay007/MLOps/actions/workflows/docker_lab1_diabetes.yml)

Based on [Docker Lab 1 from raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Docker_Labs/Lab1).

This lab packages a machine learning model in a Docker image. The container trains a **Gradient Boosting regressor** on the scikit-learn **Diabetes dataset** to predict how far a patient's diabetes progresses in one year, from 10 baseline measurements such as BMI, blood pressure and blood sugar. The trained model and its metrics are saved to a folder mounted from the host, so they survive after the container exits. The same image then serves predictions for new patients.

---

## My modifications

| Area | Original lab | This version |
|---|---|---|
| **Dataset** | Iris (150 flowers, 4 features, classify the species) | **Diabetes** (442 patients, 10 measurements in real units, predict disease progression from 25 to 346). A **regression** task instead of classification |
| **Model** | `RandomForestClassifier(n_estimators=100)` | **`GradientBoostingRegressor`** (200 trees, learning rate 0.03, depth 2, subsample 0.8), chosen by 5-fold cross-validation |
| **Evaluation** | None; prints "The model training was successful" | Test-set **RMSE, MAE and R²**, compared with a baseline that always predicts the average, plus 5-fold CV R² and the top 5 features, saved to `metrics.json` |
| **Where the model goes** | `iris_model.pkl` inside the container, lost when it exits | `model.joblib` + `metrics.json` written to `/app/output`, a **volume** you mount from the host (`-v ./output:/app/output`) |
| **What the container does** | Trains only | Two commands in one image: **`train`** and **`predict`** (one patient or a batch, as JSON or from stdin) |
| **Settings** | Hard-coded | **Environment variables** (`N_ESTIMATORS`, `LEARNING_RATE`, `MAX_DEPTH`, `SUBSAMPLE`, `TEST_SIZE`, `RANDOM_STATE`, `OUTPUT_DIR`), so retraining with other settings needs no rebuild. Bad values are rejected with the variable's name |
| **Base image** | `python:3.10` (full Debian image) | **`python:3.12-slim`**, multi-stage build: packages are installed in a builder stage and only the results are copied in |
| **Image size** | **2.06 GB** | **545 MB** (73% smaller) |
| **Build cache** | Code and requirements copied together, so any code change reinstalls every package | Requirements copied and installed **before** the code, so a code change rebuilds in seconds |
| **Security** | Runs as `root` | Runs as a **non-root user** (`appuser`, uid 1000) |
| **Reproducibility** | `scikit-learn` and `joblib` unpinned | All packages **pinned** (scikit-learn 1.5.1, joblib 1.4.2, numpy 2.1.3, scipy 1.14.1), fixed random seed |
| **Build context** | Everything in the folder is sent to Docker | **`.dockerignore`** leaves out caches, outputs, docs and image tarballs |
| **Tests** | None | **24 pytest tests** run in a **test stage** of the Dockerfile. The final image depends on that stage, so a failing test stops `docker build` and no image is produced. pytest is not in the final image |
| **Input checks** | – | `predict` rejects invalid JSON, missing or unknown features and non-numeric values, with a clear message and exit code 2. Without a trained model, it says to run `train` first |
| **Image metadata** | None | OCI labels (title, description, source repo, license) |
| **CI/CD** | None | GitHub Actions builds the image (running the tests), trains in the container and **fails if test R² < 0.40**, runs a prediction, uploads `metrics.json`, and **pushes the image to GitHub Container Registry** |
| **File names** | `dockerfile`, `ReadMe.md`, `src/requirements.txt` | `Dockerfile`, `README.md`, `requirements.txt` + `requirements-test.txt` at the lab root |

---

## Results

Measured on Nov 5, 2026: locally on Docker 29.8 (Apple Silicon), and in GitHub Actions ([run 42130651848](https://github.com/isakshay007/MLOps/actions/runs/42130651848), passed in 56 s). Both gave exactly the same metrics and prediction.

### Model performance (held-out test set, 89 patients)

| Metric | Gradient Boosting | Baseline (always predict the average) |
|---|---|---|
| RMSE | **53.05** | 73.22 |
| MAE | **42.94** | 64.01 |
| R² | **0.469** | −0.012 |
| 5-fold CV R² (training set) | **0.429 ± 0.144** | – |

The model's error is **28% lower** than the baseline's (RMSE 53.05 vs 73.22). The most important features are BMI (0.38), log of serum triglycerides `s5` (0.29) and blood pressure (0.11), which matches what the dataset is known for. An R² of about 0.47 is typical for this dataset: 10 baseline measurements explain only part of how the disease develops.

Example prediction for the first patient in the dataset (actual progression 151):

```
$ docker run --rm -v "$PWD/output:/app/output" docker-lab1 predict '{"age": 59, "sex": 2, "bmi": 32.1, "bp": 101, "s1": 157, "s2": 93.2, "s3": 38, "s4": 4, "s5": 4.8598, "s6": 87}'
{"predicted_progression": 210.8}
```

### Docker checks

| Check | Result |
|---|---|
| Image size (`docker image ls`) | Original **2.06 GB** → this version **545 MB** |
| `docker build` (test stage) | 24 passed |
| A test made to fail on purpose | `docker build` stopped at the test stage, **no image produced** |
| Rebuild after a code-only change | Both `pip install` layers reused from cache |
| `docker run ... train --min-r2 0.40` | `model.joblib` + `metrics.json` written to the mounted host folder, exit code 0 |
| `predict` with one patient / two patients from stdin | `210.8` / `[210.8, 82.4]` |
| `-e N_ESTIMATORS=400 -e LEARNING_RATE=0.02` | Settings applied without a rebuild |
| `-e N_ESTIMATORS=abc` | `Error: N_ESTIMATORS='abc' is not a valid int`, exit code 2 |
| `predict` with missing features | Lists the missing features, exit code 2 |
| `predict` without a mounted model | Says to run `train` first, exit code 2 |
| User inside the container (`id`) | `uid=1000(appuser)`, not root |
| `import pytest` in the final image | `ModuleNotFoundError` (test tools stay out of the runtime image) |

### CI and registry

| Step in [run 42130651848](https://github.com/isakshay007/MLOps/actions/runs/42130651848) | Result |
|---|---|
| Build image (test stage) | 24 passed |
| Image size on the runner (`linux/amd64`) | 554 MB on disk, **130 MB** to download |
| Train with `--min-r2 0.40` | RMSE 53.05, MAE 42.94, R² 0.4688 (passed) |
| Predict for the sample patient | `210.8` |
| Push to GHCR | `ghcr.io/isakshay007/mlops-docker-lab1:latest` and `:sha-8e549d3` |
| Anonymous `docker pull` of `:latest`, then `train` + `predict` | Same metrics and prediction; runs as `appuser`; `source` label links to this repo |

---

## Folder structure

```
Lab1/
├── src/
│   ├── main.py              # CLI and container ENTRYPOINT: `train` and `predict`
│   ├── config.py            # Training settings, read from environment variables (new)
│   ├── data.py              # Loads Diabetes in original units, train/test split (new)
│   ├── train.py             # Trains, evaluates, saves model.joblib + metrics.json (new)
│   └── predict.py           # Loads the model, validates patient JSON, predicts (new)
├── tests/                   # 24 pytest tests, run inside the Docker build (new)
├── Dockerfile               # Multi-stage: builder -> base -> test -> runtime (was `dockerfile`)
├── .dockerignore            # (new)
├── .gitignore               # Keeps output/ out of git (new)
├── requirements.txt         # Pinned runtime packages (moved from src/)
├── requirements-test.txt    # pytest, only installed in the test stage (new)
└── README.md                # (was `ReadMe.md`)
```

The workflow is at the repository root in [`.github/workflows/docker_lab1_diabetes.yml`](../../../.github/workflows/docker_lab1_diabetes.yml), because GitHub only runs workflows from there.

---

## Running the lab

You need [Docker](https://www.docker.com/get-started).

```bash
git clone https://github.com/isakshay007/MLOps.git
cd MLOps/Labs/Docker_Labs/Lab1

# Build (the tests run during the build)
docker build -t docker-lab1 .

# Train; the model and metrics are written to ./output on your machine
mkdir -p output
docker run --rm -v "$PWD/output:/app/output" docker-lab1 train
cat output/metrics.json

# Predict for one patient
docker run --rm -v "$PWD/output:/app/output" docker-lab1 predict \
  '{"age": 59, "sex": 2, "bmi": 32.1, "bp": 101, "s1": 157, "s2": 93.2, "s3": 38, "s4": 4, "s5": 4.8598, "s6": 87}'

# Predict for several patients from a file (a JSON list of patients)
docker run --rm -i -v "$PWD/output:/app/output" docker-lab1 predict - < patients.json

# Retrain with other settings, no rebuild needed
docker run --rm -e N_ESTIMATORS=400 -e LEARNING_RATE=0.02 -v "$PWD/output:/app/output" docker-lab1 train
```

On **Linux**, add `--user "$(id -u):$(id -g)"` to `docker run` so the container can write to the mounted folder. Docker Desktop on macOS and Windows handles this for you.

Other useful commands:

```bash
docker build --target test -t docker-lab1:test .   # build only up to the test stage
docker save docker-lab1 > docker-lab1.tar          # export the image, as in the original lab
docker run --rm docker-lab1 --help                 # list the commands
```

### Pull the prebuilt image

CI pushes every build of `main` to GitHub Container Registry:

```bash
docker pull ghcr.io/isakshay007/mlops-docker-lab1:latest
docker run --rm -v "$PWD/output:/app/output" ghcr.io/isakshay007/mlops-docker-lab1:latest train
```

The image is built for `linux/amd64`. On Apple Silicon, add `--platform linux/amd64` to `docker pull` and `docker run`.

### Patient features

`predict` needs all 10 features, in the dataset's original units:

| Key | Meaning |
|---|---|
| `age` | Age in years |
| `sex` | Sex (1 or 2) |
| `bmi` | Body mass index |
| `bp` | Average blood pressure (mm Hg) |
| `s1` | Total serum cholesterol (tc) |
| `s2` | Low-density lipoproteins (ldl) |
| `s3` | High-density lipoproteins (hdl) |
| `s4` | Total cholesterol / HDL (tch) |
| `s5` | Log of serum triglycerides (ltg) |
| `s6` | Blood sugar level (glu) |

### Training settings

| Environment variable | Default | Meaning |
|---|---|---|
| `N_ESTIMATORS` | 200 | Number of boosting trees |
| `LEARNING_RATE` | 0.03 | Step size of each tree (0–1) |
| `MAX_DEPTH` | 2 | Depth of each tree |
| `SUBSAMPLE` | 0.8 | Fraction of patients used for each tree (0–1) |
| `TEST_SIZE` | 0.2 | Fraction of patients held out for testing |
| `RANDOM_STATE` | 42 | Seed for the split and the model |
| `OUTPUT_DIR` | `/app/output` | Where the model and metrics are written |

---

## How the CI workflow runs

On every push to `main` that changes this lab (README-only edits are skipped), on pull requests, and on **Run workflow** in the Actions tab:

1. **Build** the image. The test stage runs the 24 tests, and a failure stops the workflow.
2. **Show** the image size.
3. **Train** in the container with `--min-r2 0.40`. If the test R² is below 0.40, the step fails.
4. **Predict** for a sample patient with the saved model.
5. **Upload** `metrics.json` and the build log as a workflow artifact.
6. **Push** the image to `ghcr.io/isakshay007/mlops-docker-lab1`, tagged `latest` and `sha-<commit>` (only for `main`, not pull requests).
