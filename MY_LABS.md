# My Lab Submissions (IE-7374 MLOps)

Fork of [raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps). Each lab is worked on in place, at the same path as the course repo, with my own modifications on top. Submit the lab folder link from the table below.

| Assignment | Lab | Status | Submission link | My modifications |
|------------|-----|--------|-----------------|------------------|
| Lab Assignment 1 | Github_Labs / Lab1 | Graded: full marks (feedback: submit lab folder links from the full course repo) | [Labs/Github_Labs/Lab1](https://github.com/isakshay007/MLOps/tree/main/Labs/Github_Labs/Lab1) | Banking & loan calculator with `loans.csv` dataset, 26 new pytest + 7 new unittest cases, fixed CI workflows running on Python 3.10–3.12 |
| Lab Assignment 2 | API_Labs / FastAPI_Labs | Submitted Oct 18, 2026 | [Labs/API_Labs/FastAPI_Labs](https://github.com/isakshay007/MLOps/tree/main/Labs/API_Labs/FastAPI_Labs) | Wine dataset + Random Forest instead of Iris + Decision Tree; class name, confidence and probabilities in responses; `/model-info` and `/predict/batch`; input range validation; saved test metrics + 5-fold CV; 13 API tests; CI on Python 3.10–3.12 (passing) |
| Lab Assignment 3 | Github_Labs / Lab2 | In progress | [Labs/Github_Labs/Lab2](https://github.com/isakshay007/MLOps/tree/main/Labs/Github_Labs/Lab2) | Breast Cancer Wisconsin + Gradient Boosting instead of synthetic data + Random Forest; held-out test metrics (F1, recall, ROC-AUC); quality gate that blocks worse models; `models/registry.json` model registry; MLflow tracking; 16 pytest tests; one working retrain workflow (tests on Python 3.10–3.12, commits passing versions as `github-actions[bot]`) |

## Workflow per lab

1. Sync with the course repo: `git fetch upstream && git merge upstream/main`
2. Work inside the lab's own folder only, e.g. `Labs/<Category>/<LabN>/`
3. Make it different from the original (dataset, model, features, tests, dashboard) and list the changes in that lab's `README.md`
4. Any GitHub Actions workflow goes in `.github/workflows/` at the repo root, named after the lab (e.g. `lab1_banking_*.yml`, `fastapi_lab_wine_tests.yml`), with a `paths:` filter on the lab folder
5. Commit, push, and submit the folder link
