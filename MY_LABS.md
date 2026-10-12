# My Lab Submissions (IE-7374 MLOps)

Fork of [raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps). Each lab is worked on in place, at the same path as the course repo, with my own modifications on top. Submit the lab folder link from the table below.

| Lab | Status | Submission link | My modifications |
|-----|--------|-----------------|------------------|
| Github_Labs / Lab1 | Done (graded) | [Labs/Github_Labs/Lab1](https://github.com/isakshay007/MLOps/tree/main/Labs/Github_Labs/Lab1) | Banking & loan calculator with `loans.csv` dataset, 26 new pytest + 7 new unittest cases, fixed CI workflows running on Python 3.10–3.12 |
| Lab 2 | Not started | | |

## Workflow per lab

1. Sync with the course repo: `git fetch upstream && git merge upstream/main`
2. Work inside the lab's own folder only, e.g. `Labs/<Category>/<LabN>/`
3. Make it different from the original (dataset, model, features, tests, dashboard) and list the changes in that lab's `README.md`
4. Any GitHub Actions workflow goes in `.github/workflows/` at the repo root, named `labN_<topic>_*.yml`, with a `paths:` filter on the lab folder
5. Commit, push, and submit the folder link
