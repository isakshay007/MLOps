# LAB1 - MLOps (IE-7374)

[![Lab1 Banking - Pytest](https://github.com/isakshay007/MLOps/actions/workflows/lab1_banking_pytest.yml/badge.svg)](https://github.com/isakshay007/MLOps/actions/workflows/lab1_banking_pytest.yml)
[![Lab1 Banking - Unittests](https://github.com/isakshay007/MLOps/actions/workflows/lab1_banking_unittest.yml/badge.svg)](https://github.com/isakshay007/MLOps/actions/workflows/lab1_banking_unittest.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

Based on [Lab1 from raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab1).

This lab focuses on 5 modules: creating a virtual environment, creating a GitHub repository, creating Python files, creating test files using pytest and unittest, and implementing GitHub Actions. This version completes all five, but replaces the basic arithmetic calculator with a **banking and loan calculator** driven by a **loan dataset**, adds a much larger test suite, and fixes and modernizes the CI workflows.

---

## Modifications made for this assignment

The assignment asks that the lab not be identical to the original repo, with my own changes such as a different dataset or model. The main modifications are:

1. **New application:** the generic arithmetic calculator (`fun1`–`fun4`) is replaced by a **banking and loan calculator** (`src/banking.py`) with simple interest, compound interest, EMI, and a full loan summary.
2. **New dataset:** the original `data/` folder is empty. This version adds **`data/loans.csv`**, a dataset of 8 loans that the code loads and analyzes end to end.
3. **New and expanded tests:** **41 tests** instead of 8, using pytest features the original only mentions (parametrize, fixtures, `pytest.raises`) and a wider set of unittest assertions.
4. **Fixed and upgraded CI/CD:** the original pytest workflow does not run (it has a typo and conflicting triggers) and uses retired action versions. Both workflows are rewritten to run on **3 Python versions** for every push and pull request, with test reports uploaded as artifacts.

The full side-by-side comparison is below.

## What I changed from the original

| Area | Original Lab1 | This version |
|---|---|---|
| **Source code** | `calculator.py`: `fun1`–`fun4` (add, subtract, multiply, sum) | New `banking.py`: simple interest, compound interest, EMI, loan summary, and CSV batch processing (`calculator.py` kept for reference) |
| **Dataset** | `data/` folder is empty | `data/loans.csv`: 8 sample loans (home, car, education, personal, business), including a 0% interest loan |
| **Input validation** | Type checks only | Type checks (rejects strings, `None`, booleans) **and** range checks (negative principal/rate, zero-month term) |
| **Pytest** | 4 tests with plain `assert` | 26 test cases using `@pytest.mark.parametrize`, `pytest.raises`, and the `tmp_path` fixture |
| **Unittest** | 4 tests with `assertEqual` | 7 new tests using `assertEqual`, `assertAlmostEqual`, `assertGreater`, `assertTrue`, `assertRaises` |
| **Total tests** | 8 | **41** (all passing) |
| **Pytest workflow** | Broken: `run-nam` typo, conflicting `branches` / `branches-ignore` filters, stray `issues`/`label` triggers | Fixed; runs on push **and** pull requests to `main` |
| **Action versions** | `checkout@v2`, `setup-python@v2`, `upload-artifact@v2` (v2 artifact action is retired) | `checkout@v7`, `setup-python@v7`, `upload-artifact@v7` |
| **Python versions** | 3.8 only | Matrix: **3.10, 3.11, 3.12** |
| **Unittest discovery** | Runs only `test.test_unittest` | `unittest discover` runs every test file |
| **CI extras** | None | pip caching, per-version test report artifacts, `fail-fast: false` |

---

## Folder structure

```
Lab1/
├── .gitignore                   # Ignores the lab_01/ venv, caches, test reports
├── requirements.txt
├── data/
│   └── loans.csv                # Loan dataset (new)
├── src/
│   ├── banking.py               # Banking & loan calculator (new)
│   └── calculator.py            # Original calculator (reference)
├── test/
│   ├── test_banking_pytest.py   # Pytest suite for banking.py (new)
│   ├── test_banking_unittest.py # Unittest suite for banking.py (new)
│   ├── test_pytest.py           # Original calculator tests
│   └── test_unittest.py         # Original calculator tests
└── workflows/                   # Original course workflow files (reference only)
```

The active workflows live at the repository root in [`.github/workflows/`](../../../.github/workflows), because GitHub only runs workflows from there.

---

## Step 1: Creating a Virtual Environment

A virtual environment isolates this project's dependencies from the global Python installation. I created one named `lab_01` inside `Lab1/`:

```bash
python3 -m venv lab_01
source lab_01/bin/activate        # macOS / Linux
# lab_01\Scripts\activate         # Windows
pip install -r requirements.txt
```

After activation, `(lab_01)` appears in the terminal prompt. The `lab_01/` folder is listed in `.gitignore`, so it is never committed.

---

## Step 2: Creating a GitHub Repository, Cloning and Folder Structure

- Forked the full course repository [`raminmohammadi/MLOps`](https://github.com/raminmohammadi/MLOps) to [`isakshay007/MLOps`](https://github.com/isakshay007/MLOps) and cloned it locally. This lab lives at `Labs/Github_Labs/Lab1`, the same path as in the course repo.
- The original Lab1 files come from the fork, so the git history shows exactly what changed afterwards.
- Kept the standard layout:
  - `data/`: datasets (now holds `loans.csv`)
  - `src/`: source code
  - `test/`: pytest and unittest test files
- Added a `.gitignore` that excludes the `lab_01/` virtual environment, `__pycache__/`, `.pytest_cache/`, and generated test reports.

Changes are pushed with the standard flow:

```bash
git add .
git commit -m "<your_commit_message>"
git push origin main
```

---

## Step 3: Creating banking.py in the src Folder

Instead of the original arithmetic calculator, `src/banking.py` contains a set of finance functions. Rates are annual percentages (e.g. `5` means 5%), and all results are rounded to 2 decimals.

| Function | Formula | Example |
|---|---|---|
| `simple_interest(principal, rate, years)` | P × R × T / 100 | `simple_interest(10000, 5, 2)` → `1000.0` |
| `compound_interest(principal, rate, years, n=1)` | P × (1 + R / 100n)^(n·T) − P | `compound_interest(10000, 10, 2)` → `2100.0` |
| `emi(principal, annual_rate, months)` | P × r × (1 + r)^n / ((1 + r)^n − 1), where r = annual_rate / 1200 | `emi(100000, 12, 12)` → `8884.88` |
| `loan_summary(principal, annual_rate, months)` | Combines `emi` into monthly EMI, total payment, and total interest | `loan_summary(100000, 12, 12)` → `{monthly_emi: 8884.88, total_payment: 106618.56, total_interest: 6618.56}` |
| `summarize_loans(path)` | Runs `loan_summary` on every row of a CSV (defaults to `data/loans.csv`) | One summary per loan |

`loan_summary` plays the same role as `fun4` in the original lab: it combines the results of the other functions.

**Validation:** non-numeric inputs, a principal ≤ 0, a negative rate, or a term ≤ 0 all raise `ValueError`. A 0% loan is handled without dividing by zero (EMI = principal ÷ months).

### Dataset: `data/loans.csv`

| loan_id | purpose | principal | annual_rate | months |
|---|---|---|---|---|
| L001 | Home | 250,000 | 6.5% | 240 |
| L002 | Car | 30,000 | 8.0% | 60 |
| L003 | Education | 50,000 | 5.5% | 120 |
| L004 | Personal | 10,000 | 12.0% | 24 |
| L005 | Business | 100,000 | 9.0% | 84 |
| L006 | Home | 400,000 | 7.0% | 360 |
| L007 | Car | 18,000 | 0% | 36 |
| L008 | Personal | 5,000 | 15.0% | 12 |

Running the module on the dataset:

```bash
python -m src.banking
```

```
{'loan_id': 'L001', 'monthly_emi': 1863.93, 'total_payment': 447343.2, 'total_interest': 197343.2}
{'loan_id': 'L002', 'monthly_emi': 608.29, 'total_payment': 36497.4, 'total_interest': 6497.4}
...
{'loan_id': 'L007', 'monthly_emi': 500.0, 'total_payment': 18000.0, 'total_interest': 0.0}
{'loan_id': 'L008', 'monthly_emi': 451.29, 'total_payment': 5415.48, 'total_interest': 415.48}
```

---

## Step 4: Creating Tests using Pytest and Unittest

### Pytest: `test/test_banking_pytest.py`

- **Parametrized tests** (`@pytest.mark.parametrize`) run each function against several inputs with hand-calculated expected values.
- **Error tests** use `pytest.raises(ValueError)` for strings, `None`, booleans, negative principal or rate, zero years, and zero-month terms.
- **Dataset tests** check that all 8 loans in `loans.csv` load, and use the `tmp_path` fixture to test a temporary CSV file.
- A missing CSV file raises `FileNotFoundError`.

```bash
pytest -v
```

### Unittest: `test/test_banking_unittest.py`

The same behaviour checked with a `unittest.TestCase` class, using `assertEqual`, `assertAlmostEqual`, `assertGreater`, `assertTrue`, and `assertRaises`.

```bash
python -m unittest discover -s test -t . -v
```

### Results

| Command | Result |
|---|---|
| `pytest -v` | **41 passed** (26 banking pytest + 7 banking unittest + 8 original calculator) |
| `python -m unittest discover -s test -t .` | **11 tests OK** (7 banking + 4 calculator) |

---

## Step 5: Implementing GitHub Actions

Two workflows in [`.github/workflows/`](../../../.github/workflows) run automatically on every **push and pull request** to `main`. Each one runs 3 jobs in parallel (Python **3.10, 3.11, 3.12**) from the `Labs/Github_Labs/Lab1/` folder, and only trigger when files in this lab change.

### `lab1_banking_pytest.yml`: "Lab1 Banking - Pytest"

1. **Checkout code** with `actions/checkout@v7`
2. **Set up Python** with `actions/setup-python@v7` (matrix version, pip cache)
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Run tests and generate XML report**: `pytest -v --junitxml=pytest-report.xml`
5. **Upload test results** with `actions/upload-artifact@v7` as `pytest-report-py<version>`
6. **Notify on success / failure** using `if: success()` and `if: failure()`

### `lab1_banking_unittest.yml`: "Lab1 Banking - Unittests"

1. **Checkout code** with `actions/checkout@v7`
2. **Set up Python** with `actions/setup-python@v7` (matrix version, pip cache)
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Run unittests**: `python -m unittest discover -s test -t . -v`
5. **Notify on success / failure** using `if: success()` and `if: failure()`

Run results are on the [Actions tab](https://github.com/isakshay007/MLOps/actions).

---

## Quick start

```bash
git clone https://github.com/isakshay007/MLOps.git
cd MLOps/Labs/Github_Labs/Lab1
python3 -m venv lab_01
source lab_01/bin/activate
pip install -r requirements.txt
pytest -v
python -m unittest discover -s test -t . -v
python -m src.banking
```
