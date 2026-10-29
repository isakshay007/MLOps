import sys
from pathlib import Path

import pytest

# The scripts import each other as top-level modules (e.g. `from data import ...`), the same
# way they run as `python src/train_model.py`, so put src/ on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import load_data, split_data  # noqa: E402
from train_model import fit_model  # noqa: E402


@pytest.fixture(scope="session")
def splits():
    X, y = load_data()
    return split_data(X, y)


@pytest.fixture(scope="session")
def trained_model(splits):
    X_train, _, y_train, _ = splits
    return fit_model(X_train, y_train)
