import sys
from pathlib import Path

import pytest

# The modules import each other as top-level modules (e.g. `from data import ...`), the same
# way they run as `python src/main.py`, so put src/ on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import TrainConfig  # noqa: E402
from train import train  # noqa: E402

# A real patient from the dataset (row 0), in original units
PATIENT = {"age": 59, "sex": 2, "bmi": 32.1, "bp": 101, "s1": 157,
           "s2": 93.2, "s3": 38, "s4": 4, "s5": 4.8598, "s6": 87}


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train once with the default settings into a temporary folder; returns (config, metrics)."""
    config = TrainConfig(output_dir=tmp_path_factory.mktemp("output"))
    return config, train(config)
