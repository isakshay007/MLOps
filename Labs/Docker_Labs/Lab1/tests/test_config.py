from pathlib import Path

import pytest

from config import DEFAULT_OUTPUT_DIR, TrainConfig


def test_defaults_without_env():
    config = TrainConfig.from_env({})
    assert config == TrainConfig()
    assert config.output_dir == DEFAULT_OUTPUT_DIR


def test_env_overrides():
    config = TrainConfig.from_env({"N_ESTIMATORS": "50", "LEARNING_RATE": "0.1",
                                   "MAX_DEPTH": "4", "OUTPUT_DIR": "/tmp/out"})
    assert config.n_estimators == 50
    assert config.learning_rate == 0.1
    assert config.max_depth == 4
    assert config.output_dir == Path("/tmp/out")


def test_empty_env_value_uses_default():
    assert TrainConfig.from_env({"N_ESTIMATORS": ""}).n_estimators == TrainConfig().n_estimators


def test_invalid_type_names_the_variable():
    with pytest.raises(ValueError, match="N_ESTIMATORS='abc'"):
        TrainConfig.from_env({"N_ESTIMATORS": "abc"})


def test_out_of_range_rejected():
    with pytest.raises(ValueError, match="LEARNING_RATE"):
        TrainConfig.from_env({"LEARNING_RATE": "0"})
    with pytest.raises(ValueError, match="TEST_SIZE"):
        TrainConfig(test_size=1.5)
