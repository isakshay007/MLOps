import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Outside Docker the model goes to Lab1/output; the image sets OUTPUT_DIR=/app/output,
# which is where you mount a host folder to keep the model after the container exits.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


@dataclass(frozen=True)
class TrainConfig:
    """Training settings. Every field can be overridden with an environment variable (see ENV_VARS)."""
    n_estimators: int = 200
    learning_rate: float = 0.03
    max_depth: int = 2
    subsample: float = 0.8
    test_size: float = 0.2
    random_state: int = 42
    output_dir: Path = field(default=DEFAULT_OUTPUT_DIR)

    def __post_init__(self):
        checks = {
            "N_ESTIMATORS": self.n_estimators >= 1,
            "LEARNING_RATE": 0 < self.learning_rate <= 1,
            "MAX_DEPTH": self.max_depth >= 1,
            "SUBSAMPLE": 0 < self.subsample <= 1,
            "TEST_SIZE": 0 < self.test_size < 1,
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"Out-of-range setting(s): {', '.join(bad)}")

    @property
    def model_params(self):
        """Keyword arguments for GradientBoostingRegressor."""
        return {"n_estimators": self.n_estimators, "learning_rate": self.learning_rate,
                "max_depth": self.max_depth, "subsample": self.subsample,
                "random_state": self.random_state}

    def to_dict(self):
        return {**asdict(self), "output_dir": str(self.output_dir)}

    @classmethod
    def from_env(cls, env=None):
        """Build a config from environment variables, falling back to the defaults above."""
        env = os.environ if env is None else env
        values = {}
        for var, (name, cast) in ENV_VARS.items():
            if env.get(var, "") != "":
                try:
                    values[name] = cast(env[var])
                except ValueError:
                    raise ValueError(f"{var}={env[var]!r} is not a valid {cast.__name__}") from None
        return cls(**values)


ENV_VARS = {
    "N_ESTIMATORS": ("n_estimators", int),
    "LEARNING_RATE": ("learning_rate", float),
    "MAX_DEPTH": ("max_depth", int),
    "SUBSAMPLE": ("subsample", float),
    "TEST_SIZE": ("test_size", float),
    "RANDOM_STATE": ("random_state", int),
    "OUTPUT_DIR": ("output_dir", Path),
}
