import sys
from pathlib import Path

# The app modules import each other as top-level modules (e.g. `from predict import ...`),
# the same way they run under `uvicorn main:app` from src/, so put src/ on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
