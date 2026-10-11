import os
from pathlib import Path
import subprocess
import sys


def test_model_loads_from_different_working_directory(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(repository)}
    result = subprocess.run([sys.executable, "-c",
        "import app; assert app.model.input_shape == (None, 400, 1); "
        "assert app.model.output_shape == (None, 1)"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
