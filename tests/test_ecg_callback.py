import numpy as np
import pandas as pd
from PIL import Image

import app


def test_real_cnn_prediction_returns_an_image_without_shared_files(tmp_path, monkeypatch):
    path = tmp_path / "ecg.csv"
    signal = np.sin(2 * np.pi * np.arange(1000) / 360)
    pd.DataFrame({"MLII": signal}).to_csv(path, index=False)
    monkeypatch.chdir(tmp_path)
    output, image = app.predict_ecg(path)
    assert isinstance(image, Image.Image), output
    assert "Result:" in output
    assert "Confidence:" in output
    assert "**Normal**" in output or "**Abnormal**" in output
    assert not (tmp_path / "temp_ecg_plot.png").exists()
