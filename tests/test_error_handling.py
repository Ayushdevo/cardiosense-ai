import logging

import numpy as np
import pandas as pd

import app


def ecg_file(tmp_path):
    path = tmp_path / "ecg.csv"
    pd.DataFrame({"MLII": np.sin(np.arange(500) / 15)}).to_csv(path, index=False)
    return path


def test_internal_model_errors_are_logged_without_exposure(tmp_path, monkeypatch, caplog):
    class BrokenModel:
        input_shape = (None, 400, 1)

        def predict(self, *args, **kwargs):
            raise RuntimeError("private/internal/model-details")

    monkeypatch.setattr(app, "model", BrokenModel())
    with caplog.at_level(logging.ERROR, logger="app"):
        output, image = app.predict_ecg(ecg_file(tmp_path))
    assert image is None
    assert "analysis failed" in output
    assert "private/internal" not in output
    assert "Traceback" not in output
    assert "private/internal/model-details" in caplog.text


def test_missing_csv_does_not_expose_internal_file_path(tmp_path, caplog):
    path = tmp_path / "private-name.csv"
    with caplog.at_level(logging.WARNING, logger="app"):
        output, image = app.predict_ecg(path)
    assert image is None
    assert "Unable to read ECG CSV" in output
    assert "private-name" not in output
    assert str(tmp_path) not in output
    assert "Unable to read ECG CSV" in caplog.text


def test_nonfinite_prediction_is_reported_as_failure(tmp_path, monkeypatch):
    class BrokenModel:
        input_shape = (None, 400, 1)

        def predict(self, *args, **kwargs):
            return [[float("nan")]]

    monkeypatch.setattr(app, "model", BrokenModel())
    output, image = app.predict_ecg(ecg_file(tmp_path))
    assert image is None
    assert "analysis failed" in output
    assert "Confidence" not in output


def test_unsupported_upload_type_has_actionable_message():
    output, image = app.predict_ecg(object())
    assert image is None
    assert "Please select an ECG CSV file" in output
    assert "class" not in output
