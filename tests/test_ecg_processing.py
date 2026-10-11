import numpy as np
import pandas as pd
import pytest

from ecg_processing import prepare_signal


@pytest.mark.parametrize("sample", [float("nan"), float("inf"), -float("inf"), "text"])
def test_invalid_samples_are_rejected_without_compressing_timing(sample):
    values = list(range(20))
    values[5] = sample
    with pytest.raises(ValueError, match="numeric and finite"):
        prepare_signal(values, 10)


@pytest.mark.parametrize("constant", [0., 7., -2.])
def test_constant_signals_are_rejected(constant):
    with pytest.raises(ValueError, match="constant signal"):
        prepare_signal([constant] * 20, 10)


def test_large_finite_signal_is_normalized_without_overflow():
    values = np.array([-1e308, 1e308] * 10)
    with np.errstate(over="raise", invalid="raise"):
        segment, signal = prepare_signal(values, 10)
    assert segment.shape == (1, 10, 1)
    assert np.isfinite(signal).all()
    np.testing.assert_allclose(signal, [-1, 1] * 10)


def test_normalization_preserves_sample_order_and_length():
    values = np.arange(20, dtype=float)
    segment, signal = prepare_signal(values, 10)
    assert len(signal) == len(values)
    np.testing.assert_allclose(signal, (values - values.mean()) / values.std())
    np.testing.assert_allclose(segment[0, :, 0], signal[:10])


def test_csv_with_missing_samples_is_rejected(tmp_path):
    import app
    path = tmp_path / "ecg.csv"
    values = np.sin(np.arange(500) / 10)
    values[10] = np.nan
    pd.DataFrame({"MLII": values}).to_csv(path, index=False)
    result, error = app.preprocess(path)
    assert result is None
    assert "missing values" in error
