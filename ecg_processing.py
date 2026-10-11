"""Numerical ECG preprocessing that can be checked without loading TensorFlow."""
from numbers import Integral

import numpy as np


def prepare_signal(values, expected_length):
    if isinstance(expected_length, bool) or not isinstance(expected_length, Integral) or expected_length < 1:
        raise ValueError("The model must declare a positive input length")
    try:
        signal = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("ECG samples must be numeric and finite") from exc
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError("ECG samples must be numeric and finite, without missing values")
    if len(signal) < expected_length:
        raise ValueError(f"Not enough numerical data points. Found {len(signal)}, need at least {expected_length}.")
    # Scale before computing statistics to avoid overflow for large finite samples.
    scale = np.max(np.abs(signal))
    if scale == 0:
        raise ValueError("ECG signal must contain variation; a constant signal cannot be analyzed")
    scaled = signal / scale
    spread = np.std(scaled)
    if spread == 0:
        raise ValueError("ECG signal must contain variation; a constant signal cannot be analyzed")
    normalized = (scaled - np.mean(scaled)) / spread
    return normalized[:expected_length].reshape(1, expected_length, 1), normalized
