import numpy as np
import pytest

from ecg_processing import binary_probability


@pytest.mark.parametrize("prediction", [
    None, [], [0.4], [[0.4, 0.6]], [[True]], [["0.4"]], [[1 + 0j]],
    [[float("nan")]], [[float("inf")]], [[-0.1]], [[1.1]], [[10**400]],
])
def test_invalid_model_outputs_cannot_be_reported_as_confidence(prediction):
    with pytest.raises(ValueError, match="probability"):
        binary_probability(prediction)


@pytest.mark.parametrize("probability", [0., 0.5, 1., np.float32(0.7)])
def test_valid_probability_boundaries_are_supported(probability):
    assert binary_probability([[probability]]) == pytest.approx(float(probability))
