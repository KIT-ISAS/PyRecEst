import numpy as np
import pytest

from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution, LinearDiracDistribution


def _sampleable_distributions():
    return (
        GaussianDistribution(array([0.0]), array([[1.0]])),
        LinearDiracDistribution(array([[0.0], [1.0]]), array([0.5, 0.5])),
    )


@pytest.mark.parametrize(
    "masked_count",
    [
        np.ma.masked,
        np.ma.array(3, mask=True),
    ],
)
def test_sampling_rejects_genuinely_masked_counts(masked_count):
    for distribution in _sampleable_distributions():
        with pytest.raises(ValueError, match="positive integer"):
            distribution.sample(masked_count)


def test_sampling_accepts_unmasked_masked_array_integer():
    count = np.ma.array(2, mask=False)

    for distribution in _sampleable_distributions():
        assert distribution.sample(count).shape == (2, 1)
