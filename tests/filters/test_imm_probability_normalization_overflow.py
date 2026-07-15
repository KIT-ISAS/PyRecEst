import unittest
import warnings

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, to_numpy
from pyrecest.filters.interacting_multiple_model_filter import (
    InteractingMultipleModelFilter,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="IMM tests currently run on the numpy backend",
)
class IMMProbabilityNormalizationOverflowTest(unittest.TestCase):
    def test_normalizes_extreme_finite_probabilities_without_overflow(self):
        maximum = np.finfo(float).max
        transition_values = np.array(
            [[maximum, maximum / 2.0], [maximum / 2.0, maximum]]
        )
        mode_values = np.array([maximum, maximum / 2.0])

        self.assertTrue(np.all(np.isfinite(transition_values)))
        self.assertTrue(np.all(np.isfinite(mode_values)))
        with np.errstate(over="ignore"):
            self.assertTrue(np.all(np.isinf(transition_values.sum(axis=1))))
            self.assertTrue(np.isinf(mode_values.sum()))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            transition_matrix = (
                InteractingMultipleModelFilter._prepare_transition_matrix(
                    array(transition_values), 2
                )
            )
            mode_probabilities = (
                InteractingMultipleModelFilter._prepare_mode_probabilities(
                    array(mode_values), 2
                )
            )

        npt.assert_allclose(
            to_numpy(transition_matrix),
            np.array([[2.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0]]),
        )
        npt.assert_allclose(
            to_numpy(mode_probabilities), np.array([2.0 / 3.0, 1.0 / 3.0])
        )
        npt.assert_allclose(to_numpy(transition_matrix).sum(axis=1), np.ones(2))
        npt.assert_allclose(to_numpy(mode_probabilities).sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
