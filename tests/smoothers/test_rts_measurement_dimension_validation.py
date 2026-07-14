import unittest

from pyrecest.backend import array, eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.smoothers import RauchTungStriebelSmoother


class RauchTungStriebelMeasurementDimensionValidationTest(unittest.TestCase):
    def test_rejects_measurement_sequence_with_changing_dimension(self):
        smoother = RauchTungStriebelSmoother()

        with self.assertRaisesRegex(ValueError, "same dimension"):
            smoother.filter(
                initial_state=GaussianDistribution(zeros(2), eye(2)),
                measurements=[zeros(2), array([1.0])],
                measurement_matrices=eye(2),
                meas_noise_covariances=eye(2),
                system_matrices=eye(2),
                sys_noise_covariances=zeros((2, 2)),
            )


if __name__ == "__main__":
    unittest.main()
