import unittest

import numpy as np
import numpy.testing as npt

import pyrecest.backend
from pyrecest.distributions import WrappedNormalDistribution
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)
from pyrecest.filters.hypertoroidal_fourier_filter import HypertoroidalFourierFilter
from pyrecest.filters.low_rank_hypertoroidal_fourier_filter import (
    LowRankHypertoroidalFourierFilter,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",  # pylint: disable=no-member
    reason="Low-rank Fourier prototype is NumPy-only",
)
class TestLowRankHypertoroidalFourierFilter(unittest.TestCase):
    def test_rejects_sqrt_transform(self):
        with self.assertRaises(NotImplementedError):
            LowRankHypertoroidalFourierFilter((9,), "sqrt")

    def test_predict_identity_matches_dense_1d(self):
        dense_filter = HypertoroidalFourierFilter((9,), "identity")
        low_rank_filter = LowRankHypertoroidalFourierFilter((9,), "identity")
        prior = HypertoroidalFourierDistribution.from_distribution(
            WrappedNormalDistribution(np.array(1.0), np.array(0.5)),
            (9,),
            "identity",
        )
        noise = HypertoroidalFourierDistribution.from_distribution(
            WrappedNormalDistribution(np.array(0.0), np.array(0.3)),
            (9,),
            "identity",
        )
        dense_filter.filter_state = prior
        low_rank_filter.filter_state = prior
        dense_filter.predict_identity(noise)
        low_rank_filter.predict_identity(noise)
        npt.assert_allclose(
            low_rank_filter.filter_state.to_dense(), dense_filter.filter_state.coeff_mat, atol=1e-10
        )

    def test_update_identity_matches_dense_1d(self):
        dense_filter = HypertoroidalFourierFilter((9,), "identity")
        low_rank_filter = LowRankHypertoroidalFourierFilter((9,), "identity")
        prior = HypertoroidalFourierDistribution.from_distribution(
            WrappedNormalDistribution(np.array(1.0), np.array(0.5)),
            (9,),
            "identity",
        )
        noise = HypertoroidalFourierDistribution.from_distribution(
            WrappedNormalDistribution(np.array(0.0), np.array(1.0)),
            (9,),
            "identity",
        )
        dense_filter.filter_state = prior
        low_rank_filter.filter_state = prior
        dense_filter.update_identity(noise, np.array([1.5]))
        low_rank_filter.update_identity(noise, np.array([1.5]))
        npt.assert_allclose(
            low_rank_filter.filter_state.to_dense(), dense_filter.filter_state.coeff_mat, atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
