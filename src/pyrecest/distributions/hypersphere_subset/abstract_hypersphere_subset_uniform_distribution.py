# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, ones

from ..abstract_uniform_distribution import AbstractUniformDistribution
from .abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)


class AbstractHypersphereSubsetUniformDistribution(
    AbstractHypersphereSubsetDistribution, AbstractUniformDistribution
):
    """
    This is an abstract class for a uniform distribution over a subset of a hypersphere.
    """

    def pdf(self, xs):
        """
        Calculates the probability density function over the subset of the hypersphere.

        Args:
            xs (): Input data points.

        Returns:
            : Probability density at the given data points.
        """
        xs = array(xs)
        if xs.ndim == 0 or xs.shape[-1] != self.input_dim:
            raise ValueError("Invalid shape of input data points.")
        manifold_size = self.get_manifold_size()
        if manifold_size == 0:
            raise ValueError("Manifold size cannot be zero.")
        if not isinstance(manifold_size, (int, float)):
            raise TypeError("Manifold size must be a numeric value.")
        p = (
            (1 / manifold_size) * ones(xs.shape[:-1])
            if xs.ndim > 1
            else 1 / manifold_size
        )

        return p
