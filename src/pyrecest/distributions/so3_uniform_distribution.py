"""Uniform distribution on SO(3)."""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    log,
    ones,
    pi,
)

from ._so3_helpers import (
    geodesic_distance,
    normalize_quaternions,
    quaternions_to_rotation_matrices,
)
from .hypersphere_subset.hyperhemispherical_uniform_distribution import (
    HyperhemisphericalUniformDistribution,
)


class SO3UniformDistribution(HyperhemisphericalUniformDistribution):
    """Uniform distribution on SO(3) using scalar-last unit quaternions.

    The distribution represents rotations by canonical unit quaternions
    ``(x, y, z, w)`` with nonnegative scalar component. This identifies each
    SO(3) rotation with one point on the upper hemisphere of S^3, except for
    the measure-zero equator where ``q`` and ``-q`` both have zero scalar part.
    """

    def __init__(self):
        super().__init__(dim=3)

    @staticmethod
    def _normalize_quaternions(quaternions):
        return normalize_quaternions(quaternions)

    def pdf(self, xs):
        """Evaluate the constant SO(3) density."""
        xs = self._normalize_quaternions(xs)
        return ones(xs.shape[:-1]) / self.get_manifold_size()

    def ln_pdf(self, xs):
        """Evaluate the natural logarithm of the constant SO(3) density."""
        xs = self._normalize_quaternions(xs)
        return -log(self.get_manifold_size()) * ones(xs.shape[:-1])

    def sample(self, n):
        """Draw ``n`` canonical scalar-last unit quaternions."""
        return self._normalize_quaternions(super().sample(n))

    def mode(self):
        """Mode is undefined for a uniform SO(3) distribution."""
        raise AttributeError("Mode not available for uniform distribution")

    def mean(self):
        """Mean rotation is undefined for a uniform SO(3) distribution."""
        raise AttributeError("Mean not available for uniform SO(3) distribution")

    def mean_axis(self):
        """Mean axis is undefined for a uniform SO(3) distribution."""
        raise AttributeError("Mean axis not available for uniform SO(3) distribution")

    geodesic_distance = staticmethod(geodesic_distance)
    as_rotation_matrices = staticmethod(quaternions_to_rotation_matrices)

    def is_valid(self, tolerance=1e-12):
        """Return whether this instance has the expected SO(3) dimensions."""
        return bool(
            self.dim == 3 and abs(self.get_manifold_size() - pi**2) <= tolerance
        )
