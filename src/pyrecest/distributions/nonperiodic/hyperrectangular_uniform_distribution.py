from pyrecest.backend import all as backend_all
from pyrecest.backend import (
    array,
    is_complex,
    isfinite,
    logical_and,
    ones,
    reshape,
    where,
    zeros,
)

from ..abstract_uniform_distribution import AbstractUniformDistribution
from .abstract_hyperrectangular_distribution import AbstractHyperrectangularDistribution


class HyperrectangularUniformDistribution(
    AbstractUniformDistribution, AbstractHyperrectangularDistribution
):
    def __init__(self, bounds):
        AbstractUniformDistribution.__init__(self)
        AbstractHyperrectangularDistribution.__init__(self, bounds)

    def get_manifold_size(self):
        return AbstractHyperrectangularDistribution.get_manifold_size(self)

    def pdf(self, xs):
        """Evaluate the uniform density inside the hyperrectangle and zero outside."""
        xs, single = self._coerce_points(xs)
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        inside = backend_all(logical_and(xs >= lower, xs <= upper), axis=1)
        values = where(
            inside,
            ones(xs.shape[0]) / self.get_manifold_size(),
            zeros(xs.shape[0]),
        )
        return values[0] if single else values

    def _coerce_points(self, xs):
        xs = array(xs)
        if is_complex(xs):
            raise ValueError("xs must be real-valued")
        if not bool(backend_all(isfinite(xs))):
            raise ValueError("xs must contain only finite values")
        if xs.ndim == 0:
            if self.dim != 1:
                raise ValueError("Scalar points are only valid for dim == 1")
            return reshape(xs, (1, 1)), True
        if xs.ndim == 1:
            if self.dim == 1:
                return reshape(xs, (-1, 1)), False
            if xs.shape[0] == self.dim:
                return reshape(xs, (1, self.dim)), True
            raise ValueError(
                f"Point dimension {xs.shape[0]} does not match dim {self.dim}"
            )
        if xs.ndim != 2 or xs.shape[1] != self.dim:
            raise ValueError(f"xs must have shape (n, {self.dim})")
        return xs, False
