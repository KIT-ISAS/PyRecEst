from typing import Union

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import asarray, int32, int64

from ..hypersphere_subset.hyperspherical_dirac_distribution import (
    HypersphericalDiracDistribution,
)
from .abstract_lin_hyperhemisphere_cart_prod_distribution import (
    AbstractLinHypersphereSubsetCartProdDistribution,
)
from .lin_bounded_cart_prod_dirac_distribution import (
    LinBoundedCartProdDiracDistribution,
)


class LinHypersphereSubsetCartProdDiracDistribution(
    LinBoundedCartProdDiracDistribution,
    AbstractLinHypersphereSubsetCartProdDistribution,
):
    def __init__(self, bound_dim: Union[int, int32, int64], d, w=None):
        d = asarray(d)
        AbstractLinHypersphereSubsetCartProdDistribution.__init__(
            self, bound_dim, d.shape[-1] - bound_dim - 1
        )
        LinBoundedCartProdDiracDistribution.__init__(self, d=d, w=w)

    def marginalize_linear(self):
        return HypersphericalDiracDistribution(
            self.d[:, : self.bound_dim + 1],  # noqa: E203
            self.w,
        )
