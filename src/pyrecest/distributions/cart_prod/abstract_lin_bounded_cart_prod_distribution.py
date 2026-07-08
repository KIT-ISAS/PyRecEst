from abc import abstractmethod
from operator import index as _operator_index
from typing import Union

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import int32, int64

from .abstract_cart_prod_distribution import AbstractCartProdDistribution


def _validate_nonnegative_dimension_count(value, name: str) -> int:
    """Return ``value`` as a non-negative Python integer dimension count."""
    message = f"{name} must be a non-negative integer"
    if isinstance(value, bool):
        raise ValueError(message)

    dtype = getattr(value, "dtype", None)
    if getattr(dtype, "kind", None) == "b" or str(dtype) == "torch.bool":
        raise ValueError(message)

    ndim = getattr(value, "ndim", None)
    if ndim not in (None, 0):
        raise ValueError(message)
    if ndim == 0 and hasattr(value, "item"):
        value = value.item()

    try:
        count = _operator_index(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if count < 0:
        raise ValueError(message)
    return int(count)


class AbstractLinBoundedCartProdDistribution(AbstractCartProdDistribution):
    """
    For Cartesian products of linear and bounded (periodic or parts of
    Euclidean spaces) domains. Assumption is that the input dimensions
    are ordered as follows: bounded dimensions first, then linear dimensions.
    """

    def __init__(
        self, bound_dim: Union[int, int32, int64], lin_dim: Union[int, int32, int64]
    ):
        """
        Parameters:
            bound_dim (int)
                number of bounded (e.g., periodic or hyperrectangular) dimensions
            lin_dim (int)
                number of linear dimensions

        """
        bound_dim = _validate_nonnegative_dimension_count(bound_dim, "bound_dim")
        lin_dim = _validate_nonnegative_dimension_count(lin_dim, "lin_dim")
        if not bound_dim + lin_dim >= 1:
            raise ValueError("total dimension must be positive")

        AbstractCartProdDistribution.__init__(self, bound_dim + lin_dim)
        self.bound_dim = bound_dim
        self.lin_dim = lin_dim

    def mean(self):
        """
        Convenient access to hybrid_mean() to have a consistent interface
        throughout manifolds.

        :return: The mean of the distribution.
        :rtype:
        """
        return self.hybrid_mean()

    def hybrid_mean(self):
        return (
            self.marginalize_linear().mean(),
            self.marginalize_periodic().mean(),
        )

    @abstractmethod
    def marginalize_linear(self):
        pass

    @abstractmethod
    def marginalize_periodic(self):
        pass
