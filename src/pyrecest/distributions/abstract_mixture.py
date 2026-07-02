import collections
import copy
import warnings
from typing import Union

import numpy as np
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    array,
    asarray,
    empty,
    int32,
    int64,
    isfinite,
    ones,
    random,
    reshape,
    sum,
    zeros,
)

from .abstract_distribution_type import AbstractDistributionType
from .abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)


def _validate_positive_sample_count(n) -> int:
    message = "n must be a positive integer"
    count_array = np.asarray(n)
    if count_array.ndim != 0:
        raise ValueError(message)

    count = count_array.item()
    if isinstance(
        count,
        (bool, np.bool_, str, bytes, bytearray, np.str_, np.bytes_),
    ):
        raise ValueError(message)

    try:
        count_int = int(count)
        count_float = float(count)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc

    if not np.isfinite(count_float) or not count_float.is_integer():
        raise ValueError(message)
    if count_int <= 0:
        raise ValueError(message)
    return count_int


class AbstractMixture(AbstractDistributionType):
    """
    Abstract base class for mixture distributions.
    """

    def __init__(
        self,
        dists: collections.abc.Sequence[AbstractManifoldSpecificDistribution],
        weights=None,
    ):
        AbstractDistributionType.__init__(self)
        dists = copy.deepcopy(dists)  # To prevent modifying the original object
        num_distributions = len(dists)
        if num_distributions == 0:
            raise ValueError("Mixture must contain at least one distribution")

        if weights is None:
            weights = ones(num_distributions) / num_distributions
        else:
            weights = reshape(asarray(weights), (-1,))

        if num_distributions != weights.shape[0]:
            raise ValueError("Sizes of distributions and weights must be equal")

        if any(not bool(isfinite(weight)) for weight in weights):
            raise ValueError("Mixture weights must be finite")

        if any(bool(weight < 0) for weight in weights):
            raise ValueError("Mixture weights must be nonnegative")

        if not all(dists[0].dim == dist.dim for dist in dists):
            raise ValueError("All distributions must have the same dimension")

        non_zero_indices = [i for i, weight in enumerate(weights) if bool(weight != 0)]

        if len(non_zero_indices) == 0:
            raise ValueError("At least one mixture weight must be nonzero")

        weight_sum = sum(weights)
        if not bool(isfinite(weight_sum)) or not bool(weight_sum > 0.0):
            raise ValueError("Mixture weights must have positive finite total mass")

        if len(non_zero_indices) < len(weights):
            warnings.warn(
                "Elements with zero weights detected. Pruning elements in mixture with weight zero."
            )
            dists = [dists[i] for i in non_zero_indices]
            weights = weights[array(non_zero_indices, dtype=int64)]

        self.dists = dists

        if abs(weight_sum - 1.0) > 1e-10:
            warnings.warn("Weights of mixture do not sum to one.")
            self.w = weights / weight_sum
        else:
            self.w = weights

    @property
    def input_dim(self) -> int:
        return self.dists[0].input_dim

    def _as_sample_matrix(self, samples, n_samples: int):
        samples = asarray(samples)

        if self.input_dim == 1 and samples.ndim == 0:
            return reshape(samples, (1, 1))

        if self.input_dim == 1 and samples.ndim == 1:
            return reshape(samples, (n_samples, 1))

        return pyrecest.backend.atleast_2d(samples)

    def _sample_component_matrix(self, component_index: int, n_samples: int):
        try:
            sample_i = self.dists[component_index].sample(n_samples)
        except (NotImplementedError, AssertionError, ValueError, TypeError):
            if pyrecest.backend.__backend_name__ != "jax":
                raise
            sample_i = self.dists[component_index].sample_metropolis_hastings(n_samples)
        return self._as_sample_matrix(sample_i, n_samples)

    def sample(self, n: Union[int, int32, int64]):
        """Draw iid mixture samples without biasing output positions.

        Sampling only multinomial component counts gives the correct unordered
        bag of component labels, but filling the output array with component-wise
        blocks makes early output positions more likely to come from early
        components. Draw one component label per requested sample and place each
        component's samples back into those sampled positions.
        """
        n = _validate_positive_sample_count(n)
        component_indices = random.choice(len(self.dists), size=n, p=self.w)
        component_indices_np = pyrecest.backend.to_numpy(component_indices).reshape(-1)

        if pyrecest.backend.__backend_name__ == "jax":
            samples = [
                self._sample_component_matrix(int(component_index), 1)
                for component_index in component_indices_np
            ]
            return pyrecest.backend.concatenate(samples, axis=0)

        s = empty((n, self.input_dim))
        for component_index in range(len(self.dists)):
            positions = np.flatnonzero(component_indices_np == component_index)
            occ_val = int(positions.size)
            if occ_val == 0:
                continue
            sample_i = self._sample_component_matrix(component_index, occ_val)
            s[array(positions, dtype=int64)] = sample_i

        return s

    def pdf(self, xs):
        xs = asarray(xs)

        if self.input_dim == 1 and xs.ndim <= 1:
            # For one-dimensional distributions, a flat array represents a batch
            # of scalar evaluation points. This matches GaussianDistribution.pdf
            # and avoids rejecting natural inputs such as xs.shape == (n,).
            p = zeros(xs.shape)
        else:
            if xs.ndim == 0 or xs.shape[-1] != self.input_dim:
                raise ValueError("Dimension mismatch")
            p = zeros(1) if xs.ndim == 1 else zeros(xs.shape[0])

        for i, dist in enumerate(self.dists):
            p += self.w[i] * dist.pdf(xs)

        return p
