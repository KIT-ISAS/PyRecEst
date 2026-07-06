"""Recursive Bayesian filter on SO(3) using Wigner-D coefficients."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable

from pyrecest.distributions.so3_wigner_distribution import SO3WignerDistribution

from .abstract_filter import AbstractFilter


class SO3WignerFilter(AbstractFilter):
    """Wigner-D identity/square-root filter for densities on ``SO(3)``.

    The state is an :class:`~pyrecest.distributions.so3_wigner_distribution.SO3WignerDistribution`.
    Prediction with right- or left-invariant multiplicative noise is implemented
    by the blockwise SO(3) convolution formula in coefficient space. Bayesian
    updates with arbitrary likelihoods are implemented by quadrature-grid
    multiplication followed by Wigner coefficient projection.
    """

    def __init__(self, degree: int, transformation: str = "identity"):
        super().__init__(SO3WignerDistribution.uniform(degree, transformation))

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        if not isinstance(new_state, SO3WignerDistribution):
            raise TypeError("filter_state must be an SO3WignerDistribution.")
        self._filter_state = copy.deepcopy(new_state)

    @property
    def state(self):
        """Alias for :attr:`filter_state`."""
        return self._filter_state

    def set_state(self, state):
        """Set the Wigner-D filter state."""
        if not isinstance(state, SO3WignerDistribution):
            raise TypeError("state must be an SO3WignerDistribution.")
        if self._filter_state.transformation != state.transformation:
            warnings.warn("set_state: new density uses a different transformation.", stacklevel=2)
        if self._filter_state.degree != state.degree:
            warnings.warn("set_state: new density uses a different Wigner degree.", stacklevel=2)
        self._filter_state = copy.deepcopy(state)

    def get_estimate(self):
        """Return the current Wigner-D distribution."""
        return self._filter_state

    def get_estimate_mean(self):
        """Return the quaternion scatter-matrix mean proxy."""
        return self._filter_state.mean_axis()

    def predict_identity(self, sys_noise: SO3WignerDistribution, side: str = "right"):
        """Predict with multiplicative SO(3) noise.

        Parameters
        ----------
        sys_noise : SO3WignerDistribution
            Noise distribution represented in the same transformation as the
            filter state.
        side : {``"right"``, ``"left"``}, optional
            ``"right"`` corresponds to ``R_{k+1} = R_k W_k``; ``"left"``
            corresponds to ``R_{k+1} = W_k R_k``.
        """
        if not isinstance(sys_noise, SO3WignerDistribution):
            raise TypeError("sys_noise must be an SO3WignerDistribution.")
        self._filter_state = self._filter_state.convolve(sys_noise, side=side)

    def update_with_distribution(self, likelihood: SO3WignerDistribution, quadrature_degree: int | None = None):
        """Bayesian update with a likelihood represented by Wigner-D coefficients."""
        if not isinstance(likelihood, SO3WignerDistribution):
            raise TypeError("likelihood must be an SO3WignerDistribution.")
        self._filter_state = self._filter_state.multiply(likelihood, quadrature_degree=quadrature_degree)

    def update_nonlinear(self, likelihood: Callable, z, quadrature_degree: int | None = None):
        """Bayesian update with an arbitrary likelihood function.

        ``likelihood`` is called as ``likelihood(z, quaternions)`` where
        ``quaternions`` has shape ``(N, 4)`` and uses scalar-last canonical unit
        quaternions.
        """
        self._filter_state = self._filter_state.multiply_by_function(
            lambda quaternions: likelihood(z, quaternions),
            quadrature_degree=quadrature_degree,
        )

    def update_nonlinear_multiple(self, likelihoods, measurements, quadrature_degree: int | None = None):
        """Bayesian update with several conditionally independent likelihoods."""
        if len(likelihoods) != len(measurements):
            raise ValueError("likelihoods and measurements must have the same length.")

        def joint_likelihood(quaternions):
            values = None
            for likelihood, measurement in zip(likelihoods, measurements):
                current = likelihood(measurement, quaternions)
                values = current if values is None else values * current
            return values

        self._filter_state = self._filter_state.multiply_by_function(
            joint_likelihood,
            quadrature_degree=quadrature_degree,
        )
