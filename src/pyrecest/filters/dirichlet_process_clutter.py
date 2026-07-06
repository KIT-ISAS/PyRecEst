# pylint: disable=no-name-in-module,no-member
"""Dirichlet-process Gaussian clutter intensity for RFS trackers.

This module provides a lightweight Bayesian-nonparametric component for
multi-target trackers that already consume clutter intensities.  The class below
keeps an online finite approximation to a Dirichlet-process Gaussian mixture and
exposes the posterior-predictive clutter intensity in the measurement space.

The intended DP/RFS usage pattern is deliberately modular: an RFS tracker handles
existence, association, missed detections, and labels, while this object learns a
structured clutter distribution from measurements or soft clutter responsibilities.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import log, pi
from numbers import Integral, Real

import numpy as np

from pyrecest.backend import array


@dataclass(frozen=True)
class _GaussianCache:
    inverse: np.ndarray
    log_determinant: float


def _validate_positive_scalar(value, name):
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


def _validate_nonnegative_scalar(value, name):
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite.")
    return value


def _as_1d_float_array(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        value = value.reshape(1)
    if value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _as_covariance_matrix(value, name, dim):
    covariance = np.asarray(value, dtype=float)
    if covariance.ndim == 0:
        covariance = float(covariance) * np.eye(dim)
    if covariance.shape != (dim, dim):
        raise ValueError(f"{name} must have shape ({dim}, {dim}) or be scalar.")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must contain only finite values.")
    covariance = 0.5 * (covariance + covariance.T)
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(log_determinant):
        raise ValueError(f"{name} must be positive definite.")
    return covariance


def _gaussian_cache(covariance):
    return _GaussianCache(
        inverse=np.linalg.inv(covariance),
        log_determinant=float(np.linalg.slogdet(covariance)[1]),
    )


def _log_gaussian_density(measurement, mean, covariance_cache):
    difference = measurement - mean
    dimension = measurement.size
    quadratic_form = float(difference.T @ covariance_cache.inverse @ difference)
    return -0.5 * (
        dimension * log(2.0 * pi)
        + covariance_cache.log_determinant
        + quadratic_form
    )


def _normalize_log_weights(log_weights):
    log_weights = np.asarray(log_weights, dtype=float)
    max_log_weight = np.max(log_weights)
    if not np.isfinite(max_log_weight):
        return np.full(log_weights.shape, 1.0 / log_weights.size)
    shifted_weights = np.exp(log_weights - max_log_weight)
    total_weight = np.sum(shifted_weights)
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        return np.full(log_weights.shape, 1.0 / log_weights.size)
    return shifted_weights / total_weight


def _as_measurement_columns(measurements, dim, name="measurements"):
    measurements = np.asarray(measurements, dtype=float)
    if measurements.ndim == 0:
        if dim != 1:
            raise ValueError(f"{name} must have shape ({dim}, num_measurements).")
        measurements = measurements.reshape(1, 1)
    elif measurements.ndim == 1:
        if measurements.shape[0] != dim:
            raise ValueError(
                f"{name} must be a vector of length {dim} or a matrix with "
                f"shape ({dim}, num_measurements)."
            )
        measurements = measurements.reshape(dim, 1)
    elif measurements.ndim == 2:
        if measurements.shape[0] != dim:
            raise ValueError(f"{name} must have shape ({dim}, num_measurements).")
    else:
        raise ValueError(f"{name} must be one- or two-dimensional.")
    if not np.all(np.isfinite(measurements)):
        raise ValueError(f"{name} must contain only finite values.")
    return measurements


def _as_observation_weights(weights, num_measurements):
    if weights is None:
        return np.ones(num_measurements)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 0:
        weights = weights.reshape(1)
    weights = weights.reshape((-1,))
    if weights.size != num_measurements:
        raise ValueError("weights must contain one value per measurement column.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and nonnegative.")
    return weights


class DirichletProcessGaussianClutterIntensity:
    """Online DP-Gaussian mixture approximation to a clutter intensity.

    The model uses the posterior-predictive form

    ``lambda_c * (alpha * p_base(z) + sum_j n_j * N(z | mu_j, R)) / (alpha + sum_j n_j)``

    where ``alpha`` is the Dirichlet-process concentration, ``lambda_c`` is the
    clutter rate, and the occupied atoms are Gaussian kernels with fixed
    covariance ``R``.  Updates are online and approximate: each weighted clutter
    observation is assigned fractionally to existing atoms and, when its new-atom
    responsibility is large enough, starts a new atom.  This keeps the object
    small enough to be used inside recursive RFS trackers.

    Parameters
    ----------
    concentration : float
        Dirichlet-process concentration parameter. Higher values keep more
        posterior-predictive mass on unseen clutter atoms.
    base_mean : array-like, shape (dim,)
        Mean of the Gaussian base predictive density.
    base_covariance : array-like, shape (dim, dim) or scalar
        Covariance of the Gaussian base measure over new clutter locations.
    kernel_covariance : array-like, shape (dim, dim) or scalar
        Fixed covariance used by occupied clutter atoms.
    clutter_rate : float, optional
        Total clutter rate multiplying the normalized predictive density.
    intensity_floor : float, optional
        Positive lower bound returned by intensity calls.
    maximum_components : int, optional
        Maximum number of occupied atoms kept after updates.
    pruning_threshold : float, optional
        Remove atoms with effective count below this threshold after updates.
    new_component_probability_threshold : float, optional
        Create a new atom when the new-atom posterior responsibility reaches this
        threshold.  If no atom exists, the first positive-weight observation
        always creates one.
    initial_means : array-like, optional
        Initial atom means with shape ``(dim, num_components)``.
    initial_counts : array-like, optional
        Initial effective counts for ``initial_means``.  If omitted, all initial
        atoms receive unit count.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        concentration,
        base_mean,
        base_covariance,
        kernel_covariance,
        clutter_rate=1.0,
        intensity_floor=1e-300,
        maximum_components=None,
        pruning_threshold=1e-6,
        new_component_probability_threshold=0.5,
        initial_means=None,
        initial_counts=None,
    ):
        self.concentration = _validate_positive_scalar(
            concentration,
            "concentration",
        )
        self.clutter_rate = _validate_nonnegative_scalar(
            clutter_rate,
            "clutter_rate",
        )
        self.intensity_floor = _validate_positive_scalar(
            intensity_floor,
            "intensity_floor",
        )
        self.pruning_threshold = _validate_nonnegative_scalar(
            pruning_threshold,
            "pruning_threshold",
        )
        threshold = _validate_nonnegative_scalar(
            new_component_probability_threshold,
            "new_component_probability_threshold",
        )
        if threshold > 1.0:
            raise ValueError("new_component_probability_threshold must be in [0, 1].")
        self.new_component_probability_threshold = threshold

        if maximum_components is not None:
            if not isinstance(maximum_components, Integral):
                raise TypeError("maximum_components must be an integer or None.")
            if int(maximum_components) <= 0:
                raise ValueError("maximum_components must be positive when supplied.")
            maximum_components = int(maximum_components)
        self.maximum_components = maximum_components

        self.base_mean = _as_1d_float_array(base_mean, "base_mean")
        self.dim = self.base_mean.size
        self.base_covariance = _as_covariance_matrix(
            base_covariance,
            "base_covariance",
            self.dim,
        )
        self.kernel_covariance = _as_covariance_matrix(
            kernel_covariance,
            "kernel_covariance",
            self.dim,
        )
        self._base_predictive_covariance = (
            self.base_covariance + self.kernel_covariance
        )
        self._base_cache = _gaussian_cache(self._base_predictive_covariance)
        self._kernel_cache = _gaussian_cache(self.kernel_covariance)

        self.component_means = np.empty((self.dim, 0))
        self.component_counts = np.empty((0,))
        if initial_means is not None:
            self._set_initial_components(initial_means, initial_counts)
            self.prune()
            self.cap()

    @classmethod
    def from_observations(
        cls,
        measurements,
        concentration,
        base_mean,
        base_covariance,
        kernel_covariance,
        weights=None,
        **kwargs,
    ):
        """Create a model and immediately update it with clutter observations."""
        model = cls(
            concentration,
            base_mean,
            base_covariance,
            kernel_covariance,
            **kwargs,
        )
        model.observe(measurements, weights=weights)
        return model

    def copy(self):
        """Return a deep copy of the clutter-intensity model."""
        return copy.deepcopy(self)

    def _set_initial_components(self, initial_means, initial_counts):
        means = _as_measurement_columns(initial_means, self.dim, name="initial_means")
        if initial_counts is None:
            counts = np.ones(means.shape[1])
        else:
            counts = _as_observation_weights(initial_counts, means.shape[1])
        if np.any(counts <= 0.0):
            raise ValueError("initial_counts must be positive.")
        self.component_means = means.copy()
        self.component_counts = counts.astype(float, copy=True)

    def get_number_of_components(self):
        """Return the number of occupied clutter atoms currently retained."""
        return int(self.component_counts.size)

    def get_component_means(self):
        """Return retained clutter-atom means as columns."""
        return array(self.component_means.copy())

    def get_component_counts(self):
        """Return retained clutter-atom effective counts."""
        return array(self.component_counts.copy())

    def get_base_weight(self):
        """Return the posterior-predictive weight of an unseen clutter atom."""
        return float(self.concentration / self._total_predictive_mass())

    def get_component_weights(self):
        """Return posterior-predictive weights of retained clutter atoms."""
        return array(self.component_counts / self._total_predictive_mass())

    def _total_predictive_mass(self):
        return self.concentration + float(np.sum(self.component_counts))

    def _base_log_density(self, measurement):
        return _log_gaussian_density(
            measurement,
            self.base_mean,
            self._base_cache,
        )

    def _kernel_log_densities(self, measurement):
        if self.get_number_of_components() == 0:
            return np.empty((0,))
        return np.asarray(
            [
                _log_gaussian_density(
                    measurement,
                    self.component_means[:, component_index],
                    self._kernel_cache,
                )
                for component_index in range(self.get_number_of_components())
            ],
            dtype=float,
        )

    def predictive_density(self, measurement):
        """Evaluate the normalized posterior-predictive clutter density at ``measurement``."""
        measurement = _as_1d_float_array(measurement, "measurement")
        if measurement.size != self.dim:
            raise ValueError(f"measurement must have length {self.dim}.")

        total_mass = self._total_predictive_mass()
        density = self.concentration * np.exp(self._base_log_density(measurement))
        if self.get_number_of_components() > 0:
            kernel_densities = np.exp(self._kernel_log_densities(measurement))
            density += float(np.sum(self.component_counts * kernel_densities))
        return max(float(density / total_mass), 0.0)

    def __call__(self, measurement, measurement_index=None):
        """Return the clutter intensity for one measurement.

        ``measurement_index`` is accepted for compatibility with tracker callback
        conventions; the current model is spatial and does not use it.
        """
        del measurement_index
        return max(
            self.clutter_rate * self.predictive_density(measurement),
            self.intensity_floor,
        )

    def intensity_for_measurements(self, measurements):
        """Return one clutter intensity per measurement column."""
        measurements = _as_measurement_columns(measurements, self.dim)
        return array(
            [
                self(measurements[:, measurement_index])
                for measurement_index in range(measurements.shape[1])
            ]
        )

    def observe(self, measurements, weights=None):
        """Assimilate clutter observations or soft clutter responsibilities.

        Parameters
        ----------
        measurements : array-like, shape (dim,) or (dim, num_measurements)
            Clutter observations in the same column-major convention used by the
            multi-Bernoulli trackers.
        weights : array-like, optional
            Nonnegative fractional observation counts.  These are the natural
            place to feed soft clutter responsibilities from an RFS association
            posterior.
        """
        measurements = _as_measurement_columns(measurements, self.dim)
        weights = _as_observation_weights(weights, measurements.shape[1])
        for measurement_index in range(measurements.shape[1]):
            weight = float(weights[measurement_index])
            if weight == 0.0:
                continue
            self._observe_single_measurement(
                measurements[:, measurement_index],
                weight,
            )
        self.prune()
        self.cap()
        return self

    def _observe_single_measurement(self, measurement, weight):
        if self.get_number_of_components() == 0:
            self._append_component(measurement, weight)
            return

        kernel_log_densities = self._kernel_log_densities(measurement)
        existing_log_scores = np.log(self.component_counts) + kernel_log_densities
        new_log_score = log(self.concentration) + self._base_log_density(measurement)
        responsibilities = _normalize_log_weights(
            np.concatenate((existing_log_scores, np.asarray([new_log_score])))
        )
        existing_responsibilities = responsibilities[:-1]
        new_component_probability = float(responsibilities[-1])

        num_existing_components = self.get_number_of_components()
        if new_component_probability >= self.new_component_probability_threshold:
            existing_effective_counts = weight * existing_responsibilities
            self._append_component(measurement, weight * new_component_probability)
        else:
            existing_total = np.sum(existing_responsibilities)
            if existing_total <= 0.0 or not np.isfinite(existing_total):
                self._append_component(measurement, weight)
                return
            existing_effective_counts = weight * existing_responsibilities / existing_total

        for component_index in range(num_existing_components):
            effective_count = float(existing_effective_counts[component_index])
            if effective_count <= 0.0:
                continue
            old_count = float(self.component_counts[component_index])
            new_count = old_count + effective_count
            self.component_means[:, component_index] = (
                old_count * self.component_means[:, component_index]
                + effective_count * measurement
            ) / new_count
            self.component_counts[component_index] = new_count

    def _append_component(self, measurement, count):
        if count <= 0.0:
            return
        self.component_means = np.column_stack((self.component_means, measurement))
        self.component_counts = np.concatenate(
            (self.component_counts, np.asarray([float(count)]))
        )

    def prune(self, pruning_threshold=None):
        """Remove retained clutter atoms with low effective count."""
        if pruning_threshold is None:
            pruning_threshold = self.pruning_threshold
        pruning_threshold = _validate_nonnegative_scalar(
            pruning_threshold,
            "pruning_threshold",
        )
        if self.get_number_of_components() == 0:
            return
        keep_mask = self.component_counts >= pruning_threshold
        self.component_means = self.component_means[:, keep_mask]
        self.component_counts = self.component_counts[keep_mask]

    def cap(self, maximum_components=None):
        """Keep only the most-supported clutter atoms."""
        if maximum_components is None:
            maximum_components = self.maximum_components
        if maximum_components is None:
            return
        if not isinstance(maximum_components, Integral):
            raise TypeError("maximum_components must be an integer or None.")
        maximum_components = int(maximum_components)
        if maximum_components <= 0:
            raise ValueError("maximum_components must be positive when supplied.")
        if self.get_number_of_components() <= maximum_components:
            return
        keep_indices = np.argsort(self.component_counts)[-maximum_components:]
        keep_indices = keep_indices[np.argsort(self.component_counts[keep_indices])[::-1]]
        self.component_means = self.component_means[:, keep_indices]
        self.component_counts = self.component_counts[keep_indices]
