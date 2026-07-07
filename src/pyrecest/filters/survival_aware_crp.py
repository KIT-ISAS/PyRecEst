"""Survival-aware CRP-style association priors for multitarget tracking.

The utilities here are small scoring components, not complete multi-object
filters.  They replace the exchangeable ``n_k`` mass of an ordinary Chinese
restaurant process with track-context terms such as survival, visibility,
last-seen time, and measurement compatibility.

With all context terms equal to one and ``time_decay == 1``, the existing/birth
weights reduce to Dirichlet-process or Pitman--Yor predictive weights.  With
context enabled, the result is a non-exchangeable CRP-inspired prior rather than
a classical exchangeable DP.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from numbers import Integral, Real


def _as_float(value, name):
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    return float(value)


def _validate_nonnegative_real(value, name):
    value = _as_float(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _validate_positive_real(value, name):
    value = _as_float(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_probability(value, name):
    value = _as_float(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return value


def _validate_nonnegative_integer(value, name):
    if not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _coerce_nonnegative_sequence(values, name):
    try:
        return tuple(_validate_nonnegative_real(value, name) for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of nonnegative real values.") from exc


def _broadcast_values(values, length, name, validator, default):
    if values is None:
        return (validator(default, name),) * length
    if isinstance(values, Real):
        return (validator(values, name),) * length
    try:
        values = tuple(validator(value, name) for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a real scalar or an iterable of length {length}.") from exc
    if len(values) != length:
        raise ValueError(f"{name} must be scalar or have length {length}.")
    return values


def posterior_existence_after_missed_detection(
    predicted_existence,
    detection_probability,
    visibility_probability=1.0,
):
    """Return Bernoulli existence probability after a missed detection.

    Low visibility makes a missed detection weaker evidence against existence.
    """
    predicted_existence = _validate_probability(predicted_existence, "predicted_existence")
    detection_probability = _validate_probability(detection_probability, "detection_probability")
    visibility_probability = _validate_probability(visibility_probability, "visibility_probability")

    effective_detection_probability = detection_probability * visibility_probability
    denominator = 1.0 - predicted_existence * effective_detection_probability
    if denominator <= 0.0:
        return 0.0
    return predicted_existence * (1.0 - effective_detection_probability) / denominator


@dataclass(frozen=True)
class SurvivalAwareCRPAssociationPrior:
    """CRP-inspired, survival-aware association prior.

    Existing-track weights are

    ``max(m_k * time_decay ** tau_k - discount, 0)
    * survival_k * detection_k * visibility_k * compatibility_k``.

    The birth weight is ``birth_rate * (strength + discount * K)``.  A separate
    clutter alternative can be added through ``clutter_rate``.
    """

    discount: float = 0.0
    strength: float = 1.0
    time_decay: float = 1.0
    minimum_weight: float = 1e-12

    def __post_init__(self):
        discount = _as_float(self.discount, "discount")
        strength = _as_float(self.strength, "strength")
        time_decay = _validate_probability(self.time_decay, "time_decay")
        minimum_weight = _validate_positive_real(self.minimum_weight, "minimum_weight")

        if not 0.0 <= discount < 1.0:
            raise ValueError("discount must satisfy 0 <= discount < 1.")
        if strength <= -discount:
            raise ValueError("strength must satisfy strength > -discount.")
        if discount == 0.0 and strength <= 0.0:
            raise ValueError("strength must be positive when discount is zero.")

        object.__setattr__(self, "discount", discount)
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "time_decay", time_decay)
        object.__setattr__(self, "minimum_weight", minimum_weight)

    @property
    def is_dirichlet_process_base(self):
        """Return whether the context-free predictive weights are DP weights."""
        return self.discount == 0.0

    def effective_track_masses(self, track_masses, time_since_seen=None):
        """Return time-decayed effective masses for existing tracks."""
        masses = _coerce_nonnegative_sequence(track_masses, "track_masses")
        lags = _broadcast_values(
            time_since_seen,
            len(masses),
            "time_since_seen",
            _validate_nonnegative_real,
            0.0,
        )
        return tuple(mass * (self.time_decay**lag) for mass, lag in zip(masses, lags))

    def existing_track_weights(
        self,
        track_masses,
        survival_probabilities=None,
        detection_probabilities=None,
        visibility_probabilities=None,
        compatibility_scores=None,
        time_since_seen=None,
    ):
        """Return unnormalized assignment weights for existing tracks."""
        masses = self.effective_track_masses(track_masses, time_since_seen=time_since_seen)
        num_tracks = len(masses)
        survival_probabilities = _broadcast_values(
            survival_probabilities,
            num_tracks,
            "survival_probabilities",
            _validate_probability,
            1.0,
        )
        detection_probabilities = _broadcast_values(
            detection_probabilities,
            num_tracks,
            "detection_probabilities",
            _validate_probability,
            1.0,
        )
        visibility_probabilities = _broadcast_values(
            visibility_probabilities,
            num_tracks,
            "visibility_probabilities",
            _validate_probability,
            1.0,
        )
        compatibility_scores = _broadcast_values(
            compatibility_scores,
            num_tracks,
            "compatibility_scores",
            _validate_nonnegative_real,
            1.0,
        )

        return tuple(
            max(mass - self.discount, 0.0)
            * survival_probability
            * detection_probability
            * visibility_probability
            * compatibility_score
            for (
                mass,
                survival_probability,
                detection_probability,
                visibility_probability,
                compatibility_score,
            ) in zip(
                masses,
                survival_probabilities,
                detection_probabilities,
                visibility_probabilities,
                compatibility_scores,
            )
        )

    def new_track_weight(self, num_tracks, birth_rate=1.0):
        """Return the unnormalized birth/new-track weight."""
        num_tracks = _validate_nonnegative_integer(num_tracks, "num_tracks")
        birth_rate = _validate_nonnegative_real(birth_rate, "birth_rate")
        return birth_rate * (self.strength + self.discount * num_tracks)

    @staticmethod
    def clutter_weight(clutter_rate=0.0):
        """Return the unnormalized clutter alternative weight."""
        return _validate_nonnegative_real(clutter_rate, "clutter_rate")

    def assignment_weights(
        self,
        track_masses,
        survival_probabilities=None,
        detection_probabilities=None,
        visibility_probabilities=None,
        compatibility_scores=None,
        time_since_seen=None,
        birth_rate=1.0,
        clutter_rate=0.0,
    ):
        """Return ``(track_0, ..., track_K_minus_1, birth, clutter)`` weights."""
        existing_weights = self.existing_track_weights(
            track_masses,
            survival_probabilities=survival_probabilities,
            detection_probabilities=detection_probabilities,
            visibility_probabilities=visibility_probabilities,
            compatibility_scores=compatibility_scores,
            time_since_seen=time_since_seen,
        )
        return existing_weights + (
            self.new_track_weight(len(existing_weights), birth_rate=birth_rate),
            self.clutter_weight(clutter_rate),
        )

    def predictive_assignment_probabilities(
        self,
        track_masses,
        survival_probabilities=None,
        detection_probabilities=None,
        visibility_probabilities=None,
        compatibility_scores=None,
        time_since_seen=None,
        birth_rate=1.0,
        clutter_rate=0.0,
    ):
        """Return normalized existing-track, birth, and clutter probabilities."""
        weights = self.assignment_weights(
            track_masses,
            survival_probabilities=survival_probabilities,
            detection_probabilities=detection_probabilities,
            visibility_probabilities=visibility_probabilities,
            compatibility_scores=compatibility_scores,
            time_since_seen=time_since_seen,
            birth_rate=birth_rate,
            clutter_rate=clutter_rate,
        )
        total_weight = sum(weights)
        if total_weight <= 0.0:
            raise ValueError("At least one assignment alternative must have positive weight.")
        return tuple(weight / total_weight for weight in weights)

    @staticmethod
    def _coerce_compatibility_matrix(compatibility_matrix, num_tracks):
        try:
            rows = tuple(
                tuple(_validate_nonnegative_real(value, "compatibility_matrix") for value in row)
                for row in compatibility_matrix
            )
        except TypeError as exc:
            raise TypeError("compatibility_matrix must be a two-dimensional iterable of nonnegative real values.") from exc

        if len(rows) != num_tracks:
            raise ValueError(f"compatibility_matrix must have {num_tracks} rows, got {len(rows)}.")
        if not rows:
            return rows

        num_measurements = len(rows[0])
        for row in rows:
            if len(row) != num_measurements:
                raise ValueError("All compatibility_matrix rows must have equal length.")
        return rows

    def association_cost_matrix(
        self,
        track_masses,
        compatibility_matrix,
        survival_probabilities=None,
        detection_probabilities=None,
        visibility_probabilities=None,
        time_since_seen=None,
        invalid_cost=1e12,
    ):
        """Return ``-log(weight)`` costs for a track/measurement matrix.

        ``compatibility_matrix[i][j]`` is a nonnegative likelihood-like score.
        Zero-weight associations are mapped to ``invalid_cost``.
        """
        masses = _coerce_nonnegative_sequence(track_masses, "track_masses")
        rows = self._coerce_compatibility_matrix(compatibility_matrix, len(masses))
        invalid_cost = _validate_positive_real(invalid_cost, "invalid_cost")

        track_weights = self.existing_track_weights(
            masses,
            survival_probabilities=survival_probabilities,
            detection_probabilities=detection_probabilities,
            visibility_probabilities=visibility_probabilities,
            time_since_seen=time_since_seen,
        )

        costs = []
        for track_weight, row in zip(track_weights, rows):
            cost_row = []
            for compatibility_score in row:
                weight = track_weight * compatibility_score
                cost_row.append(invalid_cost if weight <= 0.0 else -log(max(weight, self.minimum_weight)))
            costs.append(tuple(cost_row))
        return tuple(costs)


__all__ = [
    "SurvivalAwareCRPAssociationPrior",
    "posterior_existence_after_missed_detection",
]
