from fractions import Fraction

import numpy as np
import pytest
from pyrecest.experimental.dvs import (
    count_negative_log_likelihood,
    edge_probabilities_from_activity,
    simulate_rectangle_event_counts,
    uniform_edge_probabilities,
)

_EDGES = ("left", "right", "top", "bottom")
_ROUNDED_HALF_INTEGER = Fraction(2**54 + 1, 2)


def test_motion_gated_counts_concentrate_on_vertical_edges():
    simulation = simulate_rectangle_event_counts(
        np.array([1.0, 0.0]),
        total_events=200,
        background_activity=0.0,
        seed=1,
    )

    assert (
        simulation.observed_counts["left"] + simulation.observed_counts["right"] == 200
    )
    assert simulation.observed_counts["top"] + simulation.observed_counts["bottom"] == 0


def test_normal_flow_likelihood_beats_uniform_for_horizontal_motion():
    simulation = simulate_rectangle_event_counts(
        np.array([1.0, 0.0]),
        total_events=200,
        background_activity=1e-3,
        seed=2,
    )

    normal_flow_nll = count_negative_log_likelihood(
        simulation.observed_counts, simulation.normal_flow_probabilities
    )
    uniform_nll = count_negative_log_likelihood(
        simulation.observed_counts, simulation.uniform_probabilities
    )

    assert normal_flow_nll < uniform_nll


def test_uniform_edge_probabilities_sum_to_one():
    probabilities = uniform_edge_probabilities(
        ["left", "left", "right", "right", "top", "bottom"]
    )

    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_count_nll_rejects_invalid_observed_counts():
    probabilities = {edge: 0.25 for edge in _EDGES}

    for bad_count in (
        -1,
        1.5,
        _ROUNDED_HALF_INTEGER,
        np.nan,
        np.inf,
        True,
        "2",
        np.array([2]),
    ):
        observed_counts = {edge: 1 for edge in _EDGES}
        observed_counts["left"] = bad_count
        with pytest.raises(ValueError, match="observed_counts"):
            count_negative_log_likelihood(observed_counts, probabilities)


def test_count_nll_rejects_invalid_probabilities():
    observed_counts = {edge: 1 for edge in _EDGES}

    for bad_probability in (-0.1, 1.1, np.nan, np.inf, True, "0.25", 0.25 + 0.1j):
        probabilities = {edge: 0.25 for edge in _EDGES}
        probabilities["left"] = bad_probability
        with pytest.raises(ValueError, match="probabilities"):
            count_negative_log_likelihood(observed_counts, probabilities)


def test_count_nll_rejects_invalid_probability_floor():
    observed_counts = {edge: 1 for edge in _EDGES}
    probabilities = {edge: 0.25 for edge in _EDGES}

    for probability_floor in (0.0, -1e-3, 1.1, np.nan, np.inf, True, "1e-12"):
        with pytest.raises(ValueError, match="probability_floor"):
            count_negative_log_likelihood(
                observed_counts,
                probabilities,
                probability_floor=probability_floor,
            )


def test_count_nll_uses_probability_floor_for_zero_probability():
    counts = {edge: 0 for edge in _EDGES}
    counts["left"] = 2
    probabilities = {edge: 0.25 for edge in _EDGES}
    probabilities["left"] = 0.0

    nll = count_negative_log_likelihood(counts, probabilities, probability_floor=1e-3)

    assert nll == pytest.approx(-2.0 * np.log(1e-3))


def test_edge_probabilities_from_activity_rejects_invalid_background_activity():
    edge_labels = list(_EDGES)
    activities = np.ones(len(edge_labels), dtype=float)

    for bad_background in (np.nan, np.inf, -1.0, True, "0.1", np.array([0.1])):
        with pytest.raises(ValueError, match="background_activity"):
            edge_probabilities_from_activity(
                edge_labels,
                activities,
                background_activity=bad_background,
            )


def test_edge_probabilities_from_activity_rejects_invalid_weights():
    for bad_activities in (
        [np.nan, 1.0, 1.0, 1.0],
        [np.inf, 1.0, 1.0, 1.0],
        [-2.0, 1.0, 1.0, 1.0],
    ):
        with pytest.raises(ValueError, match="point_weights"):
            edge_probabilities_from_activity(list(_EDGES), np.array(bad_activities))


def test_simulate_rectangle_event_counts_rejects_invalid_total_events():
    for bad_total_events in (
        0,
        -1,
        1.5,
        _ROUNDED_HALF_INTEGER,
        np.nan,
        np.inf,
        True,
        "4",
    ):
        with pytest.raises(ValueError, match="total_events"):
            simulate_rectangle_event_counts(
                np.array([1.0, 0.0]),
                total_events=bad_total_events,
                samples_per_edge=2,
            )
