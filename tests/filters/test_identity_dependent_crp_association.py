import numpy as np

from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import (
    IdentityDependentCRPParams,
    KalmanFilter,
    Track,
    TrackManager,
    TrackStatus,
    build_identity_dependent_crp_associator,
    build_kalman_measurement_initiator,
    identity_dependent_crp_hypotheses,
)


def _track(track_id, mean, *, hits=1, misses=0):
    return Track(
        track_id=track_id,
        single_target_filter=KalmanFilter(
            GaussianDistribution(np.asarray(mean, dtype=float), np.eye(len(mean)))
        ),
        status=TrackStatus.CONFIRMED,
        hits=hits,
        misses=misses,
    )


def test_aging_penalty_can_overcome_large_hit_count():
    stale_track = _track(0, [0.0], hits=40, misses=5)
    fresh_track = _track(1, [0.0], hits=2, misses=0)
    params = IdentityDependentCRPParams(
        count_power=1.0,
        miss_decay=2.0,
        survival_probability=0.99,
        detection_probability=0.9,
        clutter_intensity=1.0,
    )

    hypotheses = identity_dependent_crp_hypotheses(
        [stale_track, fresh_track],
        np.array([[0.0]]),
        np.eye(1),
        np.eye(1),
        params,
    )

    costs = {hypothesis.track_index: hypothesis.cost for hypothesis in hypotheses}
    assert costs[1] < costs[0]


def test_appearance_callback_can_bias_identity_score():
    first_track = _track(0, [0.0], hits=1, misses=0)
    second_track = _track(1, [0.0], hits=1, misses=0)
    params = IdentityDependentCRPParams(
        count_power=0.0,
        miss_decay=0.0,
        detection_probability=0.9,
        clutter_intensity=1.0,
    )

    def appearance_log_likelihood(track, **kwargs):
        del kwargs
        return 5.0 if track.track_id == 1 else 0.0

    hypotheses = identity_dependent_crp_hypotheses(
        [first_track, second_track],
        np.array([[0.0]]),
        np.eye(1),
        np.eye(1),
        params,
        appearance_log_likelihood=appearance_log_likelihood,
    )

    costs = {hypothesis.track_index: hypothesis.cost for hypothesis in hypotheses}
    assert costs[1] < costs[0]
    preferred = min(hypotheses, key=lambda hypothesis: hypothesis.cost)
    assert preferred.metadata["identity_dependent_crp_log_appearance"] == 5.0


def test_associator_leaves_gated_out_measurement_unmatched_for_birth():
    track = _track(0, [0.0], hits=5, misses=0)
    params = IdentityDependentCRPParams(
        concentration=1.0,
        detection_probability=0.9,
        clutter_intensity=1.0,
        birth_log_prior=0.0,
    )
    associator = build_identity_dependent_crp_associator(
        np.eye(1),
        0.01 * np.eye(1),
        params,
    )

    association = associator(
        [track],
        np.array([[100.0]]),
        gates=None,
    )

    assert association.matches == []
    assert association.unmatched_track_indices == [0]
    assert association.unmatched_measurement_indices == [0]


def test_track_manager_births_unmatched_measurement_with_crp_associator():
    params = IdentityDependentCRPParams(
        concentration=1.0,
        detection_probability=0.9,
        clutter_intensity=1.0,
        birth_log_prior=0.0,
    )
    manager = TrackManager(
        initiator=build_kalman_measurement_initiator(np.eye(1)),
        associator=build_identity_dependent_crp_associator(
            np.eye(1),
            0.01 * np.eye(1),
            params,
        ),
        n_init=1,
        max_misses=2,
        allow_births=True,
    )
    existing_id = manager.add_track(
        GaussianDistribution(np.array([0.0]), np.eye(1)),
        status=TrackStatus.CONFIRMED,
    )

    result = manager.step([np.array([100.0])])

    assert result.missed_track_ids == [existing_id]
    assert len(result.born_track_ids) == 1
    assert manager.get_number_of_targets(confirmed_only=False) == 2
