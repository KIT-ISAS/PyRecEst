"""Regression coverage for fixed-lag Viterbi prefix commitment."""

from pyrecest.filters.tracklet_viterbi import (
    TrackletAssociationCandidate,
    TrackletViterbiConfig,
    solve_fixed_lag_tracklet_viterbi,
)


def test_fixed_lag_prefix_cannot_be_replaced_by_missed_detection():
    committed = TrackletAssociationCandidate(
        "a0",
        unary_cost=0.0,
        track_id="A",
        time_s=0.0,
    )
    switch = TrackletAssociationCandidate(
        "b1",
        unary_cost=0.0,
        track_id="B",
        time_s=10.0,
    )

    result = solve_fixed_lag_tracklet_viterbi(
        [[committed], [switch]],
        lag_s=1.0,
        config=TrackletViterbiConfig(
            missed_detection_cost=5.0,
            consecutive_miss_cost=0.0,
            switch_cost=100.0,
        ),
    )

    assert result.path == [committed, None]
    assert result.total_cost == 5.0
