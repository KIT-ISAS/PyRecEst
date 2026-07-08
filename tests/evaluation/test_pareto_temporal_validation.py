from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyrecest.evaluation import (
    constraint_mask,
    equal_quality_selection,
    is_pareto_front,
    pareto_front_indices,
    record_dominates,
    select_under_constraints,
)


def _small_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name": "a", "error": 1.0, "runtime": 2.0},
            {"name": "b", "error": 2.0, "runtime": 1.0},
        ]
    )


@pytest.mark.parametrize(
    "eps",
    [
        np.timedelta64(1, "ns"),
        np.array(np.timedelta64(1, "ns"), dtype=object),
        np.datetime64("2026-01-01"),
    ],
)
def test_pareto_helpers_reject_temporal_scalar_eps(eps: object) -> None:
    table = _small_table()

    with pytest.raises(ValueError, match="eps"):
        pareto_front_indices(
            table,
            ["error", "runtime"],
            directions={"error": "min", "runtime": "min"},
            eps=eps,
        )
    with pytest.raises(ValueError, match="eps"):
        is_pareto_front(
            table,
            ["error", "runtime"],
            directions={"error": "min", "runtime": "min"},
            eps=eps,
        )
    with pytest.raises(ValueError, match="eps"):
        record_dominates(
            table.iloc[0],
            table.iloc[1],
            ["error", "runtime"],
            directions={"error": "min", "runtime": "min"},
            eps=eps,
        )
    with pytest.raises(ValueError, match="eps"):
        constraint_mask(table, {"error": ("<=", 2.0)}, eps=eps)
    with pytest.raises(ValueError, match="eps"):
        select_under_constraints(
            table,
            constraints={"error": ("<=", 2.0)},
            objective="runtime",
            direction="min",
            eps=eps,
        )
    with pytest.raises(ValueError, match="eps"):
        equal_quality_selection(
            table,
            quality_constraints={"error": ("<=", 2.0)},
            compression_objective="runtime",
            eps=eps,
        )


@pytest.mark.parametrize(
    "threshold",
    [
        np.timedelta64(2, "ns"),
        np.array(np.timedelta64(2, "ns"), dtype=object),
        np.datetime64("2026-01-01"),
    ],
)
def test_pareto_constraint_thresholds_reject_temporal_scalars(
    threshold: object,
) -> None:
    table = pd.DataFrame([{"score": 1.0}])

    with pytest.raises(
        ValueError,
        match="Constraint threshold for 'score' must be a finite scalar",
    ):
        constraint_mask(table, {"score": ("<=", threshold)})
    with pytest.raises(
        ValueError,
        match="Constraint threshold for 'score' must be a finite scalar",
    ):
        select_under_constraints(
            table,
            constraints={"score": ("<=", threshold)},
            objective="score",
            direction="min",
        )
    with pytest.raises(
        ValueError,
        match="Constraint threshold for 'score' must be a finite scalar",
    ):
        equal_quality_selection(
            table,
            quality_constraints={"score": ("<=", threshold)},
            compression_objective="score",
        )


def test_pareto_constraints_treat_temporal_columns_as_missing() -> None:
    table = pd.DataFrame(
        {
            "score": [np.timedelta64(1, "ns"), np.timedelta64(10, "ns")],
        }
    )

    mask = constraint_mask(table, {"score": ("<=", 5.0)})
    selected = select_under_constraints(
        table,
        constraints={"score": ("<=", 5.0)},
        objective="score",
        direction="min",
    )

    assert mask.tolist() == [False, False]
    assert selected.empty


def test_temporal_object_objectives_are_not_treated_as_numeric() -> None:
    left = {"runtime": np.array(np.timedelta64(1, "ns"), dtype=object)}
    right = {"runtime": 2.0}

    assert not record_dominates(
        left,
        right,
        ["runtime"],
        directions={"runtime": "min"},
    )
    assert not record_dominates(
        left,
        right,
        ["runtime"],
        directions={"runtime": "min"},
        allow_missing=False,
    )
