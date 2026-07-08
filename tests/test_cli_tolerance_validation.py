from __future__ import annotations

import math

import numpy as np
import pytest
from pyrecest.cli import (
    _check_expected_mapping,
    _coerce_finite_real_sequence,
    _validate_tolerance,
)


def test_validate_tolerance_accepts_finite_nonnegative_numbers() -> None:
    assert _validate_tolerance(0.0) == 0.0
    assert _validate_tolerance(1e-8) == 1e-8
    assert _validate_tolerance(np.float64(1e-8)) == 1e-8
    assert _validate_tolerance(np.array(1e-8)) == 1e-8


def test_validate_tolerance_rejects_invalid_values() -> None:
    invalid_tolerances = (
        math.nan,
        math.inf,
        -1.0,
        True,
        "0.1",
        np.bool_(True),
        np.array(True, dtype=object),
        np.array("0.1", dtype=object),
        np.array([0.1]),
    )
    for tolerance in invalid_tolerances:
        with pytest.raises(ValueError, match="tolerance"):
            _validate_tolerance(tolerance)


def test_validate_tolerance_rejects_numpy_temporal_scalars() -> None:
    temporal_tolerances = (
        np.timedelta64(1, "ns"),
        np.datetime64("1970-01-01T00:00:00.000000001", "ns"),
        np.array(np.timedelta64(1, "ns"), dtype=object),
        np.array(np.datetime64("1970-01-01T00:00:00.000000001", "ns"), dtype=object),
    )
    for tolerance in temporal_tolerances:
        with pytest.raises(ValueError, match="tolerance"):
            _validate_tolerance(tolerance)


def test_expected_mapping_rejects_nan_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        _check_expected_mapping(
            "metrics",
            {"rmse": 1.0},
            {"rmse": 2.0},
            tolerance=math.nan,
        )


def test_expected_mapping_accepts_finite_numeric_actuals() -> None:
    assert (
        _check_expected_mapping(
            "metrics",
            {"rmse": 1.000001},
            {"rmse": 1.0},
            tolerance=1e-5,
        )
        == []
    )


@pytest.mark.parametrize("actual_value", [np.float64(1.000001), np.int64(1)])
def test_expected_mapping_accepts_numpy_scalar_actuals(actual_value) -> None:
    assert (
        _check_expected_mapping(
            "metrics",
            {"rmse": actual_value},
            {"rmse": 1.0},
            tolerance=1e-5,
        )
        == []
    )


@pytest.mark.parametrize(
    "actual_value",
    [True, "1.0", math.nan, math.inf, np.timedelta64(1, "ns")],
)
def test_expected_mapping_rejects_malformed_numeric_actuals(actual_value) -> None:
    errors = _check_expected_mapping(
        "metrics",
        {"rmse": actual_value},
        {"rmse": 1.0},
        tolerance=0.0,
    )

    assert errors
    assert "metrics.rmse mismatch" in errors[0]


def test_expected_mapping_does_not_compare_temporal_expected_as_numeric() -> None:
    errors = _check_expected_mapping(
        "metrics",
        {"duration": 1.0},
        {"duration": np.timedelta64(1, "ns")},
        tolerance=0.0,
    )

    assert errors
    assert "metrics.duration mismatch" in errors[0]


def test_coerce_finite_real_sequence_rejects_temporal_entries() -> None:
    with pytest.raises(ValueError, match="final_estimate"):
        _coerce_finite_real_sequence(
            [np.timedelta64(1, "ns")],
            field_name="final_estimate",
        )
