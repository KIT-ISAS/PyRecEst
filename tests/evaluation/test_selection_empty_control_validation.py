from __future__ import annotations

import numpy as np
import pytest
from pyrecest.evaluation.selection import (
    protected_tail_topk_mask,
    tail_rescue_topk_mask,
)

_EMPTY_SCORES = np.empty(0, dtype=float)


def test_protected_tail_topk_validates_controls_for_empty_inputs() -> None:
    invalid_calls = (
        ({"retention_fraction": -0.1, "tail_quantile": 0.5}, "retention_fraction"),
        ({"retention_fraction": 0.5, "tail_quantile": 0.0}, "quantile"),
        (
            {"retention_fraction": 0.5, "tail_quantile": 0.5, "min_count": True},
            "min_count",
        ),
    )

    for kwargs, error_match in invalid_calls:
        with pytest.raises(ValueError, match=error_match):
            protected_tail_topk_mask(
                _EMPTY_SCORES,
                _EMPTY_SCORES,
                _EMPTY_SCORES,
                **kwargs,
            )

    result = protected_tail_topk_mask(
        _EMPTY_SCORES,
        _EMPTY_SCORES,
        _EMPTY_SCORES,
        0.5,
        tail_quantile=0.5,
    )
    assert result.shape == (0,)


def test_tail_rescue_topk_validates_controls_for_empty_inputs() -> None:
    invalid_calls = (
        (
            {
                "retention_fraction": -0.1,
                "tail_quantile": 0.5,
                "rescue_fraction": 0.5,
            },
            "retention_fraction",
        ),
        (
            {
                "retention_fraction": 0.5,
                "tail_quantile": 0.0,
                "rescue_fraction": 0.5,
            },
            "quantile",
        ),
        (
            {
                "retention_fraction": 0.5,
                "tail_quantile": 0.5,
                "rescue_fraction": 0.0,
            },
            "rescue_fraction",
        ),
        (
            {
                "retention_fraction": 0.5,
                "tail_quantile": 0.5,
                "rescue_fraction": 0.5,
                "min_count": True,
            },
            "min_count",
        ),
    )

    for kwargs, error_match in invalid_calls:
        with pytest.raises(ValueError, match=error_match):
            tail_rescue_topk_mask(
                _EMPTY_SCORES,
                _EMPTY_SCORES,
                _EMPTY_SCORES,
                **kwargs,
            )

    result = tail_rescue_topk_mask(
        _EMPTY_SCORES,
        _EMPTY_SCORES,
        _EMPTY_SCORES,
        0.5,
        tail_quantile=0.5,
        rescue_fraction=0.5,
    )
    assert result.shape == (0,)
