from __future__ import annotations

import numpy as np
import pytest
from pyrecest.smoothers.record_smoother import smooth_records


def _transition(dt: float, state_dim: int) -> np.ndarray:
    assert state_dim == 2
    return np.array([[1.0, dt], [0.0, 1.0]])


def _process_noise(dt: float, state_dim: int) -> np.ndarray:
    assert state_dim == 2
    return 0.01 * max(dt, 0.0) * np.eye(2)


def _records() -> list[dict[str, object]]:
    return [
        {
            "time_s": 0.0,
            "state": np.array([0.0, 1.0]),
            "covariance": np.eye(2),
        },
        {
            "time_s": 1.0,
            "state": np.array([1.0, 1.0]),
            "covariance": np.eye(2),
        },
    ]


@pytest.mark.parametrize("method", ["none", "rts"])
@pytest.mark.parametrize(
    "metadata",
    [
        {"state": "corrupted"},
        {"covariance": "corrupted"},
        {"filtered_state": "corrupted"},
        {"filtered_covariance": "corrupted"},
    ],
)
def test_metadata_must_not_overwrite_state_or_covariance_fields(
    method: str, metadata: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="metadata keys must not overwrite"):
        smooth_records(
            _records(),
            method=method,
            transition_model=_transition,
            process_noise_model=_process_noise,
            metadata=metadata,
        )


def test_custom_output_keys_are_also_reserved() -> None:
    with pytest.raises(ValueError, match="smoothed_state"):
        smooth_records(
            _records(),
            method="rts",
            transition_model=_transition,
            process_noise_model=_process_noise,
            output_state_key="smoothed_state",
            metadata={"smoothed_state": "corrupted"},
        )


def test_nonconflicting_metadata_is_preserved() -> None:
    out = smooth_records(
        _records(),
        method="rts",
        transition_model=_transition,
        process_noise_model=_process_noise,
        metadata={"smoother_method": "rts"},
    )

    assert out[0]["smoother_method"] == "rts"
    assert isinstance(out[0]["state"], np.ndarray)
    assert isinstance(out[0]["filtered_state"], np.ndarray)
