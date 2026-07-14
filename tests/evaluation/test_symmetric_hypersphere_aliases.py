from __future__ import annotations

import pytest
from pyrecest.backend import array, to_numpy
from pyrecest.evaluation.get_distance_function import get_distance_function
from pyrecest.evaluation.get_extract_mean import get_extract_mean


def _as_float(value):
    return float(to_numpy(value))


@pytest.mark.parametrize(
    "manifold_name",
    ("symmetric_hypersphere", "symm_hypersphere"),
)
def test_prefixed_symmetric_hypersphere_distance_is_antipodal_invariant(
    manifold_name: str,
) -> None:
    distance = get_distance_function(manifold_name)

    distance_value = distance(array([1.0, 0.0]), array([-1.0, 0.0]))

    assert _as_float(distance_value) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "manifold_name",
    ("symmetric_hypersphere", "symm_hypersphere"),
)
def test_prefixed_symmetric_hypersphere_extract_mean_requires_convention(
    manifold_name: str,
) -> None:
    with pytest.raises(
        NotImplementedError,
        match="explicit convention|custom extractor",
    ):
        get_extract_mean(manifold_name)


@pytest.mark.parametrize(
    "manifold_name",
    (
        "circle_symmetric",
        "symmetric_circle",
        "hypertorus_symmetric",
        "symm_hypertorus",
    ),
)
def test_symmetric_toroidal_extract_mean_requires_convention(
    manifold_name: str,
) -> None:
    with pytest.raises(NotImplementedError, match="explicit convention"):
        get_extract_mean(manifold_name)
