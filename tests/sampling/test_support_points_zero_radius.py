from __future__ import annotations

import numpy as np
from pyrecest.sampling import ellipsoid_sigma_points


def test_zero_radius_contributes_center_only_once() -> None:
    support = ellipsoid_sigma_points(
        [1.0, -2.0],
        np.eye(2),
        radii=(0.0, 1.0, 0.0),
    )

    center = np.asarray([1.0, -2.0])
    assert support.shape == (5, 2)
    assert np.count_nonzero(np.all(support == center, axis=1)) == 1


def test_zero_radius_is_preserved_when_center_flag_is_false() -> None:
    support = ellipsoid_sigma_points(
        [1.0, -2.0],
        np.eye(2),
        radii=(0.0,),
        include_center=False,
    )

    assert support.shape == (1, 2)
    assert np.array_equal(support[0], [1.0, -2.0])


def test_zero_radius_is_deduplicated_for_batched_inputs() -> None:
    means = np.asarray([[0.0, 0.0], [2.0, 3.0]])
    covariances = np.broadcast_to(np.eye(2), (2, 2, 2)).copy()

    support = ellipsoid_sigma_points(
        means,
        covariances,
        radii=(0.0, 2.0),
        include_center=False,
    )

    assert support.shape == (2, 5, 2)
    for batch_index, center in enumerate(means):
        assert np.count_nonzero(np.all(support[batch_index] == center, axis=1)) == 1
        assert np.allclose(
            np.linalg.norm(support[batch_index, 1:] - center, axis=1),
            2.0,
        )
