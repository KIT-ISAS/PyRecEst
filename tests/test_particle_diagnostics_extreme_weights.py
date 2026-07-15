from math import isclose, isfinite, log

import numpy as np
from pyrecest.diagnostics import ParticleDiagnostics


def test_particle_diagnostics_normalizes_extreme_finite_weights():
    maximum = float(np.finfo(float).max)
    weights = [maximum, maximum / 2.0]

    assert all(isfinite(weight) for weight in weights)
    assert not isfinite(sum(weights))

    diagnostics = ParticleDiagnostics.from_weights(weights)
    expected_entropy = -(2.0 / 3.0 * log(2.0 / 3.0)) - (
        1.0 / 3.0 * log(1.0 / 3.0)
    )

    assert isclose(diagnostics.effective_sample_size, 1.8)
    assert isclose(diagnostics.weight_entropy, expected_entropy)
