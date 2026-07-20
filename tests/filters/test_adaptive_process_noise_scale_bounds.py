import pytest
from pyrecest.filters.adaptive_process_noise import (
    AdaptiveProcessNoiseConfig,
    RollingNISProcessNoiseAdapter,
)


@pytest.mark.parametrize(
    ("base_scale", "nis", "expected_scale"),
    (
        (2.0, 10.0, 3.0),
        (0.5, 0.0, 0.5),
    ),
)
def test_rolling_adapter_clamps_final_scale_to_configured_bounds(
    base_scale,
    nis,
    expected_scale,
):
    config = AdaptiveProcessNoiseConfig(
        base_scale=base_scale,
        min_scale=0.5,
        max_scale=3.0,
        ewma_alpha=1.0,
        high_nis_ratio=1.5,
        low_nis_ratio=0.5,
        scale_gain=1.0,
    )
    adapter = RollingNISProcessNoiseAdapter(config)

    adapter.observe(measurement_dim=1, nis=nis)

    assert adapter.scale() == expected_scale


def test_rolling_adapter_preserves_nonunit_nominal_base_scale():
    config = AdaptiveProcessNoiseConfig(
        base_scale=2.0,
        min_scale=0.5,
        max_scale=3.0,
    )

    assert RollingNISProcessNoiseAdapter(config).scale() == 2.0
