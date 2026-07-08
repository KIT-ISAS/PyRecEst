import numpy as np
import pytest
from pyrecest.calibration.bias import (
    SensorBiasCorrectionModel,
    make_bias_training_examples,
)


def _unit_feature_model_kwargs():
    return {
        "target_dim": 1,
        "feature_dim": 1,
        "intercept": np.array([0.0]),
        "coefficients": np.array([[1.0]]),
        "feature_mean": np.array([0.0]),
        "feature_scale": np.array([1.0]),
        "residual_std": np.array([0.0]),
        "training_count": 2,
        "ridge_alpha": 0.0,
    }


def test_model_rejects_text_array_field_before_float_coercion():
    kwargs = _unit_feature_model_kwargs()
    kwargs["intercept"] = np.array(["0.0"])

    with pytest.raises(ValueError) as exc_info:
        SensorBiasCorrectionModel(**kwargs)

    assert "intercept must contain numeric values" in str(exc_info.value)


def test_model_rejects_boolean_array_field_before_float_coercion():
    kwargs = _unit_feature_model_kwargs()
    kwargs["coefficients"] = np.array([[True]])

    with pytest.raises(ValueError) as exc_info:
        SensorBiasCorrectionModel(**kwargs)

    assert "coefficients must contain numeric values" in str(exc_info.value)


def test_make_bias_training_examples_rejects_text_times_before_float_coercion():
    with pytest.raises(ValueError) as exc_info:
        make_bias_training_examples(
            measurement_times_s=np.array(["0.0", "1.0"]),
            measurement_values=np.array([[1.0], [2.0]]),
            reference_times_s=np.array([0.0, 1.0]),
            reference_values=np.array([[1.5], [2.5]]),
        )

    assert "measurement_times_s must contain numeric values" in str(exc_info.value)


def test_make_bias_training_examples_rejects_non_real_measurements():
    invalid_measurements = (
        np.array([[1.0 + 2.0j]]),
        np.array([[np.complex128(1.0 + 2.0j)]], dtype=object),
        np.array([[np.datetime64("2026-01-01")]], dtype=object),
        np.array([[np.timedelta64(1, "s")]], dtype=object),
    )

    for measurement_values in invalid_measurements:
        with pytest.raises(ValueError) as exc_info:
            make_bias_training_examples(
                measurement_times_s=np.array([0.0]),
                measurement_values=measurement_values,
                reference_times_s=np.array([0.0]),
                reference_values=np.array([[0.0]]),
                max_time_delta_s=0.0,
            )

        assert "measurement_values must contain numeric values" in str(exc_info.value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "target_dim",
            np.datetime64("1970-01-01T00:00:00.000000001"),
            "target_dim must be a nonnegative integer",
        ),
        (
            "feature_dim",
            np.timedelta64(1, "ns"),
            "feature_dim must be a nonnegative integer",
        ),
        (
            "training_count",
            np.timedelta64(2, "ns"),
            "training_count must be a nonnegative integer",
        ),
        (
            "ridge_alpha",
            np.datetime64("1970-01-01T00:00:00.000000001"),
            "ridge_alpha must be a nonnegative finite scalar",
        ),
    ],
)
def test_model_rejects_temporal_scalar_fields_before_integer_payload_coercion(
    field, value, message
):
    kwargs = _unit_feature_model_kwargs()
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        SensorBiasCorrectionModel(**kwargs)


def test_constant_bias_predict_rejects_temporal_n_rows_before_integer_payload():
    model = SensorBiasCorrectionModel(
        target_dim=1,
        feature_dim=0,
        intercept=np.array([2.0]),
        coefficients=np.empty((0, 1)),
        feature_mean=np.empty(0),
        feature_scale=np.empty(0),
        residual_std=np.array([0.0]),
        training_count=3,
        ridge_alpha=0.0,
    )

    with pytest.raises(ValueError, match="n_rows must be a nonnegative integer"):
        model.predict(n_rows=np.timedelta64(2, "ns"))


def test_make_bias_training_examples_accepts_real_integer_measurements():
    examples = make_bias_training_examples(
        measurement_times_s=np.array([0.0]),
        measurement_values=np.array([[2]], dtype=np.int64),
        reference_times_s=np.array([0.0]),
        reference_values=np.array([[1.0]]),
        max_time_delta_s=0.0,
    )

    assert examples.measured.shape == (1, 1)
    assert float(examples.residual[0, 0]) == 1.0
