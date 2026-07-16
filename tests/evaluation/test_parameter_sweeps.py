import pytest

from pyrecest.evaluation import summarize_parameter_sweep_records


def test_summarize_parameter_sweep_records_groups_metrics():
    records = [
        {"method": "A", "parameter": 10, "error": 0.2, "runtime_s": 1.0},
        {"method": "A", "parameter": 10, "error": 0.4, "runtime_s": 3.0},
        {"method": "B", "parameter": 20, "error": 0.1, "runtime_s": 2.0},
    ]

    summaries = summarize_parameter_sweep_records(records, metric_names=["error", "runtime_s"])

    assert summaries == [
        {
            "method": "A",
            "parameter": 10,
            "n_repetitions": 2,
            "error_mean": 0.30000000000000004,
            "error_std": 0.1,
            "error_median": 0.30000000000000004,
            "error_q25": 0.25,
            "error_q75": 0.35000000000000003,
            "runtime_s_mean": 2.0,
            "runtime_s_std": 1.0,
            "runtime_s_median": 2.0,
            "runtime_s_q25": 1.5,
            "runtime_s_q75": 2.5,
        },
        {
            "method": "B",
            "parameter": 20,
            "n_repetitions": 1,
            "error_mean": 0.1,
            "error_std": 0.0,
            "error_median": 0.1,
            "error_q25": 0.1,
            "error_q75": 0.1,
            "runtime_s_mean": 2.0,
            "runtime_s_std": 0.0,
            "runtime_s_median": 2.0,
            "runtime_s_q25": 2.0,
            "runtime_s_q75": 2.0,
        },
    ]


def test_summarize_parameter_sweep_records_rejects_empty_metrics():
    with pytest.raises(ValueError, match="metric_names"):
        summarize_parameter_sweep_records([], metric_names=[])


def test_summarize_parameter_sweep_records_rejects_nonfinite_values():
    records = [{"method": "A", "parameter": 1, "error": float("nan")}]

    with pytest.raises(ValueError, match="error"):
        summarize_parameter_sweep_records(records, metric_names=["error"])
