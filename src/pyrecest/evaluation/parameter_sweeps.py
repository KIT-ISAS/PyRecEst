"""Utilities for summarizing parameter-sweep evaluation records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np


def summarize_parameter_sweep_records(
    records: Iterable[Mapping[str, object]],
    *,
    metric_names: Sequence[str],
    method_key: str = "method",
    parameter_key: str = "parameter",
) -> list[dict[str, object]]:
    """Aggregate flat per-run records by method and sweep parameter.

    Parameters
    ----------
    records
        Iterable of mapping-like records. Each record must contain the method,
        parameter, and all requested metric fields.
    metric_names
        Names of numeric metrics to aggregate. For each metric ``m``, the
        returned dictionaries contain ``m_mean``, ``m_std``, ``m_median``,
        ``m_q25``, and ``m_q75``.
    method_key, parameter_key
        Record keys used for grouping.
    """

    metric_names = tuple(metric_names)
    if not metric_names:
        raise ValueError("metric_names must contain at least one metric.")

    grouped: dict[tuple[str, float], list[Mapping[str, object]]] = {}
    for record in records:
        method = str(_required(record, method_key))
        parameter = _as_finite_float(_required(record, parameter_key), parameter_key)
        for metric_name in metric_names:
            _as_finite_float(_required(record, metric_name), metric_name)
        grouped.setdefault((method, parameter), []).append(record)

    summaries = []
    for (method, parameter), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary: dict[str, object] = {
            method_key: method,
            parameter_key: _restore_integer_parameter(parameter),
            "n_repetitions": len(group),
        }
        for metric_name in metric_names:
            values = np.asarray([_as_finite_float(record[metric_name], metric_name) for record in group], dtype=float)
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_std"] = float(np.std(values))
            summary[f"{metric_name}_median"] = float(np.median(values))
            summary[f"{metric_name}_q25"] = float(np.quantile(values, 0.25))
            summary[f"{metric_name}_q75"] = float(np.quantile(values, 0.75))
        summaries.append(summary)
    return summaries


def _required(record: Mapping[str, object], key: str):
    if key not in record:
        raise KeyError(f"Missing required key {key!r}.")
    return record[key]


def _as_finite_float(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, got bool.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _restore_integer_parameter(parameter: float) -> int | float:
    rounded = round(parameter)
    if abs(parameter - rounded) <= 1e-12:
        return int(rounded)
    return float(parameter)
