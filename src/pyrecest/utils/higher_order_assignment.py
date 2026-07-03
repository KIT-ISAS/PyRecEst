"""Higher-order consistency helpers for multi-session assignment costs.

The utilities in this module operate only on generic pairwise cost matrices.  A
candidate edge between two sessions is penalized when compatible low-cost support
through a third session is absent.  This is useful for multi-session assignment
problems where direct skip-session or pairwise links should agree with surrounding
pairwise evidence, but no domain-specific observation semantics are required.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

SessionEdge = tuple[int, int]


@dataclass(frozen=True)
class HigherOrderConsistencyConfig:
    """Configuration for triplet-projected higher-order cost penalties.

    The unweighted penalty for an edge is ``max(0, best_support -
    support_cost_cap)``, clipped to ``max_penalty``.  ``triplet_weight`` scales
    that penalty before it is added to admissible direct-edge costs.
    """

    triplet_weight: float = 0.0
    support_top_k: int = 8
    support_cost_cap: float = 4.0
    max_penalty: float = 2.0
    large_cost: float = 1.0e6

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "triplet_weight",
            _finite_nonnegative_float(self.triplet_weight, name="triplet_weight"),
        )
        object.__setattr__(
            self,
            "support_top_k",
            _positive_integer(self.support_top_k, name="support_top_k"),
        )
        object.__setattr__(
            self,
            "support_cost_cap",
            _finite_nonnegative_float(
                self.support_cost_cap,
                name="support_cost_cap",
            ),
        )
        object.__setattr__(
            self,
            "max_penalty",
            _finite_nonnegative_float(self.max_penalty, name="max_penalty"),
        )
        object.__setattr__(
            self,
            "large_cost",
            _finite_positive_float(self.large_cost, name="large_cost"),
        )

    @property
    def enabled(self) -> bool:
        """Return whether this configuration changes pairwise costs."""

        return self.triplet_weight > 0.0 and self.max_penalty > 0.0


def apply_higher_order_consistency(
    pairwise_costs: Mapping[SessionEdge, Any],
    *,
    session_sizes: Sequence[int],
    config: HigherOrderConsistencyConfig | Mapping[str, Any] | None = None,
) -> dict[SessionEdge, np.ndarray]:
    """Return pairwise costs with triplet-support penalties added.

    Every input edge is preserved.  Edges without any available third-session
    context are copied unchanged.  The function is shape-preserving and does not
    run an assignment solver; callers can pass the adjusted cost matrices to any
    compatible downstream assignment routine.
    """

    resolved = higher_order_consistency_config_from_mapping(config)
    copied_costs = {
        _normalise_edge(edge): np.asarray(matrix, dtype=float).copy()
        for edge, matrix in pairwise_costs.items()
    }
    _validate_pairwise_shapes(copied_costs, session_sizes=session_sizes)

    if not resolved.enabled:
        return copied_costs

    adjusted: dict[SessionEdge, np.ndarray] = {}
    for edge, matrix in copied_costs.items():
        penalty = triplet_consistency_penalty(
            copied_costs,
            edge=edge,
            session_sizes=session_sizes,
            config=resolved,
        )
        if penalty is None:
            adjusted[edge] = matrix.copy()
            continue
        admissible = _admissible_cost_mask(matrix, large_cost=resolved.large_cost)
        edge_costs = matrix.copy()
        edge_costs[admissible] = (
            edge_costs[admissible] + resolved.triplet_weight * penalty[admissible]
        )
        adjusted[edge] = edge_costs
    return adjusted


def triplet_consistency_penalty(
    pairwise_costs: Mapping[SessionEdge, Any],
    *,
    edge: SessionEdge,
    session_sizes: Sequence[int],
    config: HigherOrderConsistencyConfig | Mapping[str, Any] | None = None,
) -> np.ndarray | None:
    """Return an unweighted penalty matrix for one session edge.

    ``None`` is returned when the edge has no available third-session context.
    """

    resolved = higher_order_consistency_config_from_mapping(config)
    normalised_edge = _normalise_edge(edge)
    costs = {
        _normalise_edge(edge_key): np.asarray(matrix, dtype=float)
        for edge_key, matrix in pairwise_costs.items()
    }
    _validate_pairwise_shapes(costs, session_sizes=session_sizes)
    if normalised_edge not in costs:
        raise KeyError(f"No pairwise cost matrix for edge {normalised_edge!r}")

    contexts = _triplet_support_contexts(costs, normalised_edge)
    if not contexts:
        return None

    target_shape = costs[normalised_edge].shape
    best_support = np.full(target_shape, np.inf, dtype=float)
    for left_costs, right_costs in contexts:
        context_support = _sparse_min_shared_support_cost(
            left_costs,
            right_costs,
            support_top_k=resolved.support_top_k,
            support_cost_cap=resolved.support_cost_cap,
            large_cost=resolved.large_cost,
        )
        if context_support.shape != target_shape:
            raise ValueError(
                f"Triplet context for edge {normalised_edge!r} produced shape "
                f"{context_support.shape}, expected {target_shape}"
            )
        best_support = np.minimum(best_support, context_support)

    penalty = best_support - resolved.support_cost_cap
    penalty = np.where(np.isfinite(penalty), penalty, resolved.max_penalty)
    return np.clip(penalty, 0.0, resolved.max_penalty)


def higher_order_consistency_config_from_mapping(
    config: HigherOrderConsistencyConfig | Mapping[str, Any] | None,
) -> HigherOrderConsistencyConfig:
    """Normalize an optional higher-order-consistency configuration."""

    if config is None:
        return HigherOrderConsistencyConfig()
    if isinstance(config, HigherOrderConsistencyConfig):
        return config
    return HigherOrderConsistencyConfig(**dict(config))


def _triplet_support_contexts(
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    edge: SessionEdge,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return third-session contexts sharing one intermediate observation axis."""

    source, target = edge
    contexts: list[tuple[np.ndarray, np.ndarray]] = []

    # Backward support: previous -> source and previous -> target share the same
    # previous-session observation.
    for previous in range(source):
        previous_to_source = pairwise_costs.get((previous, source))
        previous_to_target = pairwise_costs.get((previous, target))
        if previous_to_source is not None and previous_to_target is not None:
            contexts.append((previous_to_source.T, previous_to_target.T))

    # Bridge support for skip edges: source -> middle -> target.
    for middle in range(source + 1, target):
        source_to_middle = pairwise_costs.get((source, middle))
        middle_to_target = pairwise_costs.get((middle, target))
        if source_to_middle is not None and middle_to_target is not None:
            contexts.append((source_to_middle, middle_to_target.T))

    # Forward support: source -> future and target -> future share the same
    # future-session observation.
    max_session_index = max((max(edge_key) for edge_key in pairwise_costs), default=target)
    for future in range(target + 1, max_session_index + 1):
        source_to_future = pairwise_costs.get((source, future))
        target_to_future = pairwise_costs.get((target, future))
        if source_to_future is not None and target_to_future is not None:
            contexts.append((source_to_future, target_to_future))

    return tuple(contexts)


def _sparse_min_shared_support_cost(
    left_costs: np.ndarray,
    right_costs: np.ndarray,
    *,
    support_top_k: int,
    support_cost_cap: float,
    large_cost: float,
) -> np.ndarray:
    """Approximate ``min_k left[i,k] + right[j,k]`` with per-k top lists."""

    left = np.asarray(left_costs, dtype=float)
    right = np.asarray(right_costs, dtype=float)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Triplet support inputs must be two-dimensional matrices")
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            "Triplet support matrices must have the same number of shared columns"
        )

    support = np.full((left.shape[0], right.shape[0]), np.inf, dtype=float)
    for shared_index in range(left.shape[1]):
        left_indices = _top_k_admissible_indices(
            left[:, shared_index],
            top_k=support_top_k,
            support_cost_cap=support_cost_cap,
            large_cost=large_cost,
        )
        right_indices = _top_k_admissible_indices(
            right[:, shared_index],
            top_k=support_top_k,
            support_cost_cap=support_cost_cap,
            large_cost=large_cost,
        )
        if left_indices.size == 0 or right_indices.size == 0:
            continue
        candidate_support = (
            left[left_indices, shared_index][:, None]
            + right[right_indices, shared_index][None, :]
        )
        current = support[np.ix_(left_indices, right_indices)]
        support[np.ix_(left_indices, right_indices)] = np.minimum(
            current,
            candidate_support,
        )
    return support


def _top_k_admissible_indices(
    values: np.ndarray,
    *,
    top_k: int,
    support_cost_cap: float,
    large_cost: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    admissible = _admissible_cost_mask(values, large_cost=large_cost)
    admissible &= values <= support_cost_cap
    indices = np.flatnonzero(admissible)
    if indices.size <= int(top_k):
        return indices
    partition = np.argpartition(values[indices], int(top_k) - 1)[: int(top_k)]
    return indices[partition]


def _admissible_cost_mask(values: np.ndarray, *, large_cost: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.isfinite(values) & (values < large_cost)


def _validate_pairwise_shapes(
    pairwise_costs: Mapping[SessionEdge, np.ndarray],
    *,
    session_sizes: Sequence[int],
) -> None:
    sizes = tuple(_nonnegative_integer(size, name="session_sizes") for size in session_sizes)
    for edge, matrix in pairwise_costs.items():
        source, target = _normalise_edge(edge)
        if source >= len(sizes) or target >= len(sizes):
            raise ValueError(f"Edge {edge!r} is out of bounds for {len(sizes)} sessions")
        expected_shape = (sizes[source], sizes[target])
        if np.asarray(matrix).shape != expected_shape:
            raise ValueError(
                f"Pairwise cost matrix for edge {edge!r} has shape "
                f"{np.asarray(matrix).shape}, expected {expected_shape}"
            )


def _normalise_edge(edge: SessionEdge) -> SessionEdge:
    if len(edge) != 2:
        raise ValueError("Session edges must contain exactly two indices")
    source = _integer(edge[0], name="source_session")
    target = _integer(edge[1], name="target_session")
    if source < 0 or target <= source:
        raise ValueError("Session edges must be forward edges with source < target")
    return source, target


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    try:
        as_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if as_array.shape != () or as_array.dtype == np.bool_:
        raise ValueError(f"{name} must be an integer")
    scalar = as_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if isinstance(scalar, Integral):
        return int(scalar)
    if isinstance(scalar, Real):
        value_float = float(scalar)
        if np.isfinite(value_float) and value_float.is_integer():
            return int(value_float)
    raise ValueError(f"{name} must be an integer")


def _nonnegative_integer(value: Any, *, name: str) -> int:
    parsed = _integer(value, name=name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _positive_integer(value: Any, *, name: str) -> int:
    parsed = _integer(value, name=name)
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        as_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if as_array.shape != () or as_array.dtype == np.bool_:
        raise ValueError(f"{name} must be finite")
    scalar = as_array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    if not isinstance(scalar, Real):
        raise ValueError(f"{name} must be finite")
    parsed = float(scalar)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _finite_positive_float(value: Any, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


__all__ = (
    "HigherOrderConsistencyConfig",
    "SessionEdge",
    "apply_higher_order_consistency",
    "higher_order_consistency_config_from_mapping",
    "triplet_consistency_penalty",
)
