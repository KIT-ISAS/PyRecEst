# Advanced Tracking Notes

Advanced tracking workflows should make state, measurement-set, and association
semantics explicit before exposing a new example or public API.

Each advanced tracker example should document:

1. state vector layout and units;
2. measurement-set shape and missed-detection representation;
3. clutter model and gating configuration;
4. association output type and interpretation;
5. backend assumptions, especially NumPy/SciPy-only assignment paths;
6. evaluation metrics such as cardinality error, localization error, and track continuity.

Use small deterministic measurement sets for CI smoke tests. Assert stable
properties such as number of tracks, selected associations, finite log-likelihoods,
and diagnostic container shapes.

## Nonparametric cardinality priors

`pyrecest.tracking.nonparametric_cardinality` provides small Dirichlet-process
and Pitman--Yor-process prior utilities for target-generated cluster counts. They
are intentionally not complete trackers. Instead, they expose reusable prior
terms that can be multiplied with measurement likelihoods, survival probabilities,
and clutter alternatives inside a tracker-specific association or birth model.

For current cluster sizes `n_1, ..., n_K`, `PitmanYorCardinalityPrior` returns
Chinese-restaurant-process probabilities

```text
P(existing k) = (n_k - d) / (theta + n)
P(new cluster) = (theta + d K) / (theta + n)
```

where `n = sum_k n_k`, `theta` is the strength parameter, and `d` is the discount
parameter. `DirichletProcessCardinalityPrior` is the `d = 0` special case. A
positive discount gives heavier prior mass to high cluster counts and is useful
for prior-predictive checks in scenarios with many short-lived, bursty, or rare
target-generated clusters.

Keep the interpretation narrow: the cluster-count PMF is a prior over occupied
exchangeable clusters after a given number of target-generated observations, not
a full random-finite-set posterior over live targets. A multitarget tracker still
needs explicit survival, missed-detection, clutter, and measurement-likelihood
models.
