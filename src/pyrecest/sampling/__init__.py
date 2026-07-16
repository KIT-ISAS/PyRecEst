import numpy as _np

from .abstract_sampler import AbstractSampler
from .euclidean_sampler import (
    AbstractEuclideanSampler,
    FibonacciGridSampler,
    FibonacciRejectionSampler,
    GaussianSampler,
    HaltonGridSampler,
    SobolGridSampler,
)
from .hyperspherical_sampler import (
    AbstractHopfBasedS3Sampler,
    AbstractHypersphericalUniformSampler,
    AbstractSphericalUniformSampler,
    FibonacciHopfSampler,
    HealpixHopfSampler,
    LeopardiSampler,
    SphericalFibonacciSampler,
    get_grid_hypersphere,
)
from .hypertoroidal_sampler import CircularUniformSampler
from .sigma_points import JulierSigmaPoints, MerweScaledSigmaPoints
from .support_points import (
    ellipsoid_axis_offsets,
    ellipsoid_axis_support_points,
    ellipsoid_sigma_points,
    mahalanobis_support_points,
    projected_linear_variance_from_axis_offsets,
    support_points_from_axis_offsets,
)


def _evaluate_real_rejection_pdf(pdf, samples, n_candidates):
    raw_density_values = pdf(samples)
    try:
        raw_density_array = _np.asarray(raw_density_values)
    except (TypeError, ValueError) as exc:
        raise ValueError("pdf must return real density values") from exc

    contains_object_complex = raw_density_array.dtype == object and any(
        isinstance(value, (complex, _np.complexfloating))
        for value in raw_density_array.reshape(-1)
    )
    if _np.iscomplexobj(raw_density_array) or contains_object_complex:
        raise ValueError("pdf must return real density values")

    try:
        density_values = _np.asarray(raw_density_values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("pdf must return real density values") from exc
    if density_values.ndim == 0:
        density_values = _np.full(n_candidates, density_values)
    else:
        density_values = density_values.reshape(-1)

    if density_values.shape[0] != n_candidates:
        raise ValueError("pdf must return one density value per candidate sample")
    if not _np.all(_np.isfinite(density_values)):
        raise ValueError("pdf must return finite density values")
    return density_values


_REJECTION_PDF_PATCH_ATTR = "_pyrecest_rejects_complex_density_values"
if not getattr(FibonacciRejectionSampler, _REJECTION_PDF_PATCH_ATTR, False):
    FibonacciRejectionSampler._evaluate_pdf = staticmethod(_evaluate_real_rejection_pdf)
    setattr(FibonacciRejectionSampler, _REJECTION_PDF_PATCH_ATTR, True)


__all__ = [
    "AbstractSampler",
    "AbstractEuclideanSampler",
    "GaussianSampler",
    "FibonacciGridSampler",
    "FibonacciRejectionSampler",
    "SobolGridSampler",
    "HaltonGridSampler",
    "get_grid_hypersphere",
    "CircularUniformSampler",
    "AbstractHypersphericalUniformSampler",
    "AbstractSphericalUniformSampler",
    "SphericalFibonacciSampler",
    "AbstractHopfBasedS3Sampler",
    "HealpixHopfSampler",
    "FibonacciHopfSampler",
    "LeopardiSampler",
    "JulierSigmaPoints",
    "MerweScaledSigmaPoints",
    "ellipsoid_axis_offsets",
    "ellipsoid_axis_support_points",
    "ellipsoid_sigma_points",
    "mahalanobis_support_points",
    "projected_linear_variance_from_axis_offsets",
    "support_points_from_axis_offsets",
]
