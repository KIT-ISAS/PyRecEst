"""Wigner-D spectral distributions on SO(3)."""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Callable
from operator import index as operator_index
from typing import Literal

import numpy as np
import pyrecest.backend
from pyrecest.backend import array, complex128, pi, real, sqrt, to_numpy, zeros

from ._so3_helpers import geodesic_distance, normalize_quaternions, quaternions_to_rotation_matrices
from .abstract_orthogonal_basis_distribution import AbstractOrthogonalBasisDistribution

Transformation = Literal["identity", "sqrt"]
ConvolutionSide = Literal["right", "left"]
_REAL_TOL = 5e-8
_NORM_TOL = 1e-12


def _require_numpy_backend() -> None:
    if pyrecest.backend.__backend_name__ != "numpy":  # pylint: disable=no-member
        raise NotImplementedError("SO3WignerDistribution currently requires the NumPy backend.")


def _as_nonnegative_int(name: str, value) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer.")
    try:
        value = operator_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer.") from exc
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _center(degree: int) -> int:
    return degree


def _empty_coeff_tensor(degree: int):
    return zeros((degree + 1, 2 * degree + 1, 2 * degree + 1), dtype=complex128)


def _index_triples(degree: int):
    for ell in range(degree + 1):
        for m in range(-ell, ell + 1):
            for n in range(-ell, ell + 1):
                yield ell, m, n


def _block(coeff_mat, ell: int):
    degree = coeff_mat.shape[0] - 1
    center = _center(degree)
    return coeff_mat[ell, center - ell : center + ell + 1, center - ell : center + ell + 1]


def _set_block(coeff_mat, ell: int, block) -> None:
    degree = coeff_mat.shape[0] - 1
    center = _center(degree)
    coeff_mat[ell, center - ell : center + ell + 1, center - ell : center + ell + 1] = block


def _as_numpy_vector(values, name: str) -> np.ndarray:
    values_np = np.asarray(to_numpy(values), dtype=float)
    if values_np.ndim == 0:
        values_np = values_np.reshape((1,))
    if not np.all(np.isfinite(values_np)):
        raise ValueError(f"{name} must be finite.")
    return values_np.reshape((-1,))


def _quaternion_multiply_np(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = left[:, 0], left[:, 1], left[:, 2], left[:, 3]
    x2, y2, z2, w2 = right[:, 0], right[:, 1], right[:, 2], right[:, 3]
    return np.column_stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    )


def euler_zyz_to_quaternions(alpha, beta, gamma):
    """Convert ZYZ Euler angles to canonical scalar-last unit quaternions."""
    alpha = _as_numpy_vector(alpha, "alpha")
    beta = _as_numpy_vector(beta, "beta")
    gamma = _as_numpy_vector(gamma, "gamma")
    alpha, beta, gamma = np.broadcast_arrays(alpha, beta, gamma)
    qa = np.column_stack((np.zeros(alpha.size), np.zeros(alpha.size), np.sin(alpha.ravel() / 2), np.cos(alpha.ravel() / 2)))
    qb = np.column_stack((np.zeros(beta.size), np.sin(beta.ravel() / 2), np.zeros(beta.size), np.cos(beta.ravel() / 2)))
    qg = np.column_stack((np.zeros(gamma.size), np.zeros(gamma.size), np.sin(gamma.ravel() / 2), np.cos(gamma.ravel() / 2)))
    return normalize_quaternions(_quaternion_multiply_np(_quaternion_multiply_np(qa, qb), qg))


def quaternions_to_euler_zyz(quaternions):
    """Convert scalar-last unit quaternions to ZYZ Euler angles."""
    quaternions = normalize_quaternions(quaternions)
    rotations = np.asarray(to_numpy(quaternions_to_rotation_matrices(quaternions)), dtype=float).reshape((-1, 3, 3))
    beta = np.arccos(np.clip(rotations[:, 2, 2], -1.0, 1.0))
    sin_beta = np.sin(beta)
    regular = np.abs(sin_beta) > 1e-10
    alpha = np.zeros(rotations.shape[0])
    gamma = np.zeros(rotations.shape[0])
    alpha[regular] = np.arctan2(rotations[regular, 1, 2], rotations[regular, 0, 2])
    gamma[regular] = np.arctan2(rotations[regular, 2, 1], -rotations[regular, 2, 0])
    alpha[~regular] = np.arctan2(rotations[~regular, 1, 0], rotations[~regular, 0, 0])
    return np.mod(alpha, 2 * math.pi), beta, np.mod(gamma, 2 * math.pi)


def wigner_small_d(ell: int, m: int, n: int, beta):
    """Evaluate the Wigner small-d function using the finite factorial sum."""
    ell = _as_nonnegative_int("ell", ell)
    if abs(m) > ell or abs(n) > ell:
        raise ValueError("m and n must satisfy |m| <= ell and |n| <= ell.")
    beta = _as_numpy_vector(beta, "beta")
    c = np.cos(0.5 * beta)
    s = np.sin(0.5 * beta)
    log_prefactor = 0.5 * (
        math.lgamma(ell + m + 1)
        + math.lgamma(ell - m + 1)
        + math.lgamma(ell + n + 1)
        + math.lgamma(ell - n + 1)
    )
    result = np.zeros(beta.shape, dtype=float)
    for k in range(max(0, m - n), min(ell + m, ell - n) + 1):
        denom = (
            math.lgamma(ell + m - k + 1)
            + math.lgamma(k + 1)
            + math.lgamma(n - m + k + 1)
            + math.lgamma(ell - n - k + 1)
        )
        sign = -1.0 if (k - m + n) % 2 else 1.0
        result += sign * math.exp(log_prefactor - denom) * c ** (2 * ell + m - n - 2 * k) * s ** (n - m + 2 * k)
    return result


def wigner_d(ell: int, m: int, n: int, alpha, beta, gamma):
    """Evaluate ``D^ell_{mn}(alpha, beta, gamma)``."""
    alpha = _as_numpy_vector(alpha, "alpha")
    beta = _as_numpy_vector(beta, "beta")
    gamma = _as_numpy_vector(gamma, "gamma")
    alpha, beta, gamma = np.broadcast_arrays(alpha, beta, gamma)
    return np.exp(-1j * m * alpha) * wigner_small_d(ell, m, n, beta.ravel()).reshape(beta.shape) * np.exp(-1j * n * gamma)


def normalized_wigner_d(ell: int, m: int, n: int, alpha, beta, gamma):
    """Evaluate ``sqrt(2*ell+1)/pi * D^ell_{mn}``."""
    return math.sqrt(2 * ell + 1) / math.pi * wigner_d(ell, m, n, alpha, beta, gamma)


def so3_quadrature_grid(quadrature_degree: int):
    """Return ``alpha, beta, gamma, quaternions, weights`` for SO(3) Haar volume."""
    quadrature_degree = _as_nonnegative_int("quadrature_degree", quadrature_degree)
    n_alpha = max(2 * quadrature_degree + 1, 1)
    n_gamma = n_alpha
    n_beta = max(quadrature_degree + 1, 1)
    alpha_1d = np.linspace(0, 2 * math.pi, n_alpha, endpoint=False)
    gamma_1d = np.linspace(0, 2 * math.pi, n_gamma, endpoint=False)
    x_beta, w_beta = np.polynomial.legendre.leggauss(n_beta)
    beta_1d = np.arccos(np.clip(x_beta, -1, 1))
    alpha, beta, gamma = np.meshgrid(alpha_1d, beta_1d, gamma_1d, indexing="ij")
    _, beta_weights, _ = np.meshgrid(alpha_1d, w_beta, gamma_1d, indexing="ij")
    weights = (2 * math.pi / n_alpha) * (2 * math.pi / n_gamma) * 0.125 * beta_weights.ravel()
    alpha = alpha.ravel(); beta = beta.ravel(); gamma = gamma.ravel()
    return alpha, beta, gamma, euler_zyz_to_quaternions(alpha, beta, gamma), weights


class SO3WignerDistribution(AbstractOrthogonalBasisDistribution):
    """Band-limited Wigner-D representation of an SO(3) density."""

    dim = 3

    def __init__(self, coeff_mat, transformation: Transformation = "identity", assert_real: bool = True):
        _require_numpy_backend()
        if transformation not in ("identity", "sqrt"):
            raise ValueError("transformation must be 'identity' or 'sqrt'.")
        coeff_mat_np = np.asarray(to_numpy(coeff_mat), dtype=np.complex128)
        if coeff_mat_np.ndim != 3:
            raise ValueError("coeff_mat must be three-dimensional.")
        self.degree = coeff_mat_np.shape[0] - 1
        if coeff_mat_np.shape != (self.degree + 1, 2 * self.degree + 1, 2 * self.degree + 1):
            raise ValueError("coeff_mat must have shape (L+1, 2L+1, 2L+1).")
        if not np.all(np.isfinite(coeff_mat_np)):
            raise ValueError("coeff_mat must be finite.")
        self.assert_real = assert_real
        super().__init__(array(self._zero_invalid_entries(coeff_mat_np), dtype=complex128), transformation)

    def _zero_invalid_entries(self, coeff_mat_np: np.ndarray) -> np.ndarray:
        result = coeff_mat_np.copy()
        mask = np.zeros_like(result, dtype=bool)
        center = _center(self.degree)
        for ell in range(self.degree + 1):
            mask[ell, center - ell : center + ell + 1, center - ell : center + ell + 1] = True
        result[~mask] = 0.0
        return result

    @staticmethod
    def uniform(degree: int, transformation: Transformation = "identity"):
        degree = _as_nonnegative_int("degree", degree)
        coeff_mat = _empty_coeff_tensor(degree)
        center = _center(degree)
        if transformation == "identity":
            coeff_mat[0, center, center] = 1.0 / pi
        elif transformation == "sqrt":
            coeff_mat[0, center, center] = 1.0
        else:
            raise ValueError("transformation must be 'identity' or 'sqrt'.")
        return SO3WignerDistribution(coeff_mat, transformation)

    @staticmethod
    def manifold_size():
        return pi**2

    @staticmethod
    def basis_value(ell: int, m: int, n: int, alpha, beta, gamma):
        return array(normalized_wigner_d(ell, m, n, alpha, beta, gamma), dtype=complex128)

    @staticmethod
    def quadrature_grid(quadrature_degree: int):
        alpha, beta, gamma, quats, weights = so3_quadrature_grid(quadrature_degree)
        return array(alpha), array(beta), array(gamma), array(quats, dtype=float), array(weights, dtype=float)

    @staticmethod
    def _default_quadrature_degree(degree: int) -> int:
        return max(2 * degree, 1)

    @staticmethod
    def _fit_values_on_grid(values, alpha, beta, gamma, weights, degree: int):
        coeff_mat = np.zeros((degree + 1, 2 * degree + 1, 2 * degree + 1), dtype=np.complex128)
        center = _center(degree)
        weighted_values = np.asarray(values) * weights
        for ell, m, n in _index_triples(degree):
            coeff_mat[ell, center + m, center + n] = np.sum(weighted_values * np.conjugate(normalized_wigner_d(ell, m, n, alpha, beta, gamma)))
        return array(coeff_mat, dtype=complex128)

    @staticmethod
    def from_function_via_quadrature(fun: Callable, degree: int, transformation: Transformation = "identity", quadrature_degree: int | None = None):
        degree = _as_nonnegative_int("degree", degree)
        if quadrature_degree is None:
            quadrature_degree = SO3WignerDistribution._default_quadrature_degree(degree)
        alpha, beta, gamma, quaternions, weights = so3_quadrature_grid(quadrature_degree)
        values = np.asarray(to_numpy(fun(array(quaternions, dtype=float))), dtype=float).reshape((-1,))
        if values.shape[0] != quaternions.shape[0]:
            raise ValueError("fun must return one value per quaternion.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Function values must be finite.")
        if transformation == "sqrt":
            values = np.sqrt(np.maximum(values, 0.0))
        elif transformation != "identity":
            raise ValueError("transformation must be 'identity' or 'sqrt'.")
        coeff_mat = SO3WignerDistribution._fit_values_on_grid(values, alpha, beta, gamma, weights, degree)
        return SO3WignerDistribution(coeff_mat, transformation)

    @staticmethod
    def from_distribution_via_quadrature(dist, degree: int, transformation: Transformation = "identity", quadrature_degree: int | None = None):
        if not hasattr(dist, "pdf"):
            raise TypeError("dist must provide a pdf(xs) method.")
        return SO3WignerDistribution.from_function_via_quadrature(dist.pdf, degree, transformation, quadrature_degree)

    def value(self, xs):
        alpha, beta, gamma = quaternions_to_euler_zyz(normalize_quaternions(xs))
        return self.value_euler(alpha, beta, gamma)

    def value_euler(self, alpha, beta, gamma):
        alpha = _as_numpy_vector(alpha, "alpha")
        beta = _as_numpy_vector(beta, "beta")
        gamma = _as_numpy_vector(gamma, "gamma")
        alpha, beta, gamma = np.broadcast_arrays(alpha, beta, gamma)
        coeff_np = np.asarray(to_numpy(self.coeff_mat), dtype=np.complex128)
        center = _center(self.degree)
        vals = np.zeros(alpha.shape, dtype=np.complex128)
        for ell, m, n in _index_triples(self.degree):
            coeff = coeff_np[ell, center + m, center + n]
            if coeff != 0:
                vals += coeff * normalized_wigner_d(ell, m, n, alpha, beta, gamma)
        vals = vals.reshape((-1,))
        if self.assert_real and self.transformation == "identity":
            if not np.all(np.abs(vals.imag) <= _REAL_TOL):
                raise ValueError("Coefficients apparently do not represent a real-valued function.")
            return array(vals.real)
        return array(vals, dtype=complex128)

    def normalize_in_place(self, warn_unnorm: bool = True):
        int_val = self.integrate()
        if abs(float(np.asarray(to_numpy(int_val)).real)) < _NORM_TOL:
            raise ValueError("SO3WignerDistribution normalization integral is too close to zero.")
        if abs(float(np.asarray(to_numpy(int_val)).real) - 1.0) > 1e-5 and warn_unnorm:
            warnings.warn("SO3WignerDistribution coefficients were not normalized; normalizing.", stacklevel=2)
        if self.transformation == "identity":
            self.coeff_mat = self.coeff_mat / int_val
        else:
            self.coeff_mat = self.coeff_mat / sqrt(int_val)
        return self

    def integrate(self, integration_boundaries=None):
        if integration_boundaries is not None:
            raise NotImplementedError("Only full-domain SO(3) integration is supported.")
        center = _center(self.degree)
        if self.transformation == "identity":
            int_val = pi * self.coeff_mat[0, center, center]
        elif self.transformation == "sqrt":
            coeff_np = np.asarray(to_numpy(self.coeff_mat), dtype=np.complex128)
            int_val = sum(abs(coeff_np[ell, center + m, center + n]) ** 2 for ell, m, n in _index_triples(self.degree))
        else:
            raise ValueError("Unsupported transformation.")
        int_np = np.asarray(to_numpy(int_val))
        if abs(int_np.imag) > 1e-8:
            raise ValueError("SO(3) Wigner integral has a non-negligible imaginary part.")
        return real(int_val)

    def truncate(self, degree: int):
        degree = _as_nonnegative_int("degree", degree)
        result = copy.deepcopy(self)
        coeff_new = _empty_coeff_tensor(degree)
        for ell in range(min(self.degree, degree) + 1):
            _set_block(coeff_new, ell, _block(self.coeff_mat, ell))
        result.coeff_mat = coeff_new
        result.degree = degree
        return result.normalize_in_place(warn_unnorm=False)

    def to_identity(self, degree: int | None = None, quadrature_degree: int | None = None):
        if degree is None:
            degree = self.degree if self.transformation == "identity" else 2 * self.degree
        if self.transformation == "identity" and degree == self.degree:
            return copy.deepcopy(self)
        return SO3WignerDistribution.from_function_via_quadrature(self.pdf, degree, "identity", quadrature_degree)

    def to_sqrt(self, degree: int | None = None, quadrature_degree: int | None = None):
        if degree is None:
            degree = self.degree
        if self.transformation == "sqrt" and degree == self.degree:
            return copy.deepcopy(self)
        return SO3WignerDistribution.from_function_via_quadrature(self.pdf, degree, "sqrt", quadrature_degree)

    def convolve(self, other, side: ConvolutionSide = "right"):
        if not isinstance(other, SO3WignerDistribution):
            raise TypeError("other must be an SO3WignerDistribution.")
        if side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'.")
        if self.transformation != other.transformation:
            raise ValueError("Both distributions must use the same transformation.")
        if self.transformation == "sqrt":
            return self.to_identity(2 * self.degree).convolve(other.to_identity(2 * other.degree), side=side).to_sqrt(self.degree)
        degree_out = max(self.degree, other.degree)
        coeff_out = _empty_coeff_tensor(degree_out)
        for ell in range(min(self.degree, other.degree) + 1):
            block_self = np.asarray(to_numpy(_block(self.coeff_mat, ell)), dtype=np.complex128)
            block_other = np.asarray(to_numpy(_block(other.coeff_mat, ell)), dtype=np.complex128)
            block_out = block_self @ block_other if side == "right" else block_other @ block_self
            _set_block(coeff_out, ell, array(math.pi / math.sqrt(2 * ell + 1) * block_out, dtype=complex128))
        return SO3WignerDistribution(coeff_out, "identity")

    def multiply(self, other, quadrature_degree: int | None = None):
        if not isinstance(other, SO3WignerDistribution):
            raise TypeError("other must be an SO3WignerDistribution.")
        if self.transformation != other.transformation:
            raise ValueError("Both distributions must use the same transformation.")
        if quadrature_degree is None:
            quadrature_degree = self._default_quadrature_degree(self.degree)
        alpha, beta, gamma, _, weights = so3_quadrature_grid(quadrature_degree)
        values = np.asarray(to_numpy(self.value_euler(alpha, beta, gamma))) * np.asarray(to_numpy(other.value_euler(alpha, beta, gamma)))
        return SO3WignerDistribution(self._fit_values_on_grid(values, alpha, beta, gamma, weights, self.degree), self.transformation)

    def multiply_by_function(self, fun: Callable, quadrature_degree: int | None = None):
        if quadrature_degree is None:
            quadrature_degree = self._default_quadrature_degree(self.degree)
        alpha, beta, gamma, quaternions, weights = so3_quadrature_grid(quadrature_degree)
        current = np.asarray(to_numpy(self.value_euler(alpha, beta, gamma)))
        likelihood = np.asarray(to_numpy(fun(array(quaternions, dtype=float))), dtype=float).reshape((-1,))
        if likelihood.shape[0] != quaternions.shape[0]:
            raise ValueError("Likelihood function must return one value per quaternion.")
        if not np.all(np.isfinite(likelihood)) or np.any(likelihood < 0):
            raise ValueError("Likelihood values must be finite and nonnegative.")
        values = current * likelihood if self.transformation == "identity" else current * np.sqrt(likelihood)
        return SO3WignerDistribution(self._fit_values_on_grid(values, alpha, beta, gamma, weights, self.degree), self.transformation)

    def evaluate_on_quadrature_grid(self, quadrature_degree: int | None = None):
        if quadrature_degree is None:
            quadrature_degree = self._default_quadrature_degree(self.degree)
        alpha, beta, gamma, quaternions, weights = so3_quadrature_grid(quadrature_degree)
        return array(quaternions, dtype=float), self.value_euler(alpha, beta, gamma), array(weights, dtype=float)

    def mean_axis(self, quadrature_degree: int | None = None):
        quaternions, _, weights = self.evaluate_on_quadrature_grid(quadrature_degree)
        quats_np = np.asarray(to_numpy(quaternions), dtype=float)
        weights_np = np.asarray(to_numpy(weights), dtype=float)
        pdf_vals = np.asarray(to_numpy(self.pdf(quats_np)), dtype=float)
        scatter = np.einsum("n,ni,nj->ij", weights_np * pdf_vals, quats_np, quats_np)
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (scatter + scatter.T))
        return normalize_quaternions(array(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=float))[0]

    def mean(self):
        return self.mean_axis()

    def mode(self, quadrature_degree: int | None = None):
        quaternions, _, _ = self.evaluate_on_quadrature_grid(quadrature_degree)
        pdf_vals = np.asarray(to_numpy(self.pdf(quaternions)), dtype=float)
        return quaternions[int(np.argmax(pdf_vals))]

    def is_valid(self, tolerance=1e-6):
        try:
            integral_ok = abs(float(np.asarray(to_numpy(self.integrate())).real) - 1.0) <= tolerance
        except ValueError:
            return False
        return bool(np.all(np.isfinite(to_numpy(self.coeff_mat))) and integral_ok)

    geodesic_distance = staticmethod(geodesic_distance)
    as_rotation_matrices = staticmethod(quaternions_to_rotation_matrices)
