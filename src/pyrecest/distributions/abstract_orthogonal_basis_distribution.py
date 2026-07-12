import copy
import warnings
from abc import abstractmethod

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import abs as backend_abs
from pyrecest.backend import all, exp, imag, real

from .abstract_distribution_type import AbstractDistributionType

_REAL_DENSITY_IMAG_TOL = 1e-10


class AbstractOrthogonalBasisDistribution(AbstractDistributionType):
    """
    Abstract base class for distributions based on orthogonal basis functions.
    """

    def __init__(self, coeff_mat, transformation):
        """
        Initialize the distribution.

        :param coeff_mat: Coefficient matrix.
        :param transformation: Transformation function. Possible values are "sqrt", "identity", "log".
        """
        self.transformation = transformation
        self.coeff_mat = coeff_mat
        self.normalize_in_place()

    @abstractmethod
    def normalize_in_place(self):
        """
        Abstract method to normalize the distribution. Implementation required in subclasses.
        """

    @abstractmethod
    def value(self, xs):
        """
        Abstract method to get value of the distribution for given input. Implementation required in subclasses.

        :param xs: Input data for value calculation.
        """

    def normalize(self):
        """
        Normalizes the distribution.

        :return: Normalized distribution.
        """
        result = copy.deepcopy(self)
        result.normalize_in_place()
        return result

    @staticmethod
    def _discard_negligible_imaginary_part(val):
        """
        Return the real part of a numerically real value.

        Non-square-root density representations should evaluate to real-valued
        densities. Small imaginary parts arise from floating-point roundoff in
        complex basis evaluations and can be discarded; larger imaginary parts
        indicate inconsistent coefficients and must not be silently ignored.
        """
        if not all(backend_abs(imag(val)) <= _REAL_DENSITY_IMAG_TOL):
            raise ValueError(
                "Density evaluation has a non-negligible imaginary part. "
                "Check that the coefficients define a real-valued density."
            )
        return real(val)

    def pdf(self, xs):
        """
        Calculates probability density function for the given input.

        :param xs: Input data for PDF calculation.
        :return: PDF value.
        """
        val = self.value(xs)
        if self.transformation == "sqrt":
            return backend_abs(val) ** 2

        if self.transformation == "identity":
            return self._discard_negligible_imaginary_part(val)

        if self.transformation == "log":
            warnings.warn("Density may not be normalized")
            return exp(self._discard_negligible_imaginary_part(val))

        raise ValueError("Transformation not recognized or unsupported")
