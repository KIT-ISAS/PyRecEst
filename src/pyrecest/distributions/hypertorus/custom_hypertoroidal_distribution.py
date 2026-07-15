# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import asarray, mod, pi, zeros

from ..abstract_custom_distribution import AbstractCustomDistribution
from ..circle.custom_circular_distribution import CustomCircularDistribution
from ._input_validation import as_shift_vector
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution


class CustomHypertoroidalDistribution(
    AbstractHypertoroidalDistribution, AbstractCustomDistribution
):
    def __init__(self, f, dim, shift_by=None, scale_by=1):
        # Constructor, it is the user's responsibility to ensure that f is a valid
        # hypertoroidal density and takes arguments of the same form as
        # .pdf, i.e., it needs to be vectorized.
        #
        # Parameters:
        #   f (function handle)
        #       pdf of the distribution
        #   dim (scalar)
        #       dimension of the hypertorus
        AbstractCustomDistribution.__init__(self, f, scale_by)
        AbstractHypertoroidalDistribution.__init__(self, dim)
        if shift_by is None:
            self.shift_by = zeros(dim)
        else:
            self.shift_by = as_shift_vector(shift_by, dim)

    def pdf(self, xs):
        xs = asarray(xs)
        return AbstractCustomDistribution.pdf(self, mod(xs + self.shift_by, 2 * pi))

    def to_custom_circular(self):
        # Convert to a custom circular distribution (only in 1D case)
        #
        # Returns:
        #   ccd (CustomCircularDistribution)
        #       CustomCircularDistribution with same parameters
        if self.dim != 1:
            raise ValueError(
                "Conversion to CustomCircularDistribution requires dim == 1"
            )
        ccd = CustomCircularDistribution(
            self.f, scale_by=self.scale_by, shift_by=self.shift_by[0]
        )
        return ccd

    def to_custom_toroidal(self):
        # Convert to a custom toroidal distribution (only in 2D case)
        #
        # Returns:
        #   ctd (CustomToroidalDistribution)
        #       CustomToroidalDistribution with same parameters
        from .custom_toroidal_distribution import CustomToroidalDistribution

        if self.dim != 2:
            raise ValueError(
                "Conversion to CustomToroidalDistribution requires dim == 2"
            )
        ctd = CustomToroidalDistribution(
            self.f, scale_by=self.scale_by, shift_by=self.shift_by
        )
        return ctd
