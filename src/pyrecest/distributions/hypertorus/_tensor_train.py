"""Private tensor-train helper for hypertoroidal Fourier coefficients."""

from __future__ import annotations

from math import prod, sqrt

import numpy as np


class TensorTrain:
    """Minimal complex tensor-train representation."""

    def __init__(self, cores):
        checked = tuple(np.asarray(core, dtype=np.complex128).copy() for core in cores)
        if not checked:
            raise ValueError("At least one TT core is required.")
        for core in checked:
            if core.ndim != 3:
                raise ValueError("TT cores must have shape (left_rank, mode_size, right_rank).")
        if checked[0].shape[0] != 1 or checked[-1].shape[2] != 1:
            raise ValueError("Boundary TT ranks must be one.")
        for left, right in zip(checked, checked[1:]):
            if left.shape[2] != right.shape[0]:
                raise ValueError("Adjacent TT ranks do not match.")
        self.cores = checked

    @property
    def ndim(self):
        return len(self.cores)

    @property
    def shape(self):
        return tuple(core.shape[1] for core in self.cores)

    @property
    def ranks(self):
        return (self.cores[0].shape[0],) + tuple(core.shape[2] for core in self.cores)

    @property
    def size(self):
        return prod(self.shape)

    @property
    def storage_size(self):
        return int(sum(core.size for core in self.cores))

    def copy(self):
        return TensorTrain(tuple(core.copy() for core in self.cores))
