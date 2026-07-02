import torch as _torch


def conj(x, array):
    return _torch.conj(array(x))
