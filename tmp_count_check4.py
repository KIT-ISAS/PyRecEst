import numpy as np

def f(n):
    value = np.array(n, copy=False).item()
    if isinstance(value, str):
        raise ValueError('bad')
    return int(value)
