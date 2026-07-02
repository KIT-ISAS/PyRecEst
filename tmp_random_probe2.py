def f(rng, n):
    phi = 2.0 * rng.uniform(size=n)
    sz = rng.uniform(size=n) * 2.0 - 1.0
    return phi, sz
