import numpy as np
from functools import lru_cache


def float_to_frac(target, n_primes=8, l1_radius=10, atol=1e-5, rtol=1e-3):
    """Convert a float to a fraction.

    Args:
        target (float): The target value.
        n_primes (int): The number of primes to use. ``primes[:n_primes]`` will be used.
        l1_radius (int): The L1 radius. For instance the fraction 2*3/5 is of L1 norm 3.

    Returns:
        (str): The fraction if successful, otherwise ``str(target)``.
    """
    exp, ok = _closest_fraction(target, n_primes, l1_radius, atol, rtol)
    return _print_fraction(exp) if ok else str(target)


@lru_cache(None)
def _l1_sphere(dim, radius):
    """Generate points in the dim-dimensional l1-sphere of radius radius"""
    if dim == 0:
        return np.zeros((0, 0), dtype=np.int32)
    if radius == 0:
        return np.zeros((1, dim), dtype=np.int32)

    x = _l1_sphere(dim-1, radius)
    x = np.concatenate([np.zeros((x.shape[0], 1), dtype=np.int32), x], axis=1)

    y = _l1_sphere(dim, radius-1)
    yn = y[y[:, 0] <= 0]
    yn[:, 0] -= 1

    yp = y[0 <= y[:, 0]]
    yp[:, 0] += 1

    return np.concatenate([yn, x, yp], axis=0)


_primes = np.array([
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
    79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
    167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
    257, 263, 269, 271
])


@lru_cache(None)
def _all_fractions(n_primes, l1_radius):
    m = np.concatenate([_l1_sphere(n_primes, radius) for radius in range(round(l1_radius) + 1)], axis=0)

    q = np.prod((1.0 * _primes[:n_primes])**m, axis=1)
    i = np.argsort(q)
    q = q[i]
    m = m[i]

    return q, m


def _closest_fraction(target, n_primes, l1_radius, atol, rtol):
    q, m = _all_fractions(n_primes, l1_radius)
    i = np.searchsorted(q, target)

    dist = q[i] - q[i - 1]
    err1 = target - q[i - 1]
    err2 = q[i] - target
    i = np.where(err1 < err2, i - 1, i)

    error = np.abs(q[i] - target)
    best_guess = m[i]

    ok = (error < atol) & (error < rtol * dist)

    return best_guess, ok


def _print_fraction(exponents):
    x = list(zip(_primes, exponents))
    x = [(p, e) for p, e in x if e != 0]
    x1 = [(p, e) for p, e in x if e > 0]
    x2 = [(p, -e) for p, e in x if e < 0]
    s1 = "*".join([f"{p}**{e}" if e != 1 else f"{p}" for p, e in x1])
    s2 = "*".join([f"{p}**{e}" if e != 1 else f"{p}" for p, e in x2])
    if len(x1) == 0:
        if len(x2) == 0:
            return "1"
        elif len(x2) == 1:
            return f"1/{s2}"
        else:
            return f"1/({s2})"
    else:
        if len(x2) == 0:
            return f"{s1}"
        elif len(x2) == 1:
            return f"{s1}/{s2}"
        else:
            return f"{s1}/({s2})"
