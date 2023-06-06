from typing import Tuple

import numpy as np


def normalize_integer_ratio(n, d):
    g = np.gcd(n, d)
    g = np.where(d < 0, -g, g)
    return n // g, d // g


def _as_approx_integer_ratio(x):
    # only for 0 <= x <= 1
    big = 1 << 52 - 1  # mantissa is 52 bits

    n = np.floor(x * big).astype(np.int64)
    with np.errstate(invalid="ignore"):
        d = np.round(n / x).astype(np.int64)
    d = np.where(n == 0, np.ones(d.shape, dtype=np.int64), d)
    return n, d


def as_approx_integer_ratio(x):
    assert x.dtype == np.float64
    sign = np.sign(x).astype(np.int64)
    x = np.abs(x)

    with np.errstate(divide="ignore", over="ignore"):
        px = np.where(x < 1.0, 1.0, x)
        n, d = np.where(
            x <= 1,
            _as_approx_integer_ratio(x),
            _as_approx_integer_ratio(1 / px)[::-1],
        )
    return normalize_integer_ratio(sign * n, d)


def limit_denominator(n, d, max_denominator=1_000_000):
    # (n, d) = must be normalized
    n0, d0 = n, d
    p0, q0, p1, q1 = (
        np.zeros_like(n),
        np.ones_like(n),
        np.ones_like(n),
        np.zeros_like(n),
    )
    while True:
        a = n // d
        q2 = q0 + a * q1
        stop = (q2 > max_denominator) | (d0 <= max_denominator)
        if np.all(stop):
            break
        p0, q0, p1, q1 = np.where(stop, (p0, q0, p1, q1), (p1, q1, p0 + a * p1, q2))
        n, d = np.where(stop, (n, d), (d, n - a * d))

    with np.errstate(divide="ignore"):
        k = (max_denominator - q0) // q1
    n1, d1 = p0 + k * p1, q0 + k * q1
    n2, d2 = p1, q1
    with np.errstate(over="ignore"):
        mask = np.abs(d1 * (n2 * d0 - n0 * d2)) <= np.abs(d2 * (n1 * d0 - n0 * d1))
    return np.where(
        d0 < max_denominator,
        (n0, d0),
        np.where(mask, (n2, d2), (n1, d1)),
    )


def _round_to_sqrt_rational(x, max_denominator):
    sign = np.sign(x)
    n, d = as_approx_integer_ratio(x**2)
    n, d = limit_denominator(n, d, max_denominator**2 + 1)
    return sign * np.sqrt(n / d)


def round_to_sqrt_rational(x: np.ndarray, max_denominator=4096) -> np.ndarray:
    """Round a number to the closest number of the form ``sqrt(p)/q`` for ``q <= max_denominator``"""
    x = np.array(x)
    if np.iscomplex(x).any():
        return _round_to_sqrt_rational(
            np.real(x), max_denominator
        ) + 1j * _round_to_sqrt_rational(np.imag(x), max_denominator)
    return _round_to_sqrt_rational(np.real(x), max_denominator)


def gram_schmidt(A: np.ndarray, *, epsilon=1e-5, round_fn=lambda x: x) -> np.ndarray:
    """
    Orthogonalize a matrix using the Gram-Schmidt process.
    """
    assert A.ndim == 2, "Gram-Schmidt process only works for matrices."
    assert A.dtype in [
        np.float64,
        np.complex128,
    ], "Gram-Schmidt process only works for float64 matrices."
    Q = []
    for i in range(A.shape[0]):
        v = A[i]
        for w in Q:
            v -= np.dot(np.conj(w), v) * w
        norm = np.linalg.norm(v)
        if norm > epsilon:
            v = round_fn(v / norm)
            Q += [v]
    return np.stack(Q) if len(Q) > 0 else np.empty((0, A.shape[1]))


def basis_intersection(
    basis1: np.ndarray, basis2: np.ndarray, *, epsilon=1e-5, round_fn=lambda x: x
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the intersection of two bases

    Args:
        basis1 (np.ndarray): A basis, shape ``(n1, d)``
        basis2 (np.ndarray): Another basis, shape ``(n2, d)``
        epsilon (float, optional): Tolerance for the norm of the vectors. Defaults to 1e-4.
        round_fn (function, optional): Function to round the vectors. Defaults to lambda x: x.

    Returns:
        (tuple): tuple containing:

            np.ndarray: A projection matrix that projects vectors of the first basis in the intersection of the two bases.
                Shape ``(dim_intersection, n1)``
            np.ndarray: A projection matrix that projects vectors of the second basis in the intersection of the two bases.
                Shape ``(dim_intersection, n2)``

    Examples:
        >>> basis1 = np.array([[1, 0, 0], [0, 0, 1.0]])
        >>> basis2 = np.array([[1, 1, 0], [0, 1, 0.0]])
        >>> P1, P2 = basis_intersection(basis1, basis2)
        >>> P1 @ basis1
        array([[1., 0., 0.]])
    """
    assert basis1.ndim == 2
    assert basis2.ndim == 2
    assert basis1.shape[1] == basis2.shape[1]

    p = np.concatenate(
        [
            np.concatenate([basis1 @ basis1.T, -basis1 @ basis2.T], axis=1),
            np.concatenate([-basis2 @ basis1.T, basis2 @ basis2.T], axis=1),
        ],
        axis=0,
    )
    p = round_fn(p)

    w, v = np.linalg.eigh(p)
    v = v[:, w < epsilon]

    x1 = v[: basis1.shape[0], :]
    x1 = gram_schmidt(x1 @ x1.T, epsilon=epsilon, round_fn=round_fn)

    x2 = v[basis1.shape[0] :, :]
    x2 = gram_schmidt(x2 @ x2.T, epsilon=epsilon, round_fn=round_fn)
    return x1, x2
