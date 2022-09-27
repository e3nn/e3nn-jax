from functools import lru_cache
from math import factorial
from typing import Callable

import jax
from jax import numpy as jnp


def u(p: int, x: jnp.ndarray) -> jnp.ndarray:
    r"""Equivalent to :func:`poly_envelope` with ``n0 = p-1`` and ``n1 = 2``."""
    return 1 - (p + 1) * (p + 2) / 2 * x**p + p * (p + 2) * x ** (p + 1) - p * (p + 1) / 2 * x ** (p + 2)


def _constraint(x: float, derivative: int, degree: int):
    return [0 if derivative > N else factorial(N) // factorial(N - derivative) * x ** (N - derivative) for N in range(degree)]


@lru_cache(maxsize=None)
def solve_polynomial(constraints) -> jnp.ndarray:
    with jax.ensure_compile_time_eval():
        jax_enable_x64 = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)

        degree = len(constraints)
        A = jnp.array(
            [_constraint(x, derivative, degree) for x, derivative, _ in sorted(constraints)],
            dtype=jnp.float64,
        )
        B = jnp.array([y for _, _, y in sorted(constraints)], dtype=jnp.float64)
        c = jnp.linalg.solve(A, B)

        jax.config.update("jax_enable_x64", jax_enable_x64)
    return jax.jit(lambda x: jnp.polyval(c[::-1], x))


def poly_envelope(n0: int, n1: int) -> Callable[[float], float]:
    r"""Polynomial envelope function with ``n0`` and ``n1`` derivatives euqal to 0 at ``x=0`` and ``x=1`` respectively.

    Small documentation available at ``https://mariogeiger.ch/polynomial_envelope_for_gnn.pdf``.
    This is a generalization of :math:`u_p(x)`.

    .. jupyter-execute::
        :hide-code:

        import jax.numpy as jnp
        import e3nn_jax as e3nn
        import matplotlib.pyplot as plt

    .. jupyter-execute::

        x = jnp.linspace(0.0, 1.0, 100)
        plt.plot(x, e3nn.poly_envelope(10, 5)(x), label="10, 5")
        plt.plot(x, e3nn.poly_envelope(4, 4)(x), label="4, 4")
        plt.plot(x, e3nn.poly_envelope(1, 2)(x), label="1, 2")
        plt.legend()

    Args:
        n0 (int): number of derivatives equal to 0 at ``x=0``
        n1 (int): number of derivatives equal to 0 at ``x=1``

    Returns:
        callable: polynomial envelope function
    """
    fn = solve_polynomial(
        frozenset(
            {(-0.5, 0, 1.0), (0.5, 0, 0.0)}
            | {(-0.5, derivative, 0.0) for derivative in range(1, n0 + 1)}
            | {(0.5, derivative, 0.0) for derivative in range(1, n1 + 1)}
        )
    )
    return lambda x: fn(x - 0.5)
