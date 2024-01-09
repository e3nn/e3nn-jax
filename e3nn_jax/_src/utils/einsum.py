from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def einsum(eq, *xs):
    return jnp.einsum(eq, *xs, optimize="optimal")


@einsum.defjvp
def einsum_jvp(eq: str, xs: Tuple[jax.Array], x_dots: Tuple[jax.Array]) -> jax.Array:
    return einsum(eq, *xs), sum(
        einsum(eq, *(xs[:i] + (x_dot,) + xs[i + 1 :])) for i, x_dot in enumerate(x_dots)
    )
