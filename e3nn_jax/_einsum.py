"""
einsum that optimize its derivatives contractions
"""
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def einsum(eq, *x):
    return jnp.einsum(eq, *x, optimize="optimal")


def _ein_fwd(eq, *x):
    return einsum(eq, *x), x


def _ein_bwd(eq, res, g):
    # TODO handle indices appearing in a single input like 'ij->i' or 'ii->'
    inputs, out = eq.split("->")
    inputs = inputs.split(",")
    return tuple(
        einsum(f"{','.join(inputs[:i] + inputs[i + 1:] + [out])}->{inputs[i]}", *res[:i], *res[i + 1 :], g)
        for i in range(len(res))
    )


einsum.defvjp(_ein_fwd, _ein_bwd)
