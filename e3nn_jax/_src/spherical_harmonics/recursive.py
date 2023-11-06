import math
from typing import Dict, Tuple

import jax.numpy as jnp
import sympy

import e3nn_jax as e3nn
from e3nn_jax._src.utils.sympy import sqrtQarray_to_sympy


def recursive_spherical_harmonics(
    l: int,
    context: Dict[int, jnp.ndarray],
    input: jnp.ndarray,
    normalization: str,
    algorithm: Tuple[str],
) -> sympy.Array:
    if l == 0:
        if 0 not in context:
            if normalization == "integral":
                context[0] = math.sqrt(1 / (4 * math.pi)) * jnp.ones_like(
                    input[..., :1]
                )
            elif normalization == "component":
                context[0] = jnp.ones_like(input[..., :1])
            else:
                context[0] = jnp.ones_like(input[..., :1])

        return sympy.Array([1])

    if l == 1:
        if 1 not in context:
            if normalization == "integral":
                context[1] = math.sqrt(3 / (4 * math.pi)) * input
            elif normalization == "component":
                context[1] = math.sqrt(3) * input
            else:
                context[1] = 1 * input

        return sympy.Array([1, 0, 0])

    def sh_var(l):
        return [sympy.symbols(f"sh{l}_{m}") for m in range(2 * l + 1)]

    l2 = biggest_power_of_two(l - 1)
    l1 = l - l2

    w = sqrtQarray_to_sympy(e3nn.clebsch_gordan(l1, l2, l))
    yx = sympy.Array(
        [
            sum(
                sh_var(l1)[i] * sh_var(l2)[j] * w[i, j, k]
                for i in range(2 * l1 + 1)
                for j in range(2 * l2 + 1)
            )
            for k in range(2 * l + 1)
        ]
    )

    sph_1_l1 = recursive_spherical_harmonics(
        l1, context, input, normalization, algorithm
    )
    sph_1_l2 = recursive_spherical_harmonics(
        l2, context, input, normalization, algorithm
    )

    y1 = yx.subs(zip(sh_var(l1), sph_1_l1)).subs(zip(sh_var(l2), sph_1_l2))
    norm = sympy.sqrt(sum(y1.applyfunc(lambda x: x**2)))
    y1 = y1 / norm

    if l not in context:
        if normalization == "integral":
            x = math.sqrt((2 * l + 1) / (4 * math.pi)) / (
                math.sqrt((2 * l1 + 1) / (4 * math.pi))
                * math.sqrt((2 * l2 + 1) / (4 * math.pi))
            )
        elif normalization == "component":
            x = math.sqrt((2 * l + 1) / ((2 * l1 + 1) * (2 * l2 + 1)))
        else:
            x = 1

        w = (x / float(norm)) * e3nn.clebsch_gordan(l1, l2, l)
        w = w.astype(input.dtype)

        if "dense" in algorithm:
            context[l] = jnp.einsum("...i,...j,ijk->...k", context[l1], context[l2], w)
        elif "sparse" in algorithm:
            context[l] = jnp.stack(
                [
                    sum(
                        [
                            w[i, j, k] * context[l1][..., i] * context[l2][..., j]
                            for i in range(2 * l1 + 1)
                            for j in range(2 * l2 + 1)
                            if w[i, j, k] != 0
                        ]
                    )
                    for k in range(2 * l + 1)
                ],
                axis=-1,
            )
        else:
            raise ValueError("Unknown algorithm: must be 'dense' or 'sparse'")

    return y1


def biggest_power_of_two(n):
    return 2 ** (n.bit_length() - 1)
