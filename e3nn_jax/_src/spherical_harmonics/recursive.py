from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import sympy

import e3nn_jax as e3nn
from e3nn_jax._src.utils.sympy import sqrtQarray_to_sympy


def recursive_spherical_harmonics(
    l: int,
    context: Dict[int, jax.Array],
    input: jax.Array,
    normalization: str,
    algorithm: Tuple[str],
) -> sympy.Array:
    assert normalization in ["integral", "component", "norm", "none"]

    x_var = sympy.symbols("x:3")

    if l == 0:
        if normalization == "integral":
            x = sympy.sqrt(1 / (4 * sympy.pi))
        elif normalization == "component":
            x = 1
        else:
            x = 1

        if 0 not in context:
            context[0] = float(x) * jnp.ones_like(input[..., :1])

        return x * sympy.Array([1])

    if l == 1:
        if normalization == "integral":
            x = sympy.sqrt(3 / (4 * sympy.pi))
        elif normalization == "component":
            x = sympy.sqrt(3)
        else:
            x = 1

        if 1 not in context:
            context[1] = float(x) * input

        return x * sympy.Array([x_var[0], x_var[1], x_var[2]])

    def sh_var(l):
        return [sympy.symbols(f"sh{l}_{m}") for m in range(2 * l + 1)]

    l2 = biggest_power_of_two(l - 1)
    l1 = l - l2

    C_var = sqrtQarray_to_sympy(e3nn.clebsch_gordan(l1, l2, l))
    yl_var = sympy.Array(
        [
            sum(
                sh_var(l1)[i] * sh_var(l2)[j] * C_var[i, j, k]
                for i in range(2 * l1 + 1)
                for j in range(2 * l2 + 1)
            )
            for k in range(2 * l + 1)
        ]
    )

    yl1_var = recursive_spherical_harmonics(
        l1, context, input, normalization, algorithm
    )
    yl2_var = recursive_spherical_harmonics(
        l2, context, input, normalization, algorithm
    )

    y_var = yl_var.subs(zip(sh_var(l1), yl1_var)).subs(zip(sh_var(l2), yl2_var))
    cst_var = yl_var.subs(
        {sh_var(l1)[i]: 1 if i == l1 else 0 for i in range(2 * l1 + 1)}
    ).subs({sh_var(l2)[i]: 1 if i == l2 else 0 for i in range(2 * l2 + 1)})
    norm = sympy.sqrt(sum(cst_var.applyfunc(lambda x: x**2)))

    y_var = y_var / norm
    cst_var = cst_var / norm

    if normalization == "integral":
        x = sympy.sqrt((2 * l + 1) / (4 * sympy.pi)) / (
            sympy.sqrt((2 * l1 + 1) / (4 * sympy.pi))
            * sympy.sqrt((2 * l2 + 1) / (4 * sympy.pi))
        )
    elif normalization == "component":
        x = sympy.sqrt((2 * l + 1) / (sympy.Integer((2 * l1 + 1) * (2 * l2 + 1))))
    else:
        x = 1

    if l not in context:
        C = float(x / norm) * e3nn.clebsch_gordan(l1, l2, l)
        C = C.astype(input.dtype)

        if "dense" in algorithm:
            context[l] = jnp.einsum("...i,...j,ijk->...k", context[l1], context[l2], C)
        elif "sparse" in algorithm:
            context[l] = jnp.stack(
                [
                    sum(
                        [
                            C[i, j, k] * context[l1][..., i] * context[l2][..., j]
                            for i in range(2 * l1 + 1)
                            for j in range(2 * l2 + 1)
                            if C[i, j, k] != 0
                        ]
                    )
                    for k in range(2 * l + 1)
                ],
                axis=-1,
            )
        else:
            raise ValueError("Unknown algorithm: must be 'dense' or 'sparse'")

    return x * y_var


def biggest_power_of_two(n):
    return 2 ** (n.bit_length() - 1)
