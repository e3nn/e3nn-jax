r"""Spherical Harmonics as polynomials of x, y, z
"""
import math
from functools import partial
from typing import Dict, Union

import jax
import jax.numpy as jnp
import sympy

from e3nn_jax import Irreps, IrrepsData, clebsch_gordan
from e3nn_jax.util.sympy import sqrtQarray_to_sympy


def spherical_harmonics(
    irreps_out: Union[Irreps, int], input: Union[IrrepsData, jnp.ndarray], normalize: bool, normalization: str = "integral"
) -> IrrepsData:
    r"""Spherical harmonics

    .. image:: https://user-images.githubusercontent.com/333780/79220728-dbe82c00-7e54-11ea-82c7-b3acbd9b2246.gif

    | Polynomials defined on the 3d space :math:`Y^l: \mathbb{R}^3 \longrightarrow \mathbb{R}^{2l+1}`
    | Usually restricted on the sphere (with ``normalize=True``) :math:`Y^l: S^2 \longrightarrow \mathbb{R}^{2l+1}`
    | who satisfies the following properties:

    * are polynomials of the cartesian coordinates ``x, y, z``
    * is equivariant :math:`Y^l(R x) = D^l(R) Y^l(x)`
    * are orthogonal :math:`\int_{S^2} Y^l_m(x) Y^j_n(x) dx = \text{cste} \; \delta_{lj} \delta_{mn}`

    The value of the constant depends on the choice of normalization.

    It obeys the following property:

    .. math::

        Y^{l+1}_i(x) &= \text{cste}(l) \; & C_{ijk} Y^l_j(x) x_k

        \partial_k Y^{l+1}_i(x) &= \text{cste}(l) \; (l+1) & C_{ijk} Y^l_j(x)

    Where :math:`C` are the `clebsch_gordan`.

    .. note::

        This function match with this table of standard real spherical harmonics from Wikipedia_
        when ``normalize=True``, ``normalization='integral'`` and is called with the argument in the order ``y,z,x``
        (instead of ``x,y,z``).

    .. _Wikipedia: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics

    Args:
        irreps_out (`Irreps` or int): output irreps
        input (`IrrepsData` or `jnp.ndarray`): cartesian coordinates
        normalize (bool): if True, the polynomials are restricted to the sphere
        normalization (str): normalization of the constant :math:`\text{cste}`. Default is 'integral'

    Returns:
        `jnp.ndarray`: polynomials of the spherical harmonics
    """
    assert normalization in ["integral", "component", "norm"]

    if isinstance(irreps_out, int):
        l = irreps_out
        assert isinstance(input, IrrepsData)
        [(mul, ir)] = input.irreps
        irreps_out = Irreps([(1, (l, ir.p ** l))])

    irreps_out = Irreps(irreps_out)

    assert all([l % 2 == 1 or p == 1 for _, (l, p) in irreps_out])
    assert len(set([p for _, (l, p) in irreps_out if l % 2 == 1])) <= 1

    if isinstance(input, IrrepsData):
        [(mul, ir)] = input.irreps
        assert mul == 1
        assert ir.l == 1
        assert all([ir.p == p for _, (l, p) in irreps_out if l % 2 == 1])
        x = input.contiguous
    else:
        x = input

    return _jited_spherical_harmonics(irreps_out, x, normalize, normalization)


@partial(jax.jit, static_argnums=(0, 2, 3), inline=True)
def _jited_spherical_harmonics(irreps_out, x, normalize, normalization):
    assert x.shape[-1] == 3

    if normalize:
        r = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
        x = x / jnp.where(r == 0.0, 1.0, r)

    context = dict()
    for _, ir in irreps_out:
        _spherical_harmonics(ir.l, context, x, normalization)

    sh = [jnp.repeat(context[ir.l][..., None, :], mul, -2) for mul, ir in irreps_out]
    return IrrepsData.from_list(irreps_out, sh, x.shape[:-1])


def biggest_power_of_two(n):
    return 2 ** (n.bit_length() - 1)


def _spherical_harmonics(l: int, jax_context: Dict, input: jnp.ndarray, normalization: str) -> sympy.Array:
    jax_context.update(dict(jnp=jnp, clebsch_gordan=clebsch_gordan))

    if 0 not in jax_context:
        if normalization == "integral":
            jax_context[0] = math.sqrt(1 / (4 * math.pi)) * jnp.ones_like(input[..., :1])
            jax_context[1] = math.sqrt(3 / (4 * math.pi)) * input
        elif normalization == "component":
            jax_context[0] = jnp.ones_like(input[..., :1])
            jax_context[1] = math.sqrt(3) * input
        else:
            jax_context[0] = jnp.ones_like(input[..., :1])
            jax_context[1] = input

    if l == 0:
        return sympy.Array([1])

    if l == 1:
        return sympy.Array([1, 0, 0])

    def sh_var(l):
        return [sympy.symbols(f"sh{l}_{m}") for m in range(2 * l + 1)]

    l2 = biggest_power_of_two(l - 1)
    l1 = l - l2

    w = sqrtQarray_to_sympy(clebsch_gordan(l1, l2, l))
    yx = sympy.Array(
        [
            sum(sh_var(l1)[i] * sh_var(l2)[j] * w[i, j, k] for i in range(2 * l1 + 1) for j in range(2 * l2 + 1))
            for k in range(2 * l + 1)
        ]
    )

    sph_1_l1 = _spherical_harmonics(l1, jax_context, input, normalization)
    sph_1_l2 = _spherical_harmonics(l2, jax_context, input, normalization)

    y1 = yx.subs(zip(sh_var(l1), sph_1_l1)).subs(zip(sh_var(l2), sph_1_l2))
    norm = sympy.sqrt(sum(y1.applyfunc(lambda x: x ** 2)))
    y1 = y1 / norm

    if l not in jax_context:
        if normalization == "integral":
            x = math.sqrt((2 * l + 1) / (4 * math.pi)) / (
                math.sqrt((2 * l1 + 1) / (4 * math.pi)) * math.sqrt((2 * l2 + 1) / (4 * math.pi))
            )
        elif normalization == "component":
            x = math.sqrt((2 * l + 1) / ((2 * l1 + 1) * (2 * l2 + 1)))
        else:
            x = 1

        w = (x / float(norm)) * clebsch_gordan(l1, l2, l)
        jax_context[l] = jnp.einsum("...i,...j,ijk->...k", jax_context[l1], jax_context[l2], w)

    return y1
