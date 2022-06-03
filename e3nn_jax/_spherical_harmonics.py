r"""Spherical Harmonics as polynomials of x, y, z
"""
import math
from functools import partial
from typing import Union

import jax
import jax.numpy as jnp

from e3nn_jax import Irreps, IrrepsData, clebsch_gordan


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
    if normalize:
        r = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
        x = x / jnp.where(r == 0.0, 1.0, r)

    context = dict()
    for _, ir in irreps_out:
        _spherical_harmonics(ir.l, context, x[..., 0], x[..., 1], x[..., 2])

    sh = [
        jnp.repeat(jnp.stack([context[f"sh{ir.l}_{k}"] for k in range(ir.dim)], axis=-1)[..., None, :], mul, -2)
        for mul, ir in irreps_out
    ]

    if normalization == "integral":
        sh = [(math.sqrt(ir.dim) / math.sqrt(4 * math.pi)) * y for (_, ir), y in zip(irreps_out, sh)]
    elif normalization == "component":
        sh = [math.sqrt(ir.dim) * y for (_, ir), y in zip(irreps_out, sh)]

    return IrrepsData.from_list(irreps_out, sh, x.shape[:-1])


def biggest_power_of_two(n):
    return 2 ** (n.bit_length() - 1)


def _spherical_harmonics(l, jax_context, x, y, z):
    import sympy

    from e3nn_jax.util.sympy import sqrtQarray_to_sympy

    jax_context.update(dict(x=x, y=y, z=z, sqrt=jnp.sqrt))
    jax_context["sh0_0"] = jnp.ones_like(x)
    jax_context["sh1_0"] = x
    jax_context["sh1_1"] = y
    jax_context["sh1_2"] = z

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

    sph_1_l1 = _spherical_harmonics(l1, jax_context, x, y, z)
    sph_1_l2 = _spherical_harmonics(l2, jax_context, x, y, z)

    y1 = yx.subs(zip(sh_var(l1), sph_1_l1)).subs(zip(sh_var(l2), sph_1_l2))
    norm = sympy.sqrt(sum(y1.applyfunc(lambda x: x ** 2)))
    y1 = y1 / norm
    yx = yx / norm
    yx = sympy.simplify(yx)

    for k in range(2 * l + 1):
        if f"sh{l}_{k}" not in jax_context:
            jax_context[f"sh{l}_{k}"] = eval(f"{sympy.N(yx[k])}", jax_context)

    return y1
