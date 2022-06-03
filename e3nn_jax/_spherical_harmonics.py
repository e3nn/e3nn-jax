r"""Spherical Harmonics as polynomials of x, y, z
"""
import itertools
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

    sh = _spherical_harmonics(x[..., 0], x[..., 1], x[..., 2])
    sh = [jnp.stack(next(sh), axis=-1) for _ in range(irreps_out.lmax + 1)]
    sh = [jnp.repeat(sh[ir.l][..., None, :], mul, -2) for mul, ir in irreps_out]

    if normalization == "integral":
        sh = [(math.sqrt(ir.dim) / math.sqrt(4 * math.pi)) * y for (_, ir), y in zip(irreps_out, sh)]
    elif normalization == "component":
        sh = [math.sqrt(ir.dim) * y for (_, ir), y in zip(irreps_out, sh)]

    return IrrepsData.from_list(irreps_out, sh, x.shape[:-1])


def biggest_power_of_two(n):
    return 2 ** (n.bit_length() - 1)


def _spherical_harmonics(x, y, z):
    import sympy

    from e3nn_jax.util.sympy import sqrtQarray_to_sympy

    jax_context = dict(x=x, y=y, z=z, sqrt=jnp.sqrt)
    jax_context["sh0_0"] = jnp.ones_like(x)
    jax_context["sh1_0"] = x
    jax_context["sh1_1"] = y
    jax_context["sh1_2"] = z
    yield [jax_context["sh0_0"]]
    yield [x, y, z]

    sph_x = {
        0: sympy.Array([1]),
        1: sympy.Array(sympy.symbols("x, y, z")),
    }
    sph_1 = {
        0: sympy.Array([1]),
        1: sympy.Array([1, 0, 0]),
    }

    def sh_var(l):
        return [sympy.symbols(f"sh{l}_{m}") for m in range(2 * l + 1)]

    for l in itertools.count(2):
        l2 = biggest_power_of_two(l - 1)
        l1 = l - l2

        w = sqrtQarray_to_sympy(clebsch_gordan(l1, l2, l))
        yx = sympy.Array(
            [
                sum(sh_var(l1)[i] * sh_var(l2)[j] * w[i, j, k] for i in range(2 * l1 + 1) for j in range(2 * l2 + 1))
                for k in range(2 * l + 1)
            ]
        )

        y1 = yx.subs(zip(sh_var(l1), sph_1[l1])).subs(zip(sh_var(l2), sph_1[l2]))
        norm = sympy.sqrt(sum(y1.applyfunc(lambda x: x ** 2)))
        y1 = y1 / norm
        yx = yx / norm
        yx = sympy.simplify(yx)

        sph_x[l] = yx
        sph_1[l] = y1

        values = [eval(f"{sympy.N(p)}", jax_context) for p in yx]
        yield values

        for k in range(2 * l + 1):
            jax_context[f"sh{l}_{k}"] = values[k]


def print_spherical_harmonics(lmax):  # pragma: no cover
    import sympy

    from e3nn_jax.util.sympy import sqrtQarray_to_sympy

    xyz = sympy.symbols("x, y, z")

    print("sh0_0 = 1")
    print("yield [sh0_0]\n")

    sph_x = {
        0: sympy.Array([1]),
    }
    sph_1 = {
        0: sympy.Array([1]),
    }

    for l in range(lmax):
        d = 2 * l + 1
        sh_var = [sympy.symbols(f"sh{l}_{m}") for m in range(d)]
        w = sqrtQarray_to_sympy(clebsch_gordan(1, l, l + 1))
        yx = sympy.Array([sum(xyz[i] * sh_var[n] * w[i, n, m] for i in range(3) for n in range(d)) for m in range(d + 2)])

        if l <= 1:
            yx = yx.subs(zip(sh_var, sph_x[l]))

        y1 = yx.subs(zip(xyz, (1, 0, 0))).subs(zip(sh_var, sph_1[l]))
        norm = sympy.sqrt(sum(y1.applyfunc(lambda x: x ** 2)))
        y1 = y1 / norm
        yx = yx / norm
        yx = sympy.simplify(yx)

        sph_x[l + 1] = yx
        sph_1[l + 1] = y1

        # print code
        for m, p in enumerate(yx):
            print(f"sh{l+1}_{m} = {p}")

        print(f"yield [{', '.join([f'sh{l+1}_{m}' for m in range(d + 2)])}]\n")
