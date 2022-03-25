r"""Spherical Harmonics as polynomials of x, y, z
"""
import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.numpy import sqrt

from e3nn_jax import Irreps, IrrepsData, wigner_3j_sympy


@partial(jax.jit, static_argnums=(0, 2, 3), inline=True)
def spherical_harmonics(
    irreps_out,
    x,
    normalize: bool,
    normalization: str = 'integral'
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

    Where :math:`C` are the `wigner_3j`.

    .. note::

        This function match with this table of standard real spherical harmonics from Wikipedia_
        when ``normalize=True``, ``normalization='integral'`` and is called with the argument in the order ``y,z,x`` (instead of ``x,y,z``).

    .. _Wikipedia: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics

    Args:
        irreps_out (`Irreps`): output irreps
        x (`jnp.ndarray`): cartesian coordinates
        normalize (bool): if True, the polynomials are restricted to the sphere
        normalization (str): normalization of the constant :math:`\text{cste}`. Default is 'integral'

    Returns:
        `jnp.ndarray`: polynomials of the spherical harmonics
    """
    assert normalization in ['integral', 'component', 'norm']

    irreps_out = Irreps(irreps_out)

    assert all([l % 2 == 1 or p == 1 for _, (l, p) in irreps_out])
    assert len(set([p for _, (l, p) in irreps_out if l % 2 == 1])) <= 1
    if isinstance(x, IrrepsData):
        [(mul, ir)] = x.irreps
        assert mul == 1
        assert ir.l == 1
        assert all([ir.p == p for _, (l, p) in irreps_out if l % 2 == 1])
        x = x.contiguous

    _lmax = 8
    if irreps_out.lmax > _lmax:
        raise NotImplementedError(f'spherical_harmonics maximum l implemented is {_lmax}, send us an email to ask for more')

    if normalize:
        r = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
        x = x / jnp.where(r == 0.0, 1.0, r)

    sh = _spherical_harmonics(x[..., 0], x[..., 1], x[..., 2])
    sh = [jnp.stack(next(sh), axis=-1) for _ in range(irreps_out.lmax + 1)]
    sh = [jnp.repeat(sh[ir.l][..., None, :], mul, -2) for mul, ir in irreps_out]

    if normalization == 'integral':
        sh = [
            (math.sqrt(ir.dim) / math.sqrt(4 * math.pi)) * y
            for (_, ir), y in zip(irreps_out, sh)
        ]
    elif normalization == 'component':
        sh = [
            math.sqrt(ir.dim) * y
            for (_, ir), y in zip(irreps_out, sh)
        ]

    return IrrepsData.from_list(irreps_out, sh, x.shape[:-1])


def _spherical_harmonics(x, y, z):
    sh0_0 = jnp.ones_like(x)
    yield [sh0_0]

    sh1_0 = x
    sh1_1 = y
    sh1_2 = z
    yield [sh1_0, sh1_1, sh1_2]

    sh2_0 = sqrt(3)*x*z
    sh2_1 = sqrt(3)*x*y
    sh2_2 = -x**2/2 + y**2 - z**2/2
    sh2_3 = sqrt(3)*y*z
    sh2_4 = sqrt(3)*(-x**2 + z**2)/2
    yield [sh2_0, sh2_1, sh2_2, sh2_3, sh2_4]

    sh3_0 = sqrt(30)*(sh2_0*z + sh2_4*x)/6
    sh3_1 = sqrt(5)*(sh2_0*y + sh2_1*z + sh2_3*x)/3
    sh3_2 = -sqrt(2)*sh2_0*z/6 + 2*sqrt(2)*sh2_1*y/3 + sqrt(6)*sh2_2*x/3 + sqrt(2)*sh2_4*x/6
    sh3_3 = -sqrt(3)*sh2_1*x/3 + sh2_2*y - sqrt(3)*sh2_3*z/3
    sh3_4 = -sqrt(2)*sh2_0*x/6 + sqrt(6)*sh2_2*z/3 + 2*sqrt(2)*sh2_3*y/3 - sqrt(2)*sh2_4*z/6
    sh3_5 = sqrt(5)*(-sh2_1*x + sh2_3*z + sh2_4*y)/3
    sh3_6 = sqrt(30)*(-sh2_0*x + sh2_4*z)/6
    yield [sh3_0, sh3_1, sh3_2, sh3_3, sh3_4, sh3_5, sh3_6]

    sh4_0 = sqrt(14)*(sh3_0*z + sh3_6*x)/4
    sh4_1 = sqrt(7)*(2*sh3_0*y + sqrt(6)*sh3_1*z + sqrt(6)*sh3_5*x)/8
    sh4_2 = -sqrt(2)*sh3_0*z/8 + sqrt(3)*sh3_1*y/2 + sqrt(30)*sh3_2*z/8 + sqrt(30)*sh3_4*x/8 + sqrt(2)*sh3_6*x/8
    sh4_3 = -sqrt(6)*sh3_1*z/8 + sqrt(15)*sh3_2*y/4 + sqrt(10)*sh3_3*x/4 + sqrt(6)*sh3_5*x/8
    sh4_4 = -sqrt(6)*sh3_2*x/4 + sh3_3*y - sqrt(6)*sh3_4*z/4
    sh4_5 = -sqrt(6)*sh3_1*x/8 + sqrt(10)*sh3_3*z/4 + sqrt(15)*sh3_4*y/4 - sqrt(6)*sh3_5*z/8
    sh4_6 = -sqrt(2)*sh3_0*x/8 - sqrt(30)*sh3_2*x/8 + sqrt(30)*sh3_4*z/8 + sqrt(3)*sh3_5*y/2 - sqrt(2)*sh3_6*z/8
    sh4_7 = sqrt(7)*(-sqrt(6)*sh3_1*x + sqrt(6)*sh3_5*z + 2*sh3_6*y)/8
    sh4_8 = sqrt(14)*(-sh3_0*x + sh3_6*z)/4
    yield [sh4_0, sh4_1, sh4_2, sh4_3, sh4_4, sh4_5, sh4_6, sh4_7, sh4_8]

    sh5_0 = 3*sqrt(10)*(sh4_0*z + sh4_8*x)/10
    sh5_1 = 3*sh4_0*y/5 + 3*sqrt(2)*sh4_1*z/5 + 3*sqrt(2)*sh4_7*x/5
    sh5_2 = -sqrt(2)*sh4_0*z/10 + 4*sh4_1*y/5 + sqrt(14)*sh4_2*z/5 + sqrt(14)*sh4_6*x/5 + sqrt(2)*sh4_8*x/10
    sh5_3 = -sqrt(6)*sh4_1*z/10 + sqrt(21)*sh4_2*y/5 + sqrt(42)*sh4_3*z/10 + sqrt(42)*sh4_5*x/10 + sqrt(6)*sh4_7*x/10
    sh5_4 = -sqrt(3)*sh4_2*z/5 + 2*sqrt(6)*sh4_3*y/5 + sqrt(15)*sh4_4*x/5 + sqrt(3)*sh4_6*x/5
    sh5_5 = -sqrt(10)*sh4_3*x/5 + sh4_4*y - sqrt(10)*sh4_5*z/5
    sh5_6 = -sqrt(3)*sh4_2*x/5 + sqrt(15)*sh4_4*z/5 + 2*sqrt(6)*sh4_5*y/5 - sqrt(3)*sh4_6*z/5
    sh5_7 = -sqrt(6)*sh4_1*x/10 - sqrt(42)*sh4_3*x/10 + sqrt(42)*sh4_5*z/10 + sqrt(21)*sh4_6*y/5 - sqrt(6)*sh4_7*z/10
    sh5_8 = -sqrt(2)*sh4_0*x/10 - sqrt(14)*sh4_2*x/5 + sqrt(14)*sh4_6*z/5 + 4*sh4_7*y/5 - sqrt(2)*sh4_8*z/10
    sh5_9 = -3*sqrt(2)*sh4_1*x/5 + 3*sqrt(2)*sh4_7*z/5 + 3*sh4_8*y/5
    sh5_10 = 3*sqrt(10)*(-sh4_0*x + sh4_8*z)/10
    yield [sh5_0, sh5_1, sh5_2, sh5_3, sh5_4, sh5_5, sh5_6, sh5_7, sh5_8, sh5_9, sh5_10]

    sh6_0 = sqrt(33)*(sh5_0*z + sh5_10*x)/6
    sh6_1 = sqrt(11)*sh5_0*y/6 + sqrt(110)*sh5_1*z/12 + sqrt(110)*sh5_9*x/12
    sh6_2 = -sqrt(2)*sh5_0*z/12 + sqrt(5)*sh5_1*y/3 + sqrt(2)*sh5_10*x/12 + sqrt(10)*sh5_2*z/4 + sqrt(10)*sh5_8*x/4
    sh6_3 = -sqrt(6)*sh5_1*z/12 + sqrt(3)*sh5_2*y/2 + sqrt(2)*sh5_3*z/2 + sqrt(2)*sh5_7*x/2 + sqrt(6)*sh5_9*x/12
    sh6_4 = -sqrt(3)*sh5_2*z/6 + 2*sqrt(2)*sh5_3*y/3 + sqrt(14)*sh5_4*z/6 + sqrt(14)*sh5_6*x/6 + sqrt(3)*sh5_8*x/6
    sh6_5 = -sqrt(5)*sh5_3*z/6 + sqrt(35)*sh5_4*y/6 + sqrt(21)*sh5_5*x/6 + sqrt(5)*sh5_7*x/6
    sh6_6 = -sqrt(15)*sh5_4*x/6 + sh5_5*y - sqrt(15)*sh5_6*z/6
    sh6_7 = -sqrt(5)*sh5_3*x/6 + sqrt(21)*sh5_5*z/6 + sqrt(35)*sh5_6*y/6 - sqrt(5)*sh5_7*z/6
    sh6_8 = -sqrt(3)*sh5_2*x/6 - sqrt(14)*sh5_4*x/6 + sqrt(14)*sh5_6*z/6 + 2*sqrt(2)*sh5_7*y/3 - sqrt(3)*sh5_8*z/6
    sh6_9 = -sqrt(6)*sh5_1*x/12 - sqrt(2)*sh5_3*x/2 + sqrt(2)*sh5_7*z/2 + sqrt(3)*sh5_8*y/2 - sqrt(6)*sh5_9*z/12
    sh6_10 = -sqrt(2)*sh5_0*x/12 - sqrt(2)*sh5_10*z/12 - sqrt(10)*sh5_2*x/4 + sqrt(10)*sh5_8*z/4 + sqrt(5)*sh5_9*y/3
    sh6_11 = -sqrt(110)*sh5_1*x/12 + sqrt(11)*sh5_10*y/6 + sqrt(110)*sh5_9*z/12
    sh6_12 = sqrt(33)*(-sh5_0*x + sh5_10*z)/6
    yield [sh6_0, sh6_1, sh6_2, sh6_3, sh6_4, sh6_5, sh6_6, sh6_7, sh6_8, sh6_9, sh6_10, sh6_11, sh6_12]

    sh7_0 = sqrt(182)*(sh6_0*z + sh6_12*x)/14
    sh7_1 = sqrt(13)*sh6_0*y/7 + sqrt(39)*sh6_1*z/7 + sqrt(39)*sh6_11*x/7
    sh7_2 = -sqrt(2)*sh6_0*z/14 + 2*sqrt(6)*sh6_1*y/7 + sqrt(33)*sh6_10*x/7 + sqrt(2)*sh6_12*x/14 + sqrt(33)*sh6_2*z/7
    sh7_3 = -sqrt(6)*sh6_1*z/14 + sqrt(6)*sh6_11*x/14 + sqrt(33)*sh6_2*y/7 + sqrt(110)*sh6_3*z/14 + sqrt(110)*sh6_9*x/14
    sh7_4 = sqrt(3)*sh6_10*x/7 - sqrt(3)*sh6_2*z/7 + 2*sqrt(10)*sh6_3*y/7 + 3*sqrt(10)*sh6_4*z/14 + 3*sqrt(10)*sh6_8*x/14
    sh7_5 = -sqrt(5)*sh6_3*z/7 + 3*sqrt(5)*sh6_4*y/7 + 3*sqrt(2)*sh6_5*z/7 + 3*sqrt(2)*sh6_7*x/7 + sqrt(5)*sh6_9*x/7
    sh7_6 = -sqrt(30)*sh6_4*z/14 + 4*sqrt(3)*sh6_5*y/7 + 2*sqrt(7)*sh6_6*x/7 + sqrt(30)*sh6_8*x/14
    sh7_7 = -sqrt(21)*sh6_5*x/7 + sh6_6*y - sqrt(21)*sh6_7*z/7
    sh7_8 = -sqrt(30)*sh6_4*x/14 + 2*sqrt(7)*sh6_6*z/7 + 4*sqrt(3)*sh6_7*y/7 - sqrt(30)*sh6_8*z/14
    sh7_9 = -sqrt(5)*sh6_3*x/7 - 3*sqrt(2)*sh6_5*x/7 + 3*sqrt(2)*sh6_7*z/7 + 3*sqrt(5)*sh6_8*y/7 - sqrt(5)*sh6_9*z/7
    sh7_10 = -sqrt(3)*sh6_10*z/7 - sqrt(3)*sh6_2*x/7 - 3*sqrt(10)*sh6_4*x/14 + 3*sqrt(10)*sh6_8*z/14 + 2*sqrt(10)*sh6_9*y/7
    sh7_11 = -sqrt(6)*sh6_1*x/14 + sqrt(33)*sh6_10*y/7 - sqrt(6)*sh6_11*z/14 - sqrt(110)*sh6_3*x/14 + sqrt(110)*sh6_9*z/14
    sh7_12 = -sqrt(2)*sh6_0*x/14 + sqrt(33)*sh6_10*z/7 + 2*sqrt(6)*sh6_11*y/7 - sqrt(2)*sh6_12*z/14 - sqrt(33)*sh6_2*x/7
    sh7_13 = -sqrt(39)*sh6_1*x/7 + sqrt(39)*sh6_11*z/7 + sqrt(13)*sh6_12*y/7
    sh7_14 = sqrt(182)*(-sh6_0*x + sh6_12*z)/14
    yield [sh7_0, sh7_1, sh7_2, sh7_3, sh7_4, sh7_5, sh7_6, sh7_7, sh7_8, sh7_9, sh7_10, sh7_11, sh7_12, sh7_13, sh7_14]

    sh8_0 = sqrt(15)*(sh7_0*z + sh7_14*x)/4
    sh8_1 = sqrt(15)*sh7_0*y/8 + sqrt(210)*sh7_1*z/16 + sqrt(210)*sh7_13*x/16
    sh8_2 = -sqrt(2)*sh7_0*z/16 + sqrt(7)*sh7_1*y/4 + sqrt(182)*sh7_12*x/16 + sqrt(2)*sh7_14*x/16 + sqrt(182)*sh7_2*z/16
    sh8_3 = sqrt(510)*(-sqrt(85)*sh7_1*z + sqrt(2210)*sh7_11*x + sqrt(85)*sh7_13*x + sqrt(2210)*sh7_2*y + sqrt(2210)*sh7_3*z)/1360
    sh8_4 = sqrt(33)*sh7_10*x/8 + sqrt(3)*sh7_12*x/8 - sqrt(3)*sh7_2*z/8 + sqrt(3)*sh7_3*y/2 + sqrt(33)*sh7_4*z/8
    sh8_5 = sqrt(510)*(sqrt(102)*sh7_11*x - sqrt(102)*sh7_3*z + sqrt(1122)*sh7_4*y + sqrt(561)*sh7_5*z + sqrt(561)*sh7_9*x)/816
    sh8_6 = sqrt(30)*sh7_10*x/16 - sqrt(30)*sh7_4*z/16 + sqrt(15)*sh7_5*y/4 + 3*sqrt(10)*sh7_6*z/16 + 3*sqrt(10)*sh7_8*x/16
    sh8_7 = -sqrt(42)*sh7_5*z/16 + 3*sqrt(7)*sh7_6*y/8 + 3*sh7_7*x/4 + sqrt(42)*sh7_9*x/16
    sh8_8 = -sqrt(7)*sh7_6*x/4 + sh7_7*y - sqrt(7)*sh7_8*z/4
    sh8_9 = -sqrt(42)*sh7_5*x/16 + 3*sh7_7*z/4 + 3*sqrt(7)*sh7_8*y/8 - sqrt(42)*sh7_9*z/16
    sh8_10 = -sqrt(30)*sh7_10*z/16 - sqrt(30)*sh7_4*x/16 - 3*sqrt(10)*sh7_6*x/16 + 3*sqrt(10)*sh7_8*z/16 + sqrt(15)*sh7_9*y/4
    sh8_11 = sqrt(510)*(sqrt(1122)*sh7_10*y - sqrt(102)*sh7_11*z - sqrt(102)*sh7_3*x - sqrt(561)*sh7_5*x + sqrt(561)*sh7_9*z)/816
    sh8_12 = sqrt(33)*sh7_10*z/8 + sqrt(3)*sh7_11*y/2 - sqrt(3)*sh7_12*z/8 - sqrt(3)*sh7_2*x/8 - sqrt(33)*sh7_4*x/8
    sh8_13 = sqrt(510)*(-sqrt(85)*sh7_1*x + sqrt(2210)*sh7_11*z + sqrt(2210)*sh7_12*y - sqrt(85)*sh7_13*z - sqrt(2210)*sh7_3*x)/1360
    sh8_14 = -sqrt(2)*sh7_0*x/16 + sqrt(182)*sh7_12*z/16 + sqrt(7)*sh7_13*y/4 - sqrt(2)*sh7_14*z/16 - sqrt(182)*sh7_2*x/16
    sh8_15 = -sqrt(210)*sh7_1*x/16 + sqrt(210)*sh7_13*z/16 + sqrt(15)*sh7_14*y/8
    sh8_16 = sqrt(15)*(-sh7_0*x + sh7_14*z)/4
    yield [sh8_0, sh8_1, sh8_2, sh8_3, sh8_4, sh8_5, sh8_6, sh8_7, sh8_8, sh8_9, sh8_10, sh8_11, sh8_12, sh8_13, sh8_14, sh8_15, sh8_16]


def generate_spherical_harmonics():  # pragma: no cover
    import sympy

    xyz = sympy.symbols("x, y, z")

    print("sh0_0 = 1")
    print("yield [sh0_0]\n")

    sph_x = {
        0: sympy.Array([1]),
    }
    sph_1 = {
        0: sympy.Array([1]),
    }

    for l in range(8):
        d = 2 * l + 1
        names = [sympy.symbols(f"sh{l}_{m}") for m in range(d)]
        w = wigner_3j_sympy(1, l, l + 1)
        yx = sympy.Array([sum(xyz[i] * names[n] * w[i, n, m] for i in range(3) for n in range(d)) for m in range(d + 2)])

        if l <= 1:
            yx = yx.subs(zip(names, sph_x[l]))

        y1 = yx.subs(zip(xyz, (1, 0, 0))).subs(zip(names, sph_1[l]))
        norm = sympy.sqrt(sum(y1.applyfunc(lambda x: x**2)))
        y1 = y1 / norm
        yx = yx / norm
        yx = sympy.simplify(yx)

        sph_x[l + 1] = yx
        sph_1[l + 1] = y1

        # print code
        for m, p in enumerate(yx):
            print(f"sh{l+1}_{m} = {p}")

        print(f"yield [{', '.join([f'sh{l+1}_{m}' for m in range(d + 2)])}]\n")
