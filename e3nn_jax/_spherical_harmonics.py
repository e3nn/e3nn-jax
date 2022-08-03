r"""Spherical Harmonics as polynomials of x, y, z
"""
import fractions
import math
from functools import partial
from typing import Dict, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import sympy

from e3nn_jax import Irreps, IrrepsArray, clebsch_gordan, config
from e3nn_jax.util.sympy import sqrtQarray_to_sympy


def sh(
    irreps_out: Union[Irreps, int, Sequence[int]],
    input: jnp.ndarray,
    normalize: bool,
    normalization: str = None,
    *,
    algorithm: Tuple[str] = None,
) -> jnp.ndarray:
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
        irreps_out (`Irreps` or int or Sequence[int]): the output irreps
        input (``jnp.ndarray``): cartesian coordinates
        normalize (bool): if True, the polynomials are restricted to the sphere
        normalization (str): normalization of the constant :math:`\text{cste}`. Default is 'integral'
        algorithm (Tuple[str]): algorithm to use for the computation. (legendre|recursive, dense|sparse, [custom_vjp])

    Returns:
        ``jnp.ndarray``: polynomials of the spherical harmonics
    """
    input = IrrepsArray("1e", input)
    return spherical_harmonics(irreps_out, input, normalize, normalization, algorithm=algorithm).array


def spherical_harmonics(
    irreps_out: Union[Irreps, int, Sequence[int]],
    input: Union[IrrepsArray, jnp.ndarray],
    normalize: bool,
    normalization: str = None,
    *,
    algorithm: Tuple[str] = None,
) -> IrrepsArray:
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

        Y^{l+1}_i(x) &= \text{cste}(l) \; C_{ijk} Y^l_j(x) x_k

        \partial_k Y^{l+1}_i(x) &= \text{cste}(l) \; (l+1) C_{ijk} Y^l_j(x)

    Where :math:`C` are the `clebsch_gordan`.

    .. note::

        This function match with this table of standard real spherical harmonics from Wikipedia_
        when ``normalize=True``, ``normalization='integral'`` and is called with the argument in the order ``y,z,x``
        (instead of ``x,y,z``).

    .. _Wikipedia: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics

    Args:
        irreps_out (`Irreps` or int): output irreps
        input (`IrrepsArray` or ``jnp.ndarray``): cartesian coordinates
        normalize (bool): if True, the polynomials are restricted to the sphere
        normalization (str): normalization of the constant :math:`\text{cste}`. Default is 'integral'
        algorithm (Tuple[str]): algorithm to use for the computation. (legendre|recursive, dense|sparse, [custom_vjp])

    Returns:
        `IrrepsArray`: polynomials of the spherical harmonics
    """
    if normalization is None:
        normalization = config("spherical_harmonics_normalization")
    assert normalization in ["integral", "component", "norm"]

    if isinstance(irreps_out, int):
        l = irreps_out
        assert isinstance(input, IrrepsArray)
        [(mul, ir)] = input.irreps
        irreps_out = Irreps([(1, (l, ir.p**l))])

    if all(isinstance(l, int) for l in irreps_out):
        assert isinstance(input, IrrepsArray)
        [(mul, ir)] = input.irreps
        irreps_out = Irreps([(1, (l, ir.p**l)) for l in irreps_out])

    irreps_out = Irreps(irreps_out)

    assert all([l % 2 == 1 or p == 1 for _, (l, p) in irreps_out])
    assert len(set([p for _, (l, p) in irreps_out if l % 2 == 1])) <= 1

    if algorithm is None:
        if config("spherical_harmonics_algorithm") == "automatic":
            if irreps_out.lmax <= 8:
                algorithm = ("recursive", "sparse", "custom_vjp")
            else:
                algorithm = ("legendre", "sparse", "custom_vjp")
        else:
            algorithm = config("spherical_harmonics_algorithm")

    assert all(keyword in ["legendre", "recursive", "dense", "sparse", "custom_vjp"] for keyword in algorithm)

    if isinstance(input, IrrepsArray):
        [(mul, ir)] = input.irreps
        assert mul == 1
        assert ir.l == 1
        assert all([ir.p == p for _, (l, p) in irreps_out if l % 2 == 1])
        x = input.array
    else:
        x = input

    assert x.shape[-1] == 3
    if normalize:
        r = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
        x = x / jnp.where(r == 0.0, 1.0, r)

    sh = _jited_spherical_harmonics(tuple(ir.l for _, ir in irreps_out), x, normalization, algorithm)
    sh = [jnp.repeat(y[..., None, :], mul, -2) for (mul, ir), y in zip(irreps_out, sh)]
    return IrrepsArray.from_list(irreps_out, sh, x.shape[:-1])


@partial(jax.jit, static_argnums=(0, 2, 3), inline=True)
def _jited_spherical_harmonics(
    ls: Tuple[int, ...], x: jnp.ndarray, normalization: str, algorithm: Tuple[str]
) -> List[jnp.ndarray]:
    if "custom_vjp" in algorithm:
        return _custom_vjp_spherical_harmonics(ls, x, normalization, algorithm)
    else:
        return _spherical_harmonics(ls, x, normalization, algorithm)


def _spherical_harmonics(ls: Tuple[int, ...], x: jnp.ndarray, normalization: str, algorithm: Tuple[str]) -> List[jnp.ndarray]:
    if "legendre" in algorithm:
        out = _legendre_spherical_harmonics(max(ls), x, False, normalization)
        return [out[..., l**2 : (l + 1) ** 2] for l in ls]
    if "recursive" in algorithm:
        context = dict()
        for l in ls:
            _recursive_spherical_harmonics(l, context, x, normalization, algorithm)
        return [context[l] for l in ls]
    raise ValueError("Unknown algorithm: must be 'legendre' or 'recursive'")


@partial(jax.custom_jvp, nondiff_argnums=(0, 2, 3))
@partial(jax.custom_vjp, nondiff_argnums=(0, 2, 3))
def _custom_vjp_spherical_harmonics(
    ls: Tuple[int, ...], x: jnp.ndarray, normalization: str, algorithm: Tuple[str]
) -> List[jnp.ndarray]:
    return _spherical_harmonics(ls, x, normalization, algorithm)


def _fwd(
    ls: Tuple[int, ...], x: jnp.ndarray, normalization: str, algorithm: Tuple[str]
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    js = tuple(max(0, l - 1) for l in ls)
    output = _custom_vjp_spherical_harmonics(ls + js, x, normalization, algorithm)

    return output[: len(ls)], output[len(ls) :]


def _bwd(
    ls: Tuple[int, ...], normalization: str, algorithm: Tuple[str], res: List[jnp.ndarray], grad: List[jnp.ndarray]
) -> jnp.ndarray:
    # algo in list with different code per L is faster to execute but very slow to compile!
    # TODO implement a dense version of this. No list per l. Can use jax.lax.fori_loop
    # TODO it could be max(ls) dependant. for max(ls) > 10, use a dense algorithm!

    def h(l, r, g):
        w = clebsch_gordan(l - 1, l, 1)
        if normalization == "norm":
            w *= ((2 * l + 1) * l * (2 * l - 1)) ** 0.5
        else:
            w *= l**0.5 * (2 * l + 1)

        if "dense" in algorithm:
            return jnp.einsum("...i,...j,ijk->...k", r, g, w)
        if "sparse" in algorithm:
            return jnp.stack(
                [
                    sum(
                        [
                            w[i, j, k] * r[..., i] * g[..., j]
                            for i in range(2 * l - 1)
                            for j in range(2 * l + 1)
                            if w[i, j, k] != 0
                        ]
                    )
                    for k in range(3)
                ],
                axis=-1,
            )
        raise ValueError("Unknown algorithm: must be 'dense' or 'sparse'")

    return (sum([h(l, r, g) if l > 0 else jnp.zeros_like(r, shape=r.shape[:-1] + (3,)) for l, r, g in zip(ls, res, grad)]),)


def _jvp(
    ls: Tuple[int, ...], normalization: str, algorithm: Tuple[str], primals: Tuple[jnp.ndarray], tangents: Tuple[jnp.ndarray]
) -> List[jnp.ndarray]:
    (x,) = primals
    (x_dot,) = tangents

    js = tuple(max(0, l - 1) for l in ls)
    output = _custom_vjp_spherical_harmonics(ls + js, x, normalization, algorithm)
    out, res = output[: len(ls)], output[len(ls) :]

    def h(l: int, r: jnp.ndarray) -> jnp.ndarray:
        w = clebsch_gordan(l - 1, l, 1)
        if normalization == "norm":
            w *= ((2 * l + 1) * l * (2 * l - 1)) ** 0.5
        else:
            w *= l**0.5 * (2 * l + 1)

        if "dense" in algorithm:
            return jnp.einsum("...i,...k,ijk->...j", r, x_dot, w)
        if "sparse" in algorithm:
            return jnp.stack(
                [
                    sum(
                        [w[i, j, k] * r[..., i] * x_dot[..., k] for i in range(2 * l - 1) for k in range(3) if w[i, j, k] != 0]
                    )
                    for j in range(2 * l + 1)
                ],
                axis=-1,
            )
        raise ValueError("Unknown algorithm: must be 'dense' or 'sparse'")

    return out, [h(l, r) if l > 0 else jnp.zeros_like(r) for l, r in zip(ls, res)]


_custom_vjp_spherical_harmonics.defvjp(_fwd, _bwd)
_custom_vjp_spherical_harmonics.defjvp(_jvp)


def _recursive_spherical_harmonics(
    l: int, context: Dict[int, jnp.ndarray], input: jnp.ndarray, normalization: str, algorithm: Tuple[str]
) -> sympy.Array:
    context.update(dict(jnp=jnp, clebsch_gordan=clebsch_gordan))

    if 0 not in context:
        if normalization == "integral":
            context[0] = math.sqrt(1 / (4 * math.pi)) * jnp.ones_like(input[..., :1])
            context[1] = math.sqrt(3 / (4 * math.pi)) * input
        elif normalization == "component":
            context[0] = jnp.ones_like(input[..., :1])
            context[1] = math.sqrt(3) * input
        else:
            context[0] = jnp.ones_like(input[..., :1])
            context[1] = input

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

    sph_1_l1 = _recursive_spherical_harmonics(l1, context, input, normalization, algorithm)
    sph_1_l2 = _recursive_spherical_harmonics(l2, context, input, normalization, algorithm)

    y1 = yx.subs(zip(sh_var(l1), sph_1_l1)).subs(zip(sh_var(l2), sph_1_l2))
    norm = sympy.sqrt(sum(y1.applyfunc(lambda x: x**2)))
    y1 = y1 / norm

    if l not in context:
        if normalization == "integral":
            x = math.sqrt((2 * l + 1) / (4 * math.pi)) / (
                math.sqrt((2 * l1 + 1) / (4 * math.pi)) * math.sqrt((2 * l2 + 1) / (4 * math.pi))
            )
        elif normalization == "component":
            x = math.sqrt((2 * l + 1) / ((2 * l1 + 1) * (2 * l2 + 1)))
        else:
            x = 1

        w = (x / float(norm)) * clebsch_gordan(l1, l2, l)

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


@partial(jax.jit, static_argnums=(0,))
def legendre(lmax: int, x: jnp.ndarray, phase: float) -> jnp.ndarray:
    r"""Associated Legendre polynomials

    en.wikipedia.org/wiki/Associated_Legendre_polynomials

    code inspired by: https://github.com/SHTOOLS/SHTOOLS/blob/master/src/PlmBar.f95

    Args:
        lmax: maximum l value
        x: input array of shape ``(...)``
        phase: -1 or 1, multiplies by :math:`(-1)^m`

    Returns:
        Associated Legendre polynomials ``P(l,m)``
        In an array of shape ``((lmax + 1) * (lmax + 2) // 2, ...)``
        ``(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), ...``
    """
    x = jnp.asarray(x)

    p = jnp.zeros(((lmax + 1) * (lmax + 2) // 2,) + x.shape)

    scalef = {
        jnp.dtype("float32"): 1e-35,
        jnp.dtype("float64"): 1e-280,
    }[x.dtype]

    def k(l, m):
        return l * (l + 1) // 2 + m

    def f1(l, m):
        return (2 * l - 1) / (l - m)

    def f2(l, m):
        return (l + m - 1) / (l - m)

    # Calculate P(l,0). These are not scaled.
    u = jnp.sqrt((1.0 - x) * (1.0 + x))  # sin(theta)

    p = p.at[k(0, 0)].set(1.0)
    if lmax == 0:
        return p

    p = p.at[k(1, 0)].set(x)

    p = jax.lax.fori_loop(
        2,
        lmax + 1,
        lambda l, p: p.at[k(l, 0)].set(f1(l, 0) * x * p[k(l - 1, 0)] - f2(l, 0) * p[k(l - 2, 0)]),
        p,
    )

    # Calculate P(m,m), P(m+1,m), and P(l,m)
    def g(m, vals):
        p, pmm, rescalem = vals
        rescalem = rescalem * u

        # Calculate P(m,m)
        pmm = phase * (2 * m - 1) * pmm
        p = p.at[k(m, m)].set(pmm)

        # Calculate P(m+1,m)
        p = p.at[k(m + 1, m)].set(x * (2 * m + 1) * pmm)

        # Calculate P(l,m)
        def f(l, p):
            p = p.at[k(l, m)].set(f1(l, m) * x * p[k(l - 1, m)] - f2(l, m) * p[k(l - 2, m)])
            p = p.at[k(l - 2, m)].multiply(rescalem)
            return p

        p = jax.lax.fori_loop(m + 2, lmax + 1, f, p)

        p = p.at[k(lmax - 1, m)].multiply(rescalem)
        p = p.at[k(lmax, m)].multiply(rescalem)

        return p, pmm, rescalem

    pmm = scalef  # P(0,0) * scalef
    rescalem = jnp.ones_like(x) / scalef
    p, pmm, rescalem = jax.lax.fori_loop(1, lmax, g, (p, pmm, rescalem))

    # Calculate P(lmax,lmax)
    rescalem = rescalem * u
    p = p.at[k(lmax, lmax)].set(phase * (2 * lmax - 1) * pmm * rescalem)

    return p


def _sh_alpha(l: int, alpha: jnp.ndarray) -> jnp.ndarray:
    r"""

    Args:
        l: l value
        alpha: input array of shape ``(...)``

    Returns:
        Array of shape ``(..., 2 * l + 1)``
    """
    alpha = alpha[..., None]  # [..., 1]
    m = jnp.arange(1, l + 1)  # [1, 2, 3, ..., l]
    cos = jnp.cos(m * alpha)  # [..., m]

    m = jnp.arange(l, 0, -1)  # [l, l-1, l-2, ..., 1]
    sin = jnp.sin(m * alpha)  # [..., m]

    return jnp.concatenate(
        [
            jnp.sqrt(2) * sin,
            jnp.ones_like(alpha),
            jnp.sqrt(2) * cos,
        ],
        axis=-1,
    )


def _legendre_spherical_harmonics(lmax: int, x: jnp.ndarray, normalize: bool, normalization: str) -> jnp.ndarray:
    alpha = jnp.arctan2(x[..., 0], x[..., 2])
    sh_alpha = _sh_alpha(lmax, alpha)  # [..., 2 * l + 1]

    n = jnp.linalg.norm(x, axis=-1, keepdims=True)
    x = x / jnp.where(n > 0, n, 1.0)

    sh_y = legendre(lmax, x[..., 1], 1.0)  # [(lmax + 1) * (lmax + 2) // 2, ...]
    sh_y = jnp.moveaxis(sh_y, 0, -1)  # [..., (lmax + 1) * (lmax + 2) // 2]

    f = np.array(
        [
            math.sqrt(fractions.Fraction((2 * l + 1) * math.factorial(l - m), 4 * math.factorial(l + m)) / math.pi)
            for l in range(lmax + 1)
            for m in range(l + 1)
        ]
    )
    sh_y = f * sh_y

    def f(l, sh):
        def g(m, sh):
            y = sh_y[..., l * (l + 1) // 2 + jnp.abs(m)]
            if not normalize:
                y = y * n[..., 0] ** l
            if normalization == "norm":
                y = y * (jnp.sqrt(4 * jnp.pi) / jnp.sqrt(2 * l + 1))
            elif normalization == "component":
                y = y * jnp.sqrt(4 * jnp.pi)

            a = sh_alpha[..., lmax + m]
            return sh.at[..., l**2 + l + m].set(y * a)

        return jax.lax.fori_loop(-l, l + 1, g, sh)

    sh = jnp.zeros(x.shape[:-1] + ((lmax + 1) ** 2,))
    return jax.lax.fori_loop(0, lmax + 1, f, sh)
