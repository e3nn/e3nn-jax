import fractions
import math
from functools import partial
from typing import Dict, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import sympy

import e3nn_jax as e3nn
from e3nn_jax._src.utils.sympy import sqrtQarray_to_sympy


def sh(
    irreps_out: Union[e3nn.Irreps, int, Sequence[int]],
    input: jnp.ndarray,
    normalize: bool,
    normalization: str = None,
    *,
    algorithm: Tuple[str, ...] = None,
) -> jnp.ndarray:
    r"""Spherical harmonics.

    Same function as :func:`e3nn_jax.spherical_harmonics` but with a simple interface.

    Args:
        irreps_out (`Irreps` or int or Sequence[int]): the output irreps
        input (`jax.numpy.ndarray`): cartesian coordinates, shape (..., 3)
        normalize (bool): if True, the polynomials are restricted to the sphere
        normalization (str): normalization of the constant :math:`\text{cste}`. Default is 'integral'
        algorithm (Tuple[str]): algorithm to use for the computation. (legendre|recursive, dense|sparse, [custom_jvp])

    Returns:
        `jax.numpy.ndarray`: polynomials of the spherical harmonics
    """
    input = e3nn.IrrepsArray("1e", input)
    return spherical_harmonics(
        irreps_out, input, normalize, normalization, algorithm=algorithm
    ).array


def _check_is_vector(irreps: e3nn.Irreps):
    if irreps.num_irreps != 1:
        raise ValueError("Input must be a single vector (1x1o or 1x1e).")
    [(mul, ir)] = irreps
    if not (mul == 1 and ir.l == 1):
        raise ValueError("Input must be a vector (1x1o or 1x1e).")
    return ir.p


def spherical_harmonics(
    irreps_out: Union[e3nn.Irreps, int, Sequence[int]],
    input: Union[e3nn.IrrepsArray, jnp.ndarray],
    normalize: bool,
    normalization: str = None,
    *,
    algorithm: Tuple[str, ...] = None,
) -> e3nn.IrrepsArray:
    r"""Spherical harmonics.

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
        irreps_out (`Irreps` or list of int or int): output irreps
        input (`IrrepsArray` or `jax.numpy.ndarray`): cartesian coordinates
        normalize (bool): if True, the polynomials are restricted to the sphere
        normalization (str): normalization of the constant :math:`\text{cste}`. Default is 'component'
        algorithm (Tuple[str]): algorithm to use for the computation. (legendre|recursive, dense|sparse, [custom_jvp])

    Returns:
        `IrrepsArray`: polynomials of the spherical harmonics
    """
    if normalization is None:
        normalization = e3nn.config("spherical_harmonics_normalization")
    assert normalization in ["integral", "component", "norm"]

    if isinstance(irreps_out, str):
        irreps_out = e3nn.Irreps(irreps_out)

    if not isinstance(irreps_out, e3nn.Irreps):
        if isinstance(irreps_out, range):
            irreps_out = list(irreps_out)

        if isinstance(irreps_out, int):
            l = irreps_out
            if not isinstance(input, e3nn.IrrepsArray):
                raise ValueError(
                    "If irreps_out is an int, input must be an IrrepsArray."
                )
            vec_p = _check_is_vector(input.irreps)
            irreps_out = e3nn.Irreps([(1, (l, vec_p**l))])

        if all(isinstance(l, int) for l in irreps_out):
            if not isinstance(input, e3nn.IrrepsArray):
                raise ValueError(
                    "If irreps_out is a list of int, input must be an IrrepsArray."
                )
            vec_p = _check_is_vector(input.irreps)
            irreps_out = e3nn.Irreps([(1, (l, vec_p**l)) for l in irreps_out])

    irreps_out = e3nn.Irreps(irreps_out)

    assert all([l % 2 == 1 or p == 1 for _, (l, p) in irreps_out])
    assert len(set([p for _, (l, p) in irreps_out if l % 2 == 1])) <= 1

    if isinstance(input, e3nn.IrrepsArray):
        vec_p = _check_is_vector(input.irreps)
        if not all([vec_p == p for _, (l, p) in irreps_out if l % 2 == 1]):
            raise ValueError(
                f"Input ({input.irreps}) and output ({irreps_out}) must have a compatible parity."
            )

        x = input.array
    else:
        x = input

    if irreps_out.num_irreps == 0:
        return e3nn.IrrepsArray(irreps_out, jnp.zeros(x.shape[:-1] + (0,)))

    if algorithm is None:
        if e3nn.config("spherical_harmonics_algorithm") == "automatic":
            # NOTE the dense algorithm is faster to jit than the sparse one
            if irreps_out.lmax <= 8:
                algorithm = ("recursive", "dense", "custom_jvp")
            else:
                algorithm = ("legendre", "dense", "custom_jvp")
        else:
            algorithm = e3nn.config("spherical_harmonics_algorithm")

    assert all(
        keyword in ["legendre", "recursive", "dense", "sparse", "custom_jvp"]
        for keyword in algorithm
    )

    assert x.shape[-1] == 3
    if normalize:
        r2 = jnp.sum(x**2, axis=-1, keepdims=True)
        r2 = jnp.where(r2 == 0.0, 1.0, r2)
        x = x / jnp.sqrt(r2)

    sh = _jited_spherical_harmonics(
        tuple(ir.l for _, ir in irreps_out), x, normalization, algorithm
    )
    sh = [
        jnp.repeat(y[..., None, :], mul, -2) if mul != 1 else y[..., None, :]
        for (mul, ir), y in zip(irreps_out, sh)
    ]
    return e3nn.from_chunks(irreps_out, sh, x.shape[:-1], x.dtype)


@partial(jax.jit, static_argnums=(0, 2, 3), inline=True)
def _jited_spherical_harmonics(
    ls: Tuple[int, ...], x: jnp.ndarray, normalization: str, algorithm: Tuple[str]
) -> List[jnp.ndarray]:
    if "custom_jvp" in algorithm:
        return _custom_jvp_spherical_harmonics(ls, x, normalization, algorithm)
    else:
        return _spherical_harmonics(ls, x, normalization, algorithm)


def _spherical_harmonics(
    ls: Tuple[int, ...], x: jnp.ndarray, normalization: str, algorithm: Tuple[str]
) -> List[jnp.ndarray]:
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
def _custom_jvp_spherical_harmonics(
    ls: Tuple[int, ...], x: jnp.ndarray, normalization: str, algorithm: Tuple[str]
) -> List[jnp.ndarray]:
    return _spherical_harmonics(ls, x, normalization, algorithm)


@_custom_jvp_spherical_harmonics.defjvp
def _jvp(
    ls: Tuple[int, ...],
    normalization: str,
    algorithm: Tuple[str],
    primals: Tuple[jnp.ndarray],
    tangents: Tuple[jnp.ndarray],
) -> List[jnp.ndarray]:
    (x,) = primals
    (x_dot,) = tangents

    js = tuple(max(0, l - 1) for l in ls)
    output = _custom_jvp_spherical_harmonics(ls + js, x, normalization, algorithm)
    primal, res = output[: len(ls)], output[len(ls) :]

    def h(l: int, r: jnp.ndarray) -> jnp.ndarray:
        w = e3nn.clebsch_gordan(l - 1, l, 1)
        if normalization == "norm":
            w *= ((2 * l + 1) * l * (2 * l - 1)) ** 0.5
        else:
            w *= l**0.5 * (2 * l + 1)
        w = w.astype(x.dtype)

        if "dense" in algorithm:
            return jnp.einsum("...i,...k,ijk->...j", r, x_dot, w)
        if "sparse" in algorithm:
            return jnp.stack(
                [
                    sum(
                        [
                            w[i, j, k] * r[..., i] * x_dot[..., k]
                            for i in range(2 * l - 1)
                            for k in range(3)
                            if w[i, j, k] != 0
                        ]
                    )
                    for j in range(2 * l + 1)
                ],
                axis=-1,
            )
        raise ValueError("Unknown algorithm: must be 'dense' or 'sparse'")

    tangent = [h(l, r) if l > 0 else jnp.zeros_like(r) for l, r in zip(ls, res)]
    return primal, tangent


def _recursive_spherical_harmonics(
    l: int,
    context: Dict[int, jnp.ndarray],
    input: jnp.ndarray,
    normalization: str,
    algorithm: Tuple[str],
) -> sympy.Array:
    context.update(dict(jnp=jnp, clebsch_gordan=e3nn.clebsch_gordan))

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
                context[1] = input

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

    sph_1_l1 = _recursive_spherical_harmonics(
        l1, context, input, normalization, algorithm
    )
    sph_1_l2 = _recursive_spherical_harmonics(
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


@partial(jax.jit, static_argnums=(0,))
def legendre(lmax: int, x: jnp.ndarray, phase: float) -> jnp.ndarray:
    r"""Associated Legendre polynomials.

    en.wikipedia.org/wiki/Associated_Legendre_polynomials

    code inspired by: https://github.com/SHTOOLS/SHTOOLS/blob/master/src/PlmBar.f95

    Args:
        lmax (int): maximum l value
        x (jnp.ndarray): input array of shape ``(...)``
        phase (float): -1 or 1, multiplies by :math:`(-1)^m`

    Returns:
        jnp.ndarray: Associated Legendre polynomials ``P(l,m)``
        In an array of shape ``((lmax + 1) * (lmax + 2) // 2, ...)``
        ``(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), ...``
    """
    x = jnp.asarray(x)

    p = jnp.zeros(((lmax + 1) * (lmax + 2) // 2,) + x.shape, x.dtype)

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
        lambda l, p: p.at[k(l, 0)].set(
            f1(l, 0) * x * p[k(l - 1, 0)] - f2(l, 0) * p[k(l - 2, 0)]
        ),
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
            p = p.at[k(l, m)].set(
                f1(l, m) * x * p[k(l - 1, m)] - f2(l, m) * p[k(l - 2, m)]
            )
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
    r"""Alpha dependence of spherical harmonics.

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


def _sh_beta(lmax: int, cos_betas: jnp.ndarray) -> jnp.ndarray:
    r"""Beta dependence of spherical harmonics.

    Args:
        lmax: l value
        cos_betas: input array of shape ``(...)``

    Returns:
        Array of shape ``(..., (lmax + 1) * (lmax + 2) // 2 + 1)``
    """
    sh_y = legendre(lmax, cos_betas, 1.0)  # [(lmax + 1) * (lmax + 2) // 2, ...]
    sh_y = jnp.moveaxis(sh_y, 0, -1)  # [..., (lmax + 1) * (lmax + 2) // 2]

    sh_y = sh_y * np.array(
        [
            math.sqrt(
                fractions.Fraction(
                    (2 * l + 1) * math.factorial(l - m), 4 * math.factorial(l + m)
                )
                / math.pi
            )
            for l in range(lmax + 1)
            for m in range(l + 1)
        ],
        sh_y.dtype,
    )
    return sh_y


def _legendre_spherical_harmonics(
    lmax: int, x: jnp.ndarray, normalize: bool, normalization: str
) -> jnp.ndarray:
    alpha = jnp.arctan2(x[..., 0], x[..., 2])
    sh_alpha = _sh_alpha(lmax, alpha)  # [..., 2 * l + 1]

    n = jnp.linalg.norm(x, axis=-1, keepdims=True)
    x = x / jnp.where(n > 0, n, 1.0)

    sh_y = _sh_beta(lmax, x[..., 1])  # [..., (lmax + 1) * (lmax + 2) // 2]

    sh = jnp.zeros(x.shape[:-1] + ((lmax + 1) ** 2,), x.dtype)

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

    sh = jax.lax.fori_loop(0, lmax + 1, f, sh)
    return sh
