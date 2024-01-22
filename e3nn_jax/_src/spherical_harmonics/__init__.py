from functools import partial
from typing import List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

import e3nn_jax as e3nn

from .legendre import legendre_spherical_harmonics
from .recursive import recursive_spherical_harmonics


def sh(
    irreps_out: Union[e3nn.Irreps, int, Sequence[int]],
    input: jax.Array,
    normalize: bool,
    normalization: str = None,
    *,
    algorithm: Tuple[str, ...] = None,
) -> jax.Array:
    r"""Spherical harmonics.

    Same function as :func:`e3nn_jax.spherical_harmonics` but with a simple interface.

    Args:
        irreps_out (`Irreps` or int or Sequence[int]): the output irreps
        input (`jax.Array`): cartesian coordinates, shape (..., 3)
        normalize (bool): if True, the polynomials are restricted to the sphere
        normalization (str): normalization of the constant :math:`\text{cste}`. Default is 'component'
        algorithm (Tuple[str]): algorithm to use for the computation. (legendre|recursive, dense|sparse, [custom_jvp])

    Returns:
        `jax.Array`: polynomials of the spherical harmonics
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
    input: Union[e3nn.IrrepsArray, jax.Array],
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
        input (`IrrepsArray` or `jax.Array`): cartesian coordinates
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
    ls: Tuple[int, ...], x: jax.Array, normalization: str, algorithm: Tuple[str]
) -> List[jax.Array]:
    if "custom_jvp" in algorithm:
        return _custom_jvp_spherical_harmonics(ls, x, normalization, algorithm)
    else:
        return _spherical_harmonics(ls, x, normalization, algorithm)


def _spherical_harmonics(
    ls: Tuple[int, ...], x: jax.Array, normalization: str, algorithm: Tuple[str]
) -> List[jax.Array]:
    if "legendre" in algorithm:
        out = legendre_spherical_harmonics(max(ls), x, False, normalization)
        return [out[..., l**2 : (l + 1) ** 2] for l in ls]
    if "recursive" in algorithm:
        context = dict()
        for l in ls:
            recursive_spherical_harmonics(l, context, x, normalization, algorithm)
        return [context[l] for l in ls]
    raise ValueError("Unknown algorithm: must be 'legendre' or 'recursive'")


@partial(jax.custom_jvp, nondiff_argnums=(0, 2, 3))
def _custom_jvp_spherical_harmonics(
    ls: Tuple[int, ...], x: jax.Array, normalization: str, algorithm: Tuple[str]
) -> List[jax.Array]:
    return _spherical_harmonics(ls, x, normalization, algorithm)


@_custom_jvp_spherical_harmonics.defjvp
def _jvp(
    ls: Tuple[int, ...],
    normalization: str,
    algorithm: Tuple[str],
    primals: Tuple[jax.Array],
    tangents: Tuple[jax.Array],
) -> List[jax.Array]:
    (x,) = primals
    (x_dot,) = tangents

    js = tuple(max(0, l - 1) for l in ls)
    output = _custom_jvp_spherical_harmonics(ls + js, x, normalization, algorithm)
    primal, res = output[: len(ls)], output[len(ls) :]

    def h(l: int, r: jax.Array) -> jax.Array:
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
