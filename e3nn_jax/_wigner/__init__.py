import os
from functools import lru_cache, partial

import jax
import jax.numpy as jnp

from ._Gx import Gx
from ._J import Jd
from ._w3j import w3j


def wigner_J(l):
    if not l < len(Jd):
        raise NotImplementedError(f'wigner J maximum l implemented is {len(Jd) - 1}')

    return Jd[l]


def wigner_rot_y(l, angle):
    r"""
    Create the matrix representation of a y-axis rotation by the given angle,
    in the irrep l of dimension 2 * l + 1, in the basis of real centered
    spherical harmonics (RC basis in rep_bases.py in lie_learn).

    Note: this function is easy to use, but inefficient: only the entries
    on the diagonal and anti-diagonal are non-zero, so explicitly constructing
    this matrix is unnecessary.
    """
    angle = jnp.asarray(angle)
    shape = angle.shape
    M = jnp.zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = jnp.arange(0, 2 * l + 1, 1)
    reversed_inds = jnp.arange(2 * l, -1, -1)
    frequencies = jnp.arange(l, -l - 1, -1.0)
    M = M.at[..., inds, reversed_inds].set(jnp.sin(frequencies * angle[..., None]))
    M = M.at[..., inds, inds].set(jnp.cos(frequencies * angle[..., None]))
    return M


def wigner_generator_x(l):
    r"""
    Generator of rotation around X axis.

    Equivalent to ``wigner_J(l) @ wigner_generator_y(l) @ wigner_J(l)``.
    """
    return Gx[l]


def wigner_generator_y(l):
    r"""
    Generator of rotation around Y axis.
    """
    M = jnp.zeros((2 * l + 1, 2 * l + 1))
    inds = jnp.arange(0, 2 * l + 1, 1)
    reversed_inds = jnp.arange(2 * l, -1, -1)
    frequencies = jnp.arange(l, -l - 1, -1.0)
    M = M.at[..., inds, reversed_inds].set(frequencies)
    return M


def wigner_generator_z(l):
    r"""
    Generator of rotation around Z axis.
    """
    K = wigner_rot90_y(l)
    return K.T @ Gx[l] @ K


def wigner_rot90_y(l):
    r"""
    90 degree rotation around Y axis.
    """
    M = jnp.zeros((2 * l + 1, 2 * l + 1))
    inds = jnp.arange(0, 2 * l + 1, 1)
    reversed_inds = jnp.arange(2 * l, -1, -1)
    frequencies = jnp.arange(l, -l - 1, -1)
    M = M.at[..., inds, reversed_inds].set(jnp.array([0, 1, 0, -1])[frequencies % 4])
    M = M.at[..., inds, inds].set(jnp.array([1, 0, -1, 0])[frequencies % 4])
    return M


@partial(jax.jit, static_argnums=(0,), inline=True)
def wigner_D(l, alpha, beta, gamma):
    r"""Wigner D matrix representation of :math:`SO(3)`.

    It satisfies the following properties:

    * :math:`D(\text{identity rotation}) = \text{identity matrix}`
    * :math:`D(R_1 \circ R_2) = D(R_1) \circ D(R_2)`
    * :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`
    * :math:`D(\text{rotation around Y axis})` has some property that allows us to use FFT in `ToS2Grid`

    Code of this function has beed copied from `lie_learn <https://github.com/AMLab-Amsterdam/lie_learn>`_ made by Taco Cohen.

    Args:
        l (int): :math:`l`
        alpha: rotation :math:`\alpha` around Y axis, applied third.
        beta: rotation :math:`\beta` around X axis, applied second.
        gamma: rotation :math:`\gamma` around Y axis, applied first.

    Returns:
        array :math:`D^l(\alpha, \beta, \gamma)` of shape :math:`(..., 2l+1, 2l+1)`
    """
    alpha, beta, gamma = jnp.broadcast_arrays(alpha, beta, gamma)
    Xa = wigner_rot_y(l, alpha)
    Xb = wigner_rot_y(l, beta)
    Xc = wigner_rot_y(l, gamma)
    J = wigner_J(l)
    return Xa @ J @ Xb @ J @ Xc


def wigner_3j(l1, l2, l3):
    r"""Wigner 3j symbols :math:`C_{lmn}`.

    It satisfies the following two properties:

        .. math::

            C_{lmn} = C_{ijk} D_{il}(g) D_{jm}(g) D_{kn}(g) \qquad \forall g \in SO(3)

        where :math:`D` are given by `wigner_D`.

        .. math::

            C_{ijk} C_{ijk} = 1

    Args:
        l1 (int): :math:`l_1`
        l2 (int): :math:`l_2`
        l3 (int): :math:`l_3`

    Returns:
        array :math:`C_{lmn}` of shape :math:`(2l_1+1, 2l_2+1, 2l_3+1)`
    """
    assert abs(l2 - l3) <= l1 <= l2 + l3

    try:
        if l1 <= l2 <= l3:
            out = w3j[(l1, l2, l3)].reshape(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)
        if l1 <= l3 <= l2:
            out = w3j[(l1, l3, l2)].reshape(2 * l1 + 1, 2 * l3 + 1, 2 * l2 + 1).transpose(0, 2, 1) * ((-1) ** (l1 + l2 + l3))
        if l2 <= l1 <= l3:
            out = w3j[(l2, l1, l3)].reshape(2 * l2 + 1, 2 * l1 + 1, 2 * l3 + 1).transpose(1, 0, 2) * ((-1) ** (l1 + l2 + l3))
        if l3 <= l2 <= l1:
            out = w3j[(l3, l2, l1)].reshape(2 * l3 + 1, 2 * l2 + 1, 2 * l1 + 1).transpose(2, 1, 0) * ((-1) ** (l1 + l2 + l3))
        if l2 <= l3 <= l1:
            out = w3j[(l2, l3, l1)].reshape(2 * l2 + 1, 2 * l3 + 1, 2 * l1 + 1).transpose(2, 0, 1)
        if l3 <= l1 <= l2:
            out = w3j[(l3, l1, l2)].reshape(2 * l3 + 1, 2 * l1 + 1, 2 * l2 + 1).transpose(1, 2, 0)
    except KeyError:
        raise NotImplementedError(f'Wigner 3j symbols maximum l implemented is {max(w3j.keys())[0]}')

    return out


def wigner_3j_sympy(l1, l2, l3):
    import sympy
    assert abs(l2 - l3) <= l1 <= l2 + l3

    if l1 <= l2 <= l3:
        out = _wigner_3j_sympy(l1, l2, l3).reshape(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)
    if l1 <= l3 <= l2:
        out = sympy.permutedims(_wigner_3j_sympy(l1, l3, l2).reshape(2 * l1 + 1, 2 * l3 + 1, 2 * l2 + 1), (0, 2, 1)) * ((-1) ** (l1 + l2 + l3))
    if l2 <= l1 <= l3:
        out = sympy.permutedims(_wigner_3j_sympy(l2, l1, l3).reshape(2 * l2 + 1, 2 * l1 + 1, 2 * l3 + 1), (1, 0, 2)) * ((-1) ** (l1 + l2 + l3))
    if l3 <= l2 <= l1:
        out = sympy.permutedims(_wigner_3j_sympy(l3, l2, l1).reshape(2 * l3 + 1, 2 * l2 + 1, 2 * l1 + 1), (2, 1, 0)) * ((-1) ** (l1 + l2 + l3))
    if l2 <= l3 <= l1:
        out = sympy.permutedims(_wigner_3j_sympy(l2, l3, l1).reshape(2 * l2 + 1, 2 * l3 + 1, 2 * l1 + 1), (2, 0, 1))
    if l3 <= l1 <= l2:
        out = sympy.permutedims(_wigner_3j_sympy(l3, l1, l2).reshape(2 * l3 + 1, 2 * l1 + 1, 2 * l2 + 1), (1, 2, 0))
    return out


@lru_cache()
def _cached_simplify(expr):
    import sympy
    return sympy.simplify(expr)


@lru_cache(maxsize=None)
def _wigner_3j_sympy(l1, l2, l3):
    import sympy

    with open(os.path.join(os.path.dirname(__file__), '_w3j.py'), 'rt') as f:
        xs = f.read().split("# split")[1:-1]

    for x in xs:
        a, bs = x.split(": np.array([")
        a = eval(a.strip())
        if a == (l1, l2, l3):
            b = [_cached_simplify(b.strip()) for b in bs.split(',')[:-2]]
            return sympy.Array(b)
