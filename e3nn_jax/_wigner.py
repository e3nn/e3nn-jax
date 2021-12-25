from functools import partial

import jax
import jax.numpy as jnp

from ._constants._G_beta import G_beta
from ._constants._J import Jd
from ._constants._w3j import w3j


def wigner_J(l):
    return Jd[l]


def _y_rot_mat(l, angle):
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


def wigner_generator_alpha(l):
    r"""
    Generator for the angle alpha of the Wigner D matrices.

    .. math::
        1 + \alpha G

    is the infinitesimal rotation matrix around Y axis.

    Args:
        l (int): :math:`l`

    Returns:
        jnp.array: matrix of shape :math:`(2l+1, 2l+1)`
    """
    M = jnp.zeros((2 * l + 1, 2 * l + 1))
    inds = jnp.arange(0, 2 * l + 1, 1)
    reversed_inds = jnp.arange(2 * l, -1, -1)
    frequencies = jnp.arange(l, -l - 1, -1.0)
    M = M.at[..., inds, reversed_inds].set(frequencies)
    return M


def wigner_generator_beta(l):
    return G_beta[l]


def wigner_generator_delta(l):
    K = _rot_90_alpha(l)
    return K @ G_beta[l] @ K.T


def _rot_90_alpha(l):
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
    if not l < len(Jd):
        raise NotImplementedError(f'wigner D maximum l implemented is {len(Jd) - 1}')

    alpha, beta, gamma = jnp.broadcast_arrays(alpha, beta, gamma)
    Xa = _y_rot_mat(l, alpha)
    Xb = _y_rot_mat(l, beta)
    Xc = _y_rot_mat(l, gamma)
    J = Jd[l]
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
