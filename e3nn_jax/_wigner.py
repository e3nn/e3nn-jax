import os

import jax.numpy as jnp
import numpy as np

_Jd, _W3j_flat, _W3j_indices = np.load(os.path.join(os.path.dirname(__file__), 'constants.npy'), allow_pickle=True)
# _Jd is a list of tensors of shape (2l+1, 2l+1)
# _W3j_flat is a flatten version of W3j symbols
# _W3j_indices is a dict from (l1, l2, l3) -> slice(i, j) to index the flat tensor
# only l1 <= l2 <= l3 are stored


def _z_rot_mat(l, angle):
    r"""
    Create the matrix representation of a z-axis rotation by the given angle,
    in the irrep l of dimension 2 * l + 1, in the basis of real centered
    spherical harmonics (RC basis in rep_bases.py).

    Note: this function is easy to use, but inefficient: only the entries
    on the diagonal and anti-diagonal are non-zero, so explicitly constructing
    this matrix is unnecessary.
    """
    shape = angle.shape
    M = jnp.zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = jnp.arange(0, 2 * l + 1, 1)
    reversed_inds = jnp.arange(2 * l, -1, -1)
    frequencies = jnp.arange(l, -l - 1, -1.0)
    M[..., inds, reversed_inds] = jnp.sin(frequencies * angle[..., None])
    M[..., inds, inds] = jnp.cos(frequencies * angle[..., None])
    return M


def wigner_D(l, alpha, beta, gamma):
    r"""Wigner D matrix representation of :math:`SO(3)`.

    It satisfies the following properties:

    * :math:`D(\text{identity rotation}) = \text{identity matrix}`
    * :math:`D(R_1 \circ R_2) = D(R_1) \circ D(R_2)`
    * :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`
    * :math:`D(\text{rotation around Y axis})` has some property that allows us to use FFT in `ToS2Grid`

    Code of this function has beed copied from `lie_learn <https://github.com/AMLab-Amsterdam/lie_learn>`_ made by Taco Cohen.

    Parameters
    ----------
    l : int
        :math:`l`

    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\alpha` around Y axis, applied third.

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\beta` around X axis, applied second.

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\gamma` around Y axis, applied first.

    Returns
    -------
    `torch.Tensor`
        tensor :math:`D^l(\alpha, \beta, \gamma)` of shape :math:`(2l+1, 2l+1)`
    """
    if not l < len(_Jd):
        raise NotImplementedError(f'wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more')

    alpha, beta, gamma = jnp.broadcast_arrays(alpha, beta, gamma)
    Xa = _z_rot_mat(l, alpha)
    Xb = _z_rot_mat(l, beta)
    Xc = _z_rot_mat(l, gamma)
    J = _Jd[l]
    return Xa @ J @ Xb @ J @ Xc


def wigner_3j(l1, l2, l3, flat_src=_W3j_flat):
    r"""Wigner 3j symbols :math:`C_{lmn}`.

    It satisfies the following two properties:

        .. math::

            C_{lmn} = C_{ijk} D_{il}(g) D_{jm}(g) D_{kn}(g) \qquad \forall g \in SO(3)

        where :math:`D` are given by `wigner_D`.

        .. math::

            C_{ijk} C_{ijk} = 1

    Parameters
    ----------
    l1 : int
        :math:`l_1`

    l2 : int
        :math:`l_2`

    l3 : int
        :math:`l_3`

    dtype : torch.dtype or None
        ``dtype`` of the returned tensor. If ``None`` then set to ``torch.get_default_dtype()``.

    device : torch.device or None
        ``device`` of the returned tensor. If ``None`` then set to the default device of the current context.

    Returns
    -------
    `torch.Tensor`
        tensor :math:`C` of shape :math:`(2l_1+1, 2l_2+1, 2l_3+1)`
    """
    assert abs(l2 - l3) <= l1 <= l2 + l3

    try:
        if l1 <= l2 <= l3:
            out = flat_src[_W3j_indices[(l1, l2, l3)]].reshape(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)
        if l1 <= l3 <= l2:
            out = flat_src[_W3j_indices[(l1, l3, l2)]].reshape(2 * l1 + 1, 2 * l3 + 1, 2 * l2 + 1).transpose(0, 2, 1) * ((-1) ** (l1 + l2 + l3))
        if l2 <= l1 <= l3:
            out = flat_src[_W3j_indices[(l2, l1, l3)]].reshape(2 * l2 + 1, 2 * l1 + 1, 2 * l3 + 1).transpose(1, 0, 2) * ((-1) ** (l1 + l2 + l3))
        if l3 <= l2 <= l1:
            out = flat_src[_W3j_indices[(l3, l2, l1)]].reshape(2 * l3 + 1, 2 * l2 + 1, 2 * l1 + 1).transpose(2, 1, 0) * ((-1) ** (l1 + l2 + l3))
        if l2 <= l3 <= l1:
            out = flat_src[_W3j_indices[(l2, l3, l1)]].reshape(2 * l2 + 1, 2 * l3 + 1, 2 * l1 + 1).transpose(2, 0, 1)
        if l3 <= l1 <= l2:
            out = flat_src[_W3j_indices[(l3, l1, l2)]].reshape(2 * l3 + 1, 2 * l1 + 1, 2 * l2 + 1).transpose(1, 2, 0)
    except KeyError:
        raise NotImplementedError(f'Wigner 3j symbols maximum l implemented is {max(_W3j_indices.keys())[0]}, send us an email to ask for more')

    return out
