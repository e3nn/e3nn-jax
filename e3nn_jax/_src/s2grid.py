r"""Transformation between two representations of a signal on the sphere.

.. math:: f: S^2 \longrightarrow \mathbb{R}

is a signal on the sphere.

One representation that we like to call "spherical tensor" is

.. math:: f(x) = \sum_{l=0}^{l_{\mathit{max}}} F^l \cdot Y^l(x)

it is made of :math:`(l_{\mathit{max}} + 1)^2` real numbers represented in the above formula by the familly of vectors
:math:`F^l \in \mathbb{R}^{2l+1}`.

Another representation is the discretization around the sphere. For this representation we chose a particular grid of size
:math:`(N, M)`

.. math::

    x_{ij} &= (\sin(\beta_i) \sin(\alpha_j), \cos(\beta_i), \sin(\beta_i) \cos(\alpha_j))

    \beta_i &= \pi (i + 0.5) / N

    \alpha_j &= 2 \pi j / M

In the code, :math:`N` is called ``res_beta`` and :math:`M` is ``res_alpha``.

The discrete representation is therefore

.. math:: \{ h_{ij} = f(x_{ij}) \}_{ij}
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Sequence, Tuple, Union
from e3nn_jax._src.rotation import angles_to_xyz
from e3nn_jax._src.spherical_harmonics import _legendre_spherical_harmonics, _sh_alpha, _sh_beta


def _quadrature_weights(b):
    r"""
    function copied from ``lie_learn.spaces.S3``
    Compute quadrature weights for the grid used by Kostelec & Rockmore [1, 2].

    This grid is:
    alpha = 2 pi i / 2b
    beta = pi (2 j + 1) / 4b
    gamma = 2 pi k / 2b
    where 0 <= i, j, k < 2b are indices

    Args:
        b: bandwidth

    Returns:
        array of shape (2*b) containing quadrature weights
    """
    k = np.arange(b)
    w = np.array(
        [
            (
                (2.0 / b)
                * np.sin(np.pi * (2.0 * j + 1.0) / (4.0 * b))
                * ((1.0 / (2 * k + 1)) * np.sin((2 * j + 1) * (2 * k + 1) * np.pi / (4.0 * b))).sum()
            )
            for j in np.arange(2 * b)
        ],
    )

    w /= 2.0 * ((2 * b) ** 2)
    return jnp.asarray(w)


def _complete_lmax_res(lmax, res_beta, res_alpha):
    """
    determine alpha/beta resolutions if they're not specified

    try to use FFT
    i.e. 2 * lmax + 1 == res_alpha
    """
    if res_beta is None:
        if lmax is not None:
            res_beta = 2 * (lmax + 1)  # minimum req. to go on sphere and back
        elif res_alpha is not None:
            res_beta = 2 * ((res_alpha + 1) // 2)
        else:
            raise ValueError("All the entries are None")

    if res_alpha is None:
        if lmax is not None:
            if res_beta is not None:
                res_alpha = max(2 * lmax + 1, res_beta - 1)
            else:
                res_alpha = 2 * lmax + 1  # minimum req. to go on sphere and back
        elif res_beta is not None:
            res_alpha = res_beta - 1

    if lmax is None:
        lmax = min(res_beta // 2 - 1, (res_alpha - 1) // 2)  # maximum possible to go on sphere and back
        # see tests -------------------------------^

    assert res_beta % 2 == 0
    assert lmax + 1 <= res_beta // 2

    return lmax, res_beta, res_alpha


def s2_grid(res_beta: int, res_alpha: int):
    r"""grid on the sphere
    Args:
        res_beta: int
            :math:`N`
        res_alpha: int
            :math:`M`

    Returns:
        betas: `jnp.ndarray`
            array of shape ``(res_beta)``
        alphas: `jnp.ndarray`
            array of shape ``(res_alpha)``
    """

    i = jnp.arange(res_beta)
    betas = (i + 0.5) / res_beta * jnp.pi

    i = jnp.arange(res_alpha)
    alphas = i / res_alpha * 2 * jnp.pi
    return betas, alphas


def spherical_harmonics_s2_grid(lmax: int, res_beta: int, res_alpha: int):
    r"""spherical harmonics evaluated on the grid on the sphere
    .. math::
        f(x) = \sum_{l=0}^{l_{\mathit{max}}} F^l \cdot Y^l(x)
        f(\beta, \alpha) = \sum_{l=0}^{l_{\mathit{max}}} F^l \cdot S^l(\alpha) P^l(\cos(\beta))
    Args:
        lmax: int
            :math:`l_{\mathit{max}}`
        res_beta: int
            :math:`N`
        res_alpha: int
            :math:`M`

    Returns:
        betas: `jnp.ndarray`
            array of shape ``(res_beta)``
        alphas: `jnp.ndarray`
            array of shape ``(res_alpha)``
        sh_beta: `jnp.ndarray`
            array of shape ``(res_beta, (lmax + 1)(lmax + 2)/2)``
        sh_alpha: `jnp.ndarray`
            array of shape ``(res_alpha, 2 * lmax + 1)``
    """
    betas, alphas = s2_grid(res_beta, res_alpha)
    sh_alpha = _sh_alpha(lmax, alphas)  # [..., 2 * l + 1]
    sh_beta = _sh_beta(lmax, jnp.cos(betas))  # [..., (lmax + 1) * (lmax + 2) // 2]
    return betas, alphas, sh_beta, sh_alpha


def from_s2grid(x: jnp.ndarray, lmax: int, normalization="component", lmax_in=None):
    r"""Transform signal on the sphere into spherical tensor
    The inverse transformation of `ToS2Grid`
    Args:
        x: `jnp.ndarray`
            signal of shape ``(..., beta, alpha)``
        res: int, tuple of int
            resolution in ``beta`` and in ``alpha``
        lmax: int
        normalization: {'norm', 'component', 'integral'}
        lmax_in: int, optional
    Returns:
        `jnp.ndarray`
            array of coefficients, of shape ``(..., (l+1)^2)``
    """
    res_beta, res_alpha = x.shape[-2:]

    if lmax_in is None:
        lmax_in = lmax  # what is lmax_in?

    _, _, shb, sha = spherical_harmonics_s2_grid(lmax, res_beta, res_alpha)
    # sh_alpha: (res_alpha, 2*l+1); sh_beta: (res_beta, (l+1)(l+2)/2)

    # normalize such that it is the inverse of ToS2Grid
    n = None
    # lmax_in = max frequency in input; lmax = max freq in output
    if normalization == "component":
        n = jnp.sqrt(4 * jnp.pi) * jnp.asarray([jnp.sqrt(2 * l + 1) for l in range(lmax + 1)]) * jnp.sqrt(lmax_in + 1)
    elif normalization == "norm":
        n = jnp.sqrt(4 * jnp.pi) * jnp.ones(lmax + 1) * jnp.sqrt(lmax_in + 1)
    elif normalization == "integral":
        n = 4 * jnp.pi * jnp.ones(lmax + 1)
    else:
        raise Exception("normalization needs to be 'norm', 'component' or 'integral'")

    m = _expand_matrix(range(lmax + 1))  # [l, m, i]
    shb = _rollout_sh(shb, lmax)

    assert res_beta % 2 == 0, "res_beta needs to be even for quadrature weights to be computed properly"
    qw = _quadrature_weights(res_beta // 2) * res_beta**2 / res_alpha  # [b]
    # beta integrand
    shb = jnp.einsum("lmj,bj,lmi,l,b->mbi", m, shb, m, n, qw)  # [m, b, i]

    size = x.shape[:-2]
    x = x.reshape(-1, res_beta, res_alpha)

    # integrate over alpha
    int_a = rfft(x, lmax)  # [..., res_beta, 2*l+1]
    # integrate over beta
    int_b = jnp.einsum("mbi,zbm->zi", shb, int_a)
    return int_b.reshape(*size, int_b.shape[1])


def to_s2grid(coeffs: jnp.ndarray, res=None, normalization="component"):
    r"""Transform spherical tensor into signal on the sphere
    The inverse transformation of `FromS2Grid`

    Args:
        coeffs: `jnp.ndarray`
            coefficients of the spherical harmonics - array of shape ``(..., (lmax+1)**2)``
        lmax: int
        res: int, tuple of int
            resolution in ``beta`` and in ``alpha``
        normalization : {'norm', 'component', 'integral'}
    Returns:
        `jnp.array`
            signal of shape ``(..., beta, alpha)``
    """
    lmax = int(np.sqrt(coeffs.shape[-1])) - 1

    if isinstance(res, int) or res is None:
        lmax, res_beta, res_alpha = _complete_lmax_res(lmax, res, None)
    else:
        lmax, res_beta, res_alpha = _complete_lmax_res(lmax, *res)

    _, _, shb, sha = spherical_harmonics_s2_grid(lmax, res_beta, res_alpha)

    n = None
    if normalization == "component":
        # normalize such that all l has the same variance on the sphere
        # given that all component has mean 0 and variance 1
        n = jnp.sqrt(4 * jnp.pi) * jnp.asarray([1 / jnp.sqrt(2 * l + 1) for l in range(lmax + 1)]) / jnp.sqrt(lmax + 1)
    elif normalization == "norm":
        # normalize such that all l has the same variance on the sphere
        # given that all component has mean 0 and variance 1/(2L+1)
        n = jnp.sqrt(4 * jnp.pi) * jnp.ones(lmax + 1) / jnp.sqrt(lmax + 1)
    elif normalization == "integral":
        n = jnp.ones(lmax + 1)
    else:
        raise Exception("normalization needs to be 'norm', 'component' or 'integral'")

    m = _expand_matrix(range(lmax + 1))  # [l, m, i]
    # put beta component in summable form
    shb = _rollout_sh(shb, lmax)
    shb = jnp.einsum("lmj,bj,lmi,l->mbi", m, shb, m, n)  # [m, b, i]

    size = coeffs.shape[:-1]
    coeffs = coeffs.reshape(-1, coeffs.shape[-1])

    # multiply spherical harmonics by their coefficients
    signal_b = jnp.einsum("mbi,zi->zbm", shb, coeffs)  # [batch, beta, m]

    signal = irfft(signal_b, res_alpha) * res_alpha
    return signal.reshape(*size, *signal.shape[1:])


def rfft(x: jnp.ndarray, l: int):
    r"""Real fourier transform
    Args:
        x: `jnp.ndarray`
            input array of shape ``(..., res_beta, res_alpha)``
        l: int
            value of `l` for which the transform is being run
    Returns:
        `jnp.ndarray`
            transformed values - array of shape ``(..., res_beta, 2*l+1)``
    """
    x_reshaped = x.reshape((-1, x.shape[-1]))
    x_transformed_c = jnp.fft.rfft(x_reshaped)  # (..., 2*l+1)
    x_transformed = jnp.concatenate(
        [
            jnp.flip(jnp.imag(x_transformed_c[..., 1 : l + 1]), -1) * -np.sqrt(2),
            jnp.real(x_transformed_c[..., :1]),
            jnp.real(x_transformed_c[..., 1 : l + 1]) * np.sqrt(2),
        ],
        axis=-1,
    )
    return x_transformed.reshape((*x.shape[:-1], 2 * l + 1))


def irfft(x: jnp.ndarray, res: int):
    r"""Inverse of the real fourier transform
    Args:
        x: `jnp.ndarray`
            array of shape ``(..., 2*l + 1)``
        res: int
            output resolution, has to be an odd number
    Returns:
        `jnp.ndarray`
            positions on the sphere, array of shape ``(..., res, 3)``
    """
    assert res % 2 == 1

    l = (x.shape[-1] - 1) // 2
    x_reshaped = jnp.concatenate(
        [x[..., l : l + 1], (x[..., l + 1 :] + jnp.flip(x[..., :l], -1) * -1j) / np.sqrt(2), jnp.zeros((*x.shape[:-1], l))],
        axis=-1,
    ).reshape((-1, x.shape[-1]))
    x_transformed = jnp.fft.irfft(x_reshaped, res)
    return x_transformed.reshape((*x.shape[:-1], x_transformed.shape[-1]))


def _expand_matrix(ls):
    """
    conversion matrix between a flatten vector (L, m) like that
    (0, 0) (1, -1) (1, 0) (1, 1) (2, -2) (2, -1) (2, 0) (2, 1) (2, 2)
    and a bidimensional matrix representation like that
                    (0, 0)
            (1, -1) (1, 0) (1, 1)
    (2, -2) (2, -1) (2, 0) (2, 1) (2, 2)
    Args:
        ls: list of l values
    Returns:
        tensor [l, m, l * m]
    """
    lmax = max(ls)
    m = np.zeros((len(ls), 2 * lmax + 1, sum(2 * l + 1 for l in ls)))
    i = 0
    for j, l in enumerate(ls):
        m[j, lmax - l : lmax + l + 1, i : i + 2 * l + 1] = np.eye(2 * l + 1)
        i += 2 * l + 1
    return jnp.asarray(m)


def _rollout_sh(m, lmax):
    """
    Expand spherical harmonic representation.
    Args:
        m: jnp.ndarray of shape (..., (lmax+1)*(lmax+2)/2)
    Returns:
        jnp.ndarray of shape (..., (lmax+1)**2)
    """
    assert m.shape[-1] == (lmax + 1) * (lmax + 2) // 2
    m_full = np.zeros((*m.shape[:-1], (lmax + 1) ** 2))
    for l in range(lmax + 1):
        i_mid = l**2 + l
        for i in range(l + 1):
            m_full[..., i_mid + i] = m[..., l * (l + 1) // 2 + i]
            m_full[..., i_mid - i] = m[..., l * (l + 1) // 2 + i]
    return m_full
