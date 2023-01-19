r"""Transformation between two representations of a signal on the sphere.

.. math:: f: S^2 \longrightarrow \mathbb{R}

is a signal on the sphere.

It can be decomposed into the basis of the spherical harmonics:

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

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import e3nn_jax as e3nn

from .spherical_harmonics import _sh_alpha, _sh_beta


def _quadrature_weights_soft(b: int) -> np.ndarray:
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
    return w


def s2grid(res_beta: int, res_alpha: int, *, quadrature: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""grid on the sphere
    Args:
        res_beta (int): :math:`N`
        res_alpha (int): :math:`M`
        quadrature (str): "soft" or "gausslegendre"

    Returns:
        y (`numpy.ndarray`): array of shape ``(res_beta)``
        alpha (`numpy.ndarray`): array of shape ``(res_alpha)``
        qw (`numpy.ndarray`): array of shape ``(res_beta)``, ``sum(qw) = 1``
    """

    if quadrature == "soft":
        i = np.arange(res_beta)
        betas = (i + 0.5) / res_beta * np.pi
        y = -np.cos(betas)  # minus sign is here to go from -1 to 1 in both quadratures

        assert res_beta % 2 == 0, "res_beta needs to be even for soft quadrature weights to be computed properly"
        qw = _quadrature_weights_soft(res_beta // 2) * res_beta**2
    elif quadrature == "gausslegendre":
        y, qw = np.polynomial.legendre.leggauss(res_beta)
        qw /= 2
    else:
        raise Exception("quadrature needs to be 'soft' or 'gausslegendre'")

    i = np.arange(res_alpha)
    alpha = i / res_alpha * 2 * np.pi
    return y, alpha, qw


def _spherical_harmonics_s2grid(lmax: int, res_beta: int, res_alpha: int, *, quadrature: str, dtype: np.dtype = np.float32):
    r"""spherical harmonics evaluated on the grid on the sphere
    .. math::
        f(x) = \sum_{l=0}^{l_{\mathit{max}}} F^l \cdot Y^l(x)
        f(\beta, \alpha) = \sum_{l=0}^{l_{\mathit{max}}} F^l \cdot S^l(\alpha) P^l(\cos(\beta))
    Args:
        lmax (int): :math:`l_{\mathit{max}}`
        res_beta (int): :math:`N`
        res_alpha (int): :math:`M`
        quadrature (str): "soft" or "gausslegendre"

    Returns:
        y (`jax.numpy.ndarray`): array of shape ``(res_beta)``
        alphas (`jax.numpy.ndarray`): array of shape ``(res_alpha)``
        sh_y (`jax.numpy.ndarray`): array of shape ``(res_beta, (lmax + 1)(lmax + 2)/2)``
        sh_alpha (`jax.numpy.ndarray`): array of shape ``(res_alpha, 2 * lmax + 1)``
        qw (`jax.numpy.ndarray`): array of shape ``(res_beta)``
    """
    y, alphas, qw = s2grid(res_beta, res_alpha, quadrature=quadrature)
    y, alphas, qw = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype), (y, alphas, qw))
    sh_alpha = _sh_alpha(lmax, alphas)  # [..., 2 * l + 1]
    sh_y = _sh_beta(lmax, y)  # [..., (lmax + 1) * (lmax + 2) // 2]
    return y, alphas, sh_y, sh_alpha, qw


def s2_irreps(lmax: int, p_val: int = 1, p_arg: int = -1) -> e3nn.Irreps:
    return e3nn.Irreps([(1, (l, p_val * p_arg**l)) for l in range(lmax + 1)])


def _check_parities(irreps: e3nn.Irreps):
    if not (
        {ir.p for mul, ir in irreps if ir.l % 2 == 0} in [{1}, {-1}, set()]
        and {ir.p for mul, ir in irreps if ir.l % 2 == 1} in [{1}, {-1}, set()]
    ):
        raise ValueError("irrep parities should be of the form (p_val * p_arg**l) for all l, where p_val and p_arg are ±1")


def from_s2grid(
    x: jnp.ndarray,
    irreps: e3nn.Irreps,
    *,
    normalization: str = "component",
    quadrature: str,
    lmax_in: Optional[int] = None,
    fft: bool = True,
):
    r"""Transform signal on the sphere into spherical harmonics coefficients.

    The output has degree :math:`l` between 0 and lmax, and parity :math:`p = p_{val}p_{arg}^l`

    The inverse transformation of :func:`e3nn_jax.to_s2grid`

    Args:
        x (`jax.numpy.ndarray`): signal on the sphere of shape ``(..., y/beta, alpha)``
        irreps (e3nn.Irreps): irreps of the coefficients
        normalization ({'norm', 'component', 'integral'}): normalization of the spherical harmonics basis
        lmax_in (int, optional): maximum degree of the input signal, only used for normalization purposes
        quadrature (str): "soft" or "gausslegendre"
        fft (bool): True if we use FFT, False if we use the naive implementation

    Returns:
        `e3nn_jax.IrrepsArray`: coefficient array of shape ``(..., (lmax+1)^2)``
    """
    res_beta, res_alpha = x.shape[-2:]

    irreps = e3nn.Irreps(irreps)

    if not all(mul == 1 for mul, _ in irreps.regroup()):
        raise ValueError("multiplicities should be ones")

    _check_parities(irreps)

    lmax = max(irreps.ls)

    if lmax_in is None:
        lmax_in = lmax

    _, _, sh_y, sha, qw = _spherical_harmonics_s2grid(lmax, res_beta, res_alpha, quadrature=quadrature, dtype=x.dtype)
    # sh_y: (res_beta, (l+1)(l+2)/2)

    # normalize such that it is the inverse of ToS2Grid
    n = None
    # lmax_in = max frequency in input; lmax = max freq in output
    if normalization == "component":
        n = jnp.sqrt(4 * jnp.pi) * jnp.asarray([jnp.sqrt(2 * l + 1) for l in range(lmax + 1)], x.dtype) * jnp.sqrt(lmax_in + 1)
    elif normalization == "norm":
        n = jnp.sqrt(4 * jnp.pi) * jnp.ones(lmax + 1, x.dtype) * jnp.sqrt(lmax_in + 1)
    elif normalization == "integral":
        n = 4 * jnp.pi * jnp.ones(lmax + 1, x.dtype)
    else:
        raise Exception("normalization needs to be 'norm', 'component' or 'integral'")

    # prepare beta integrand
    m_in = jnp.asarray(_expand_matrix(range(lmax + 1)), x.dtype)  # [l, m, j]
    m_out = jnp.asarray(_expand_matrix(irreps.ls), x.dtype)  # [l, m, i]
    sh_y = _rollout_sh(sh_y, lmax)
    sh_y = jnp.einsum("lmj,bj,lmi,l,b->mbi", m_in, sh_y, m_out, n, qw)  # [m, b, i]

    # integrate over alpha
    if fft:
        int_a = rfft(x, lmax) / res_alpha  # [..., res_beta, 2*l+1]
    else:
        int_a = jnp.einsum("...ba,am->...bm", x, sha) / res_alpha  # [..., res_beta, 2*l+1]

    # integrate over beta
    int_b = jnp.einsum("mbi,...bm->...i", sh_y, int_a)  # [..., irreps]

    # convert to IrrepsArray
    return e3nn.IrrepsArray(irreps, int_b)


def to_s2grid(
    coeffs: e3nn.IrrepsArray,
    res_beta: int,
    res_alpha: int,
    *,
    normalization: str = "component",
    quadrature: str,
    fft: bool = True,
):
    r"""Sample a signal on the sphere given by the coefficient in the spherical harmonics basis.

    The inverse transformation of :func:`e3nn_jax.from_s2grid`

    Args:
        coeffs (`e3nn_jax.IrrepsArray`): coefficient array
        res_beta (int): number of points on the sphere in the :math:`\theta` direction
        res_alpha (int): number of points on the sphere in the :math:`\phi` direction
        normalization ({'norm', 'component', 'integral'}): normalization of the basis
        quadrature (str): "soft" or "gausslegendre"
        fft (bool): True if we use FFT, False if we use the naive implementation

    Returns:
        `jax.numpy.ndarray`: signal on the sphere of shape ``(..., y/beta, alpha)``
    """
    coeffs = coeffs.regroup()
    lmax = coeffs.irreps.ls[-1]

    if not all(mul == 1 for mul, _ in coeffs.irreps):
        raise ValueError("multiplicities should be ones")

    _check_parities(coeffs.irreps)

    _, _, sh_y, sha, _ = _spherical_harmonics_s2grid(lmax, res_beta, res_alpha, quadrature=quadrature, dtype=coeffs.dtype)

    n = None
    if normalization == "component":
        # normalize such that all l has the same variance on the sphere
        # given that all component has mean 0 and variance 1
        n = (
            jnp.sqrt(4 * jnp.pi)
            * jnp.asarray([1 / jnp.sqrt(2 * l + 1) for l in range(lmax + 1)], coeffs.dtype)
            / jnp.sqrt(lmax + 1)
        )
    elif normalization == "norm":
        # normalize such that all l has the same variance on the sphere
        # given that all component has mean 0 and variance 1/(2L+1)
        n = jnp.sqrt(4 * jnp.pi) * jnp.ones(lmax + 1, coeffs.dtype) / jnp.sqrt(lmax + 1)
    elif normalization == "integral":
        n = jnp.ones(lmax + 1, coeffs.dtype)
    else:
        raise Exception("normalization needs to be 'norm', 'component' or 'integral'")

    m_in = jnp.asarray(_expand_matrix(range(lmax + 1)), coeffs.dtype)  # [l, m, j]
    m_out = jnp.asarray(_expand_matrix(coeffs.irreps.ls), coeffs.dtype)  # [l, m, i]
    # put beta component in summable form
    sh_y = _rollout_sh(sh_y, lmax)
    sh_y = jnp.einsum("lmj,bj,lmi,l->mbi", m_in, sh_y, m_out, n)  # [m, b, i]

    # multiply spherical harmonics by their coefficients
    signal_b = jnp.einsum("mbi,...i->...bm", sh_y, coeffs.array)  # [batch, beta, m]

    if fft:
        if res_alpha % 2 == 0:
            raise ValueError("res_alpha must be odd for fft")

        signal = irfft(signal_b, res_alpha) * res_alpha  # [..., res_beta, res_alpha]
    else:
        signal = jnp.einsum("...bm,am->...ba", signal_b, sha)  # [..., res_beta, res_alpha]

    return signal


def rfft(x: jnp.ndarray, l: int) -> jnp.ndarray:
    r"""Real fourier transform
    Args:
        x (`jax.numpy.ndarray`): input array of shape ``(..., res_beta, res_alpha)``
        l (int): value of `l` for which the transform is being run
    Returns:
        `jax.numpy.ndarray`: transformed values - array of shape ``(..., res_beta, 2*l+1)``
    """
    x_reshaped = x.reshape((-1, x.shape[-1]))
    x_transformed_c = jnp.fft.rfft(x_reshaped)  # (..., 2*l+1)
    x_transformed = jnp.concatenate(
        [
            jnp.flip(jnp.imag(x_transformed_c[..., 1 : l + 1]), -1) * -jnp.sqrt(2),
            jnp.real(x_transformed_c[..., :1]),
            jnp.real(x_transformed_c[..., 1 : l + 1]) * jnp.sqrt(2),
        ],
        axis=-1,
    )
    return x_transformed.reshape((*x.shape[:-1], 2 * l + 1))


def irfft(x: jnp.ndarray, res: int) -> jnp.ndarray:
    r"""Inverse of the real fourier transform
    Args:
        x (`jax.numpy.ndarray`): array of shape ``(..., 2*l + 1)``
        res (int): output resolution, has to be an odd number
    Returns:
        `jax.numpy.ndarray`: positions on the sphere, array of shape ``(..., res)``
    """
    assert res % 2 == 1

    l = (x.shape[-1] - 1) // 2
    x_reshaped = jnp.concatenate(
        [
            x[..., l : l + 1],
            (x[..., l + 1 :] + jnp.flip(x[..., :l], -1) * -1j) / jnp.sqrt(2),
            jnp.zeros((*x.shape[:-1], l), x.dtype),
        ],
        axis=-1,
    ).reshape((-1, x.shape[-1]))
    x_transformed = jnp.fft.irfft(x_reshaped, res)
    return x_transformed.reshape((*x.shape[:-1], x_transformed.shape[-1]))


def _expand_matrix(ls: List[int]) -> np.ndarray:
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
        array of shape ``[l, m, l * m]``
    """
    lmax = max(ls)
    m = np.zeros((lmax + 1, 2 * lmax + 1, sum(2 * l + 1 for l in ls)), np.float64)
    i = 0
    for l in ls:
        m[l, lmax - l : lmax + l + 1, i : i + 2 * l + 1] = np.eye(2 * l + 1, dtype=np.float64)
        i += 2 * l + 1
    return m


def _rollout_sh(m: jnp.ndarray, lmax: int) -> jnp.ndarray:
    """
    Expand spherical harmonic representation.
    Args:
        m: jnp.ndarray of shape (..., (lmax+1)*(lmax+2)/2)
    Returns:
        jnp.ndarray of shape (..., (lmax+1)**2)
    """
    assert m.shape[-1] == (lmax + 1) * (lmax + 2) // 2
    m_full = jnp.zeros((*m.shape[:-1], (lmax + 1) ** 2), dtype=m.dtype)
    for l in range(lmax + 1):
        i_mid = l**2 + l
        for i in range(l + 1):
            m_full = m_full.at[..., i_mid + i].set(m[..., l * (l + 1) // 2 + i])
            m_full = m_full.at[..., i_mid - i].set(m[..., l * (l + 1) // 2 + i])
    return m_full


def s2grid_vectors(y: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    r"""Calculate the points on the sphere.

    Args:
        y: array with y values, shape ``(res_beta)``
        alpha: array with alpha values, shape ``(res_alpha)``

    Returns:
        r: array of vectors, shape ``(res_beta, res_alpha, 3)``
    """
    assert y.ndim == 1
    assert alpha.ndim == 1
    return np.stack(
        [
            np.sqrt(1.0 - y[:, None] ** 2) * np.sin(alpha),
            y[:, None] * np.ones_like(alpha),
            np.sqrt(1.0 - y[:, None] ** 2) * np.cos(alpha),
        ],
        axis=2,
    )


def pad_to_plot_on_s2grid(
    y: np.ndarray,  # [beta_res]
    alpha: np.ndarray,  # [alpha_res]
    signal: np.ndarray,  # [beta_res, alpha_res]
    *,
    translation: Optional[np.ndarray] = None,
    scale_radius_by_amplitude: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Postprocess the borders of a given signal to allow the plot it with plotly.

    Args:
        y: array with y values, shape ``(res_beta)``
        alpha: array with alpha values, shape ``(res_alpha)``
        signal: array with the signal on the sphere, shape ``(res_beta, res_alpha)``
        translation (optional): translation vector
        scale_radius_by_amplitude (bool): to rescale the output vectors with the amplitude of the signal

    Returns:
        r (np.ndarray): vectors on the sphere, shape ``(res_beta + 2, res_alpha + 1, 3)``
        f (np.ndarray): padded signal, shape ``(res_beta + 2, res_alpha + 1)``
    """
    assert signal.shape == (len(y), len(alpha))

    f = np.array(signal)

    # y: [-1, 1]
    one = np.ones_like(y, shape=(1,))
    ones = np.ones_like(f, shape=(1, len(alpha)))
    y = np.concatenate([-one, y, one])  # [res_beta + 2]
    f = np.concatenate([np.mean(f[0]) * ones, f, np.mean(f[-1]) * ones], axis=0)  # [res_beta + 2, res_alpha]

    # alpha: [0, 2pi]
    alpha = np.concatenate([alpha, alpha[:1]])  # [res_alpha + 1]
    f = np.concatenate([f, f[:, :1]], axis=1)  # [res_beta + 2, res_alpha + 1]

    r = s2grid_vectors(y, alpha)  # [res_beta + 2, res_alpha + 1, 3]

    if scale_radius_by_amplitude:
        r *= np.abs(f)[:, :, None]

    if translation is not None:
        r = r + translation

    return r, f
