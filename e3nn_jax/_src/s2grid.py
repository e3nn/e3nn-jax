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

import e3nn_jax as e3nn
from e3nn_jax._src.spherical_harmonics import _sh_alpha, _sh_beta
import jax.numpy as jnp
import numpy as np


def _quadrature_weights_soft(b):
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

    Args:
        lmax (int)
        res_beta (int): :math:`N`
        res_alpha (int): :math:`M`
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


def s2_grid(res_beta: int, res_alpha: int, *, quadrature: str):
    r"""grid on the sphere
    Args:
        res_beta (int): :math:`N`
        res_alpha (int): :math:`M`
        quadrature (str): "soft" or "gausslegendre"

    Returns:
        y (`numpy.ndarray`): array of shape ``(res_beta)``
        alphas: `numpy.ndarray`
            array of shape ``(res_alpha)``
    """

    if quadrature == "soft":
        i = np.arange(res_beta)
        betas = (i + 0.5) / res_beta * jnp.pi
        y = np.cos(betas)
    elif quadrature == "gausslegendre":
        y, _ = np.polynomial.legendre.leggauss(res_beta)
    else:
        raise Exception("quadrature needs to be 'soft' or 'gausslegendre'")

    i = jnp.arange(res_alpha)
    alphas = i / res_alpha * 2 * jnp.pi
    return jnp.array(y), alphas


def spherical_harmonics_s2_grid(lmax: int, res_beta: int, res_alpha: int, *, quadrature: str):
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
    """
    y, alphas = s2_grid(res_beta, res_alpha, quadrature=quadrature)
    sh_alpha = _sh_alpha(lmax, alphas)  # [..., 2 * l + 1]
    sh_y = _sh_beta(lmax, y)  # [..., (lmax + 1) * (lmax + 2) // 2]
    return y, alphas, sh_y, sh_alpha


def from_s2grid(
    x: jnp.ndarray,
    lmax: int,
    normalization="component",
    lmax_in=None,
    *,
    quadrature: str,
    fft: bool = True,
    p_val: int = 1,
    p_arg: int = -1,
):
    r"""Transform signal on the sphere into spherical tensors.

    The output has degree :math:`l` between 0 and lmax, and parity :math:`p = p_{val}p_{arg}^l`

    The inverse transformation of :func:`e3nn_jax.to_s2grid`

    Args:
        x (`jax.numpy.ndarray`): signal on the sphere of shape ``(..., beta, alpha)``
        lmax (int): maximum degree of the spherical tensor
        normalization ({'norm', 'component', 'integral'}): normalization of the spherical tensor
        lmax_in (int, optional): maximum degree of the input signal, only used for normalization purposes
        quadrature (str): "soft" or "gausslegendre"
        fft (bool): True if we use FFT, False if we use the naive implementation
        p_val (int): ``+1`` or ``-1``, the parity of the value of the input signal
        p_arg (int): ``+1`` or ``-1``, the parity of the argument of the input signal

    Returns:
        `e3nn_jax.IrrepsArray`: output spherical tensor, with coefficient array of shape ``(..., (lmax+1)^2)``
    """
    res_beta, res_alpha = x.shape[-2:]

    if lmax_in is None:
        lmax_in = lmax

    _, _, sh_y, sha = spherical_harmonics_s2_grid(lmax, res_beta, res_alpha, quadrature=quadrature)
    # sh_y: (res_beta, (l+1)(l+2)/2)

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

    # get quadrature weights
    if quadrature == "soft":
        assert res_beta % 2 == 0, "res_beta needs to be even for soft quadrature weights to be computed properly"
        qw = _quadrature_weights_soft(res_beta // 2) * res_beta**2  # [b]
    elif quadrature == "gausslegendre":
        _, qw = np.polynomial.legendre.leggauss(res_beta)
        qw /= 2
    else:
        raise Exception("quadrature needs to be 'soft' or 'gausslegendre'")

    # prepare beta integrand
    m = _expand_matrix(range(lmax + 1))  # [l, m, i]
    sh_y = _rollout_sh(sh_y, lmax)
    sh_y = jnp.einsum("lmj,bj,lmi,l,b->mbi", m, sh_y, m, n, qw)  # [m, b, i]

    # integrate over alpha
    if fft:
        int_a = rfft(x, lmax) / res_alpha  # [..., res_beta, 2*l+1]
    else:
        int_a = jnp.einsum("...ba,am->...bm", x, sha) / res_alpha  # [..., res_beta, 2*l+1]

    # integrate over beta
    int_b = jnp.einsum("mbi,...bm->...i", sh_y, int_a)  # [..., irreps]

    # convert to IrrepsArray
    irreps = [(1, (l, p_val * p_arg**l)) for l in range(lmax + 1)]
    return e3nn.IrrepsArray(irreps, int_b)


def to_s2grid(tensor: e3nn.IrrepsArray, res=None, normalization="component", *, quadrature: str, fft=True):
    r"""Transform spherical tensor into signal on the sphere

    The inverse transformation of :func:`e3nn_jax.from_s2grid`

    Args:
        tensor (`e3nn_jax.IrrepsArray`): spherical tensor, with coefficient array of shape ``(..., (lmax+1)^2)``
        res (tuple, optional): resolution of the grid on the sphere ``(beta, alpha)``
        normalization ({'norm', 'component', 'integral'}): normalization of the spherical tensor
        quadrature (str): "soft" or "gausslegendre"
        fft (bool): True if we use FFT, False if we use the naive implementation

    Returns:
        `jax.numpy.ndarray`: signal on the sphere of shape ``(..., beta, alpha)``
    """
    lmax = tensor.irreps.ls[-1]

    # check l values of irreps
    if not all(mul == 1 and ir.l == l for (mul, ir), l in zip(tensor.irreps, range(lmax + 1))):
        raise ValueError("multiplicities should be ones and irreps should range from l=0 to l=lmax")

    # check parities of irreps
    if not (
        {ir.p for mul, ir in tensor.irreps if ir.l % 2 == 0} in [{1}, {-1}, {}]
        and {ir.p for mul, ir in tensor.irreps if ir.l % 2 == 1} in [{1}, {-1}, {}]
    ):
        raise ValueError("irrep parities should be of the form (p_val * p_arg**l) for all l, where p_val and p_arg are ±1")

    if isinstance(res, int) or res is None:
        lmax, res_beta, res_alpha = _complete_lmax_res(lmax, res, None)
    else:
        lmax, res_beta, res_alpha = _complete_lmax_res(lmax, *res)

    _, _, sh_y, sha = spherical_harmonics_s2_grid(lmax, res_beta, res_alpha, quadrature=quadrature)

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
    sh_y = _rollout_sh(sh_y, lmax)
    sh_y = jnp.einsum("lmj,bj,lmi,l->mbi", m, sh_y, m, n)  # [m, b, i]

    # multiply spherical harmonics by their coefficients
    signal_b = jnp.einsum("mbi,...i->...bm", sh_y, tensor.array)  # [batch, beta, m]

    if fft:
        signal = irfft(signal_b, res_alpha) * res_alpha  # [..., res_beta, res_alpha]
    else:
        signal = jnp.einsum("...bm,am->...ba", signal_b, sha)  # [..., res_beta, res_alpha]

    return signal


def rfft(x: jnp.ndarray, l: int):
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
        x (`jax.numpy.ndarray`): array of shape ``(..., 2*l + 1)``
        res (int): output resolution, has to be an odd number
    Returns:
        `jax.numpy.ndarray`: positions on the sphere, array of shape ``(..., res)``
    """
    assert res % 2 == 1

    l = (x.shape[-1] - 1) // 2
    x_reshaped = jnp.concatenate(
        [x[..., l : l + 1], (x[..., l + 1 :] + jnp.flip(x[..., :l], -1) * -1j) / np.sqrt(2), jnp.zeros((*x.shape[:-1], l))],
        axis=-1,
    ).reshape((-1, x.shape[-1]))
    x_transformed = jnp.fft.irfft(x_reshaped, res)
    print((*x.shape[:-1], x_transformed.shape[-1]))
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
