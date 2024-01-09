from functools import partial

import jax
import jax.numpy as jnp


def legendre(
    lmax: int, x: jax.Array, phase: float, is_normalized: bool = False
) -> jax.Array:
    r"""Associated Legendre polynomials.

    en.wikipedia.org/wiki/Associated_Legendre_polynomials

    Args:
        lmax (int): maximum l value
        x (jax.Array): input array of shape ``(...)``
        phase (float): -1 or 1, multiplies by :math:`(-1)^m`
        is_normalized (bool): True if the associated Legendre functions are normalized.

    Returns:
        jax.Array: Associated Legendre polynomials ``P(l,m)``
        In an array of shape ``(lmax + 1, lmax + 1, ...)``
    """
    x = jnp.asarray(x)
    return _legendre(lmax, x, phase, is_normalized)


@partial(jax.jit, static_argnums=(0, 3))
def _legendre(lmax: int, x: jax.Array, phase: float, is_normalized: bool) -> jax.Array:
    p = jax.scipy.special.lpmn_values(
        lmax, lmax, x.flatten(), is_normalized
    )  # [m, l, x]
    p = (-phase) ** jnp.arange(lmax + 1)[:, None, None] * p
    p = jnp.transpose(p, (1, 0, 2))  # [l, m, x]
    p = jnp.reshape(p, (lmax + 1, lmax + 1) + x.shape)
    return p


def _sh_alpha(l: int, alpha: jax.Array) -> jax.Array:
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


def _sh_beta(lmax: int, cos_betas: jax.Array) -> jax.Array:
    r"""Beta dependence of spherical harmonics.

    Args:
        lmax: l value
        cos_betas: input array of shape ``(...)``

    Returns:
        Array of shape ``(..., l, m)``
    """
    sh_y = legendre(lmax, cos_betas, phase=1.0, is_normalized=True)  # [l, m, ...]
    sh_y = jnp.moveaxis(sh_y, 0, -1)  # [m, ..., l]
    sh_y = jnp.moveaxis(sh_y, 0, -1)  # [..., l, m]
    return sh_y


def legendre_spherical_harmonics(
    lmax: int, x: jax.Array, normalize: bool, normalization: str
) -> jax.Array:
    alpha = jnp.arctan2(x[..., 0], x[..., 2])
    sh_alpha = _sh_alpha(lmax, alpha)  # [..., 2 * l + 1]

    n = jnp.linalg.norm(x, axis=-1, keepdims=True)
    x = x / jnp.where(n > 0, n, 1.0)

    sh_y = _sh_beta(lmax, x[..., 1])  # [..., l, m]

    sh = jnp.zeros(x.shape[:-1] + ((lmax + 1) ** 2,), x.dtype)

    def f(l, sh):
        def g(m, sh):
            y = sh_y[..., l, jnp.abs(m)]
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
