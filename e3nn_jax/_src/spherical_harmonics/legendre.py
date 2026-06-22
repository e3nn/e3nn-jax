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


def _gen_recurrence_mask(l_max: int, is_normalized: bool, dtype):
    """Mask used by the off-diagonal recurrence relation."""
    m_mat, l_mat = jnp.meshgrid(
        jnp.arange(l_max + 1, dtype=dtype),
        jnp.arange(l_max + 1, dtype=dtype),
        indexing="ij",
    )

    # 1. Define strictly valid regions for d0 (l >= m + 1) and d1 (l >= m + 2)
    # These exactly match the bounds of jnp.triu_indices used later.
    valid_d0 = l_mat >= m_mat + 1
    valid_d1 = l_mat >= m_mat + 2

    if is_normalized:
        c0 = l_mat * l_mat
        c1 = m_mat * m_mat
        c2 = 2.0 * l_mat
        c3 = (l_mat - 1.0) * (l_mat - 1.0)

        # 2. Patch denominators: Use 1.0 where invalid to prevent division by zero
        denom_d0 = jnp.where(valid_d0, c0 - c1, 1.0)
        denom_d1 = jnp.where(valid_d1, (c2 - 3.0) * (c0 - c1), 1.0)

        # 3. Patch radicands: Ensure inputs to sqrt are strictly positive (using 1.0)
        # grad() of jnp.sqrt(0.0) is NaN, so we must force invalid regions to 1.0, not 0.0.
        radicand_d0 = jnp.where(valid_d0, (4.0 * c0 - 1.0) / denom_d0, 1.0)
        radicand_d1 = jnp.where(valid_d1, ((c2 + 1.0) * (c3 - c1)) / denom_d1, 1.0)

        d0 = jnp.sqrt(radicand_d0)
        d1 = jnp.sqrt(radicand_d1)
    else:
        # 2b. Patch denominators for unnormalized branch
        denom_d0 = jnp.where(valid_d0, l_mat - m_mat, 1.0)
        denom_d1 = jnp.where(valid_d1, l_mat - m_mat, 1.0)

        d0 = jnp.where(valid_d0, (2.0 * l_mat - 1.0) / denom_d0, 0.0)
        d1 = jnp.where(valid_d1, (l_mat + m_mat - 1.0) / denom_d1, 0.0)

    d0_mask_indices = jnp.triu_indices(l_max + 1, 1)
    d1_mask_indices = jnp.triu_indices(l_max + 1, 2)
    d_zeros = jnp.zeros((l_max + 1, l_max + 1), dtype=dtype)

    # Because our 'valid' masks align perfectly with triu_indices,
    # the dummy 1.0 values are safely ignored here.
    d0_mask = d_zeros.at[d0_mask_indices].set(d0[d0_mask_indices])
    d1_mask = d_zeros.at[d1_mask_indices].set(d1[d1_mask_indices])

    i, j, k = jnp.ogrid[: l_max + 1, : l_max + 1, : l_max + 1]
    mask = (i + j - k == 0).astype(dtype)

    return (
        jnp.einsum("jk,ijk->ijk", d0_mask, mask),
        jnp.einsum("jk,ijk->ijk", d1_mask, mask),
    )


@partial(jax.jit, static_argnums=(0, 2))
def _assoc_legendre(l_max: int, x: jax.Array, is_normalized: bool) -> jax.Array:
    """Associated Legendre functions P(m, l, x), shape (l_max+1, l_max+1, len(x)).

    A drop-in replacement for the deprecated jax.scipy.special.lpmn_values
    (called as lpmn_values(l_max, l_max, x, is_normalized)).
    """
    p = jnp.zeros((l_max + 1, l_max + 1, x.shape[0]), dtype=x.dtype)

    a_idx = jnp.arange(1, l_max + 1, dtype=x.dtype)
    b_idx = jnp.arange(l_max, dtype=x.dtype)
    if is_normalized:
        initial_value = 0.5 / jnp.sqrt(jnp.pi)
        f_a = jnp.cumprod(-1 * jnp.sqrt(1.0 + 0.5 / a_idx))
        f_b = jnp.sqrt(2.0 * b_idx + 3.0)
    else:
        initial_value = 1.0
        f_a = jnp.cumprod(1.0 - 2.0 * a_idx)
        f_b = 2.0 * b_idx + 1.0

    p = p.at[(0, 0)].set(initial_value)

    y = jnp.cumprod(
        jnp.broadcast_to(jnp.sqrt(1.0 - x * x), (l_max, x.shape[0])), axis=0
    )
    p_diag = initial_value * jnp.einsum("i,ij->ij", f_a, y)
    diag_indices = jnp.diag_indices(l_max + 1)
    p = p.at[(diag_indices[0][1:], diag_indices[1][1:])].set(p_diag)

    p_offdiag = jnp.einsum(
        "ij,ij->ij", jnp.einsum("i,j->ij", f_b, x), p[jnp.diag_indices(l_max)]
    )
    offdiag_indices = (diag_indices[0][:l_max], diag_indices[1][:l_max] + 1)
    p = p.at[offdiag_indices].set(p_offdiag)

    d0_mask_3d, d1_mask_3d = _gen_recurrence_mask(
        l_max, is_normalized=is_normalized, dtype=x.dtype
    )

    def body_fun(i, p_val):
        coeff_0, coeff_1 = d0_mask_3d[i], d1_mask_3d[i]
        h = jnp.einsum(
            "ij,ijk->ijk",
            coeff_0,
            jnp.einsum("ijk,k->ijk", jnp.roll(p_val, shift=1, axis=1), x),
        ) - jnp.einsum("ij,ijk->ijk", coeff_1, jnp.roll(p_val, shift=2, axis=1))
        return p_val + h

    p = p.astype(jnp.result_type(p, x, d0_mask_3d))
    if l_max > 1:
        p = jax.lax.fori_loop(2, l_max + 1, body_fun, p)

    return p


@partial(jax.jit, static_argnums=(0, 3))
def _legendre(lmax: int, x: jax.Array, phase: float, is_normalized: bool) -> jax.Array:
    p = _assoc_legendre(lmax, x.flatten(), is_normalized)  # [m, l, x]
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
