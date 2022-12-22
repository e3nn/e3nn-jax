import jax
import numpy as np
import pytest
import jax.numpy as jnp

import e3nn_jax as e3nn
from e3nn_jax._src.s2grid import irfft, rfft, _spherical_harmonics_s2grid
from e3nn_jax.util import assert_output_dtype


@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
@pytest.mark.parametrize("fft_to", [False, True])
@pytest.mark.parametrize("fft_from", [False, True])
def test_s2grid_transforms(keys, quadrature, fft_to, fft_from):
    assert quadrature in ["soft", "gausslegendre"], "quadrature must be 'soft' or 'gausslegendre"
    res_alpha = 51
    res_beta = 30
    lmax = 10
    p_val = 1
    p_arg = -1

    c = jax.random.uniform(keys[0], shape=(1, (lmax + 1) ** 2))
    irreps = e3nn.Irreps([(1, (l, p_val * p_arg**l)) for l in range(lmax + 1)])
    irreps_in = e3nn.IrrepsArray(irreps, c)

    res = e3nn.to_s2grid(irreps_in, res_beta, res_alpha, quadrature=quadrature, fft=fft_to)
    irreps_out = e3nn.from_s2grid(res, lmax, quadrature=quadrature, fft=fft_from)
    np.testing.assert_allclose(c, irreps_out.array, rtol=1e-5, atol=1e-5)
    assert irreps_in.irreps == irreps_out.irreps


def test_fft(keys):
    res_alpha = 11  # 2l+1
    l = 5
    x = jax.random.uniform(keys[0], shape=(8, res_alpha))
    x_t = rfft(x, l)
    x_p = irfft(x_t, res_alpha)
    np.testing.assert_allclose(x, x_p, rtol=1e-5)


@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
def test_s2grid_vectors(quadrature):
    y, alpha, sh_y, sh_alpha, _ = _spherical_harmonics_s2grid(lmax=1, res_beta=4, res_alpha=5, quadrature=quadrature)
    r = e3nn.s2grid_vectors(y, alpha)

    sh_y = np.stack([sh_y[:, 2], sh_y[:, 1], sh_y[:, 2]], axis=1)  # for l=1
    sh = sh_y[:, None, :] * sh_alpha
    sh = sh / np.linalg.norm(sh, axis=2, keepdims=True)

    np.testing.assert_allclose(sh, r, atol=1e-7)


@pytest.mark.parametrize("normalization", ["component", "norm"])
@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
@pytest.mark.parametrize("fft", [False, True])
def test_to_s2grid_dtype(normalization, quadrature, fft):
    jax.config.update("jax_enable_x64", True)

    assert_output_dtype(
        lambda x: e3nn.to_s2grid(x, 4, 5, normalization=normalization, quadrature=quadrature, fft=fft),
        e3nn.IrrepsArray("0e", jnp.array([1.0])),
    )


@pytest.mark.parametrize("normalization", ["component", "norm"])
@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
@pytest.mark.parametrize("fft", [False, True])
def test_from_s2grid_dtype(normalization, quadrature, fft):
    jax.config.update("jax_enable_x64", True)

    assert_output_dtype(
        lambda x: e3nn.from_s2grid(x, 4, normalization=normalization, quadrature=quadrature, fft=fft),
        jnp.ones((10, 11)),
    )


def test_fft_dtype():
    jax.config.update("jax_enable_x64", True)

    assert_output_dtype(lambda x: rfft(x, 4), jnp.ones((10, 11)))
    assert_output_dtype(lambda x: irfft(x, 11), jnp.ones((10, 11)))
