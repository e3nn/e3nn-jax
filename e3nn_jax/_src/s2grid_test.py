import jax
import numpy as np
import pytest
import jax.numpy as jnp

import e3nn_jax as e3nn
from e3nn_jax._src.s2grid import irfft, rfft, _spherical_harmonics_s2grid, SphericalSignal
from e3nn_jax.util import assert_output_dtype_matches_input_dtype


@pytest.mark.parametrize("irreps", ["0e", "0e + 1o", "1o + 2e", "2e + 0e", e3nn.s2_irreps(4)])
@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
@pytest.mark.parametrize("fft_to", [False, True])
@pytest.mark.parametrize("fft_from", [False, True])
def test_s2grid_transforms(keys, irreps, quadrature, fft_to, fft_from):
    @jax.jit
    def f(c):
        res = e3nn.to_s2grid(c, 30, 51, quadrature=quadrature, fft=fft_to, p_arg=-1)
        return e3nn.from_s2grid(res, c.irreps, fft=fft_from)

    a = e3nn.normal(irreps, keys[0])
    b = f(a)
    assert a.irreps == b.irreps
    np.testing.assert_allclose(a.array, b.array, rtol=1e-5, atol=1e-5)


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

    assert_output_dtype_matches_input_dtype(
        lambda x: e3nn.to_s2grid(x, 4, 5, normalization=normalization, quadrature=quadrature, fft=fft, p_arg=1),
        e3nn.IrrepsArray("0e", jnp.array([1.0])),
    )


@pytest.mark.parametrize("normalization", ["component", "norm"])
@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
@pytest.mark.parametrize("fft", [False, True])
def test_from_s2grid_dtype(normalization, quadrature, fft):
    jax.config.update("jax_enable_x64", True)

    assert_output_dtype_matches_input_dtype(
        lambda x: e3nn.from_s2grid(x, "0e + 4e", normalization=normalization, fft=fft),
        SphericalSignal(jnp.ones((10, 11)), quadrature=quadrature),
    )


@pytest.mark.parametrize("normalization", ["component", "norm"])
@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
@pytest.mark.parametrize("fft", [False, True])
def test_spherical_signal_vmap(quadrature):
    pointwise_func = jax.nn.relu
    signal_func = lambda sig: sig.apply(pointwise_func)
    grid_values = jnp.ones((2, 2, 2))

    sigs = SphericalSignal(grid_values=grid_values, quadrature=quadrature)
    transformed_sigs = jax.vmap(signal_func)(sigs)
    transformed_grid_values = jax.vmap(pointwise_func)(grid_values)

    np.testing.assert_allclose(transformed_sigs.grid_values, transformed_grid_values, atol=1e-7)


def test_fft_dtype():
    jax.config.update("jax_enable_x64", True)

    assert_output_dtype_matches_input_dtype(lambda x: rfft(x, 4), jnp.ones((10, 11)))
    assert_output_dtype_matches_input_dtype(lambda x: irfft(x, 11), jnp.ones((10, 11)))
