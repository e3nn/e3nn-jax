import jax
import numpy as np
import pytest
import jax.numpy as jnp

import e3nn_jax as e3nn
from e3nn_jax._src.s2grid import _irfft, _rfft, _spherical_harmonics_s2grid, SphericalSignal, sum_of_diracs
from e3nn_jax.util import assert_output_dtype_matches_input_dtype


@pytest.mark.parametrize("irreps", ["0e", "0e + 1o", "1o + 2e", "2e + 0e", e3nn.s2_irreps(4)])
@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
@pytest.mark.parametrize("fft_to", [False, True])
@pytest.mark.parametrize("fft_from", [False, True])
def test_s2grid_transforms(keys, irreps, quadrature, fft_to, fft_from):
    @jax.jit
    def f(c):
        res = e3nn.to_s2grid(c, 30, 51, quadrature=quadrature, fft=fft_to, p_val=1, p_arg=-1)
        return e3nn.from_s2grid(res, c.irreps, fft=fft_from)

    a = e3nn.normal(irreps, keys[0])
    b = f(a)
    assert a.irreps == b.irreps
    np.testing.assert_allclose(a.array, b.array, rtol=1e-5, atol=1e-5)


def test_fft(keys):
    res_alpha = 11  # 2l+1
    l = 5
    x = jax.random.uniform(keys[0], shape=(8, res_alpha))
    x_t = _rfft(x, l)
    x_p = _irfft(x_t, res_alpha)
    np.testing.assert_allclose(x, x_p, rtol=1e-5)


@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
def test_grid_vectors(quadrature):
    _, _, sh_y, sh_alpha, _ = _spherical_harmonics_s2grid(lmax=1, res_beta=4, res_alpha=5, quadrature=quadrature)
    r = e3nn.SphericalSignal(jnp.empty((4, 5)), quadrature).grid_vectors

    sh_y = np.stack([sh_y[:, 2], sh_y[:, 1], sh_y[:, 2]], axis=1)  # for l=1
    sh = sh_y[:, None, :] * sh_alpha
    sh = sh / np.linalg.norm(sh, axis=2, keepdims=True)

    np.testing.assert_allclose(sh, r, atol=1e-7)


def test_properties():
    x = e3nn.SphericalSignal(jnp.ones((3, 10, 11)), quadrature="soft")
    assert x.res_beta == 10
    assert x.res_alpha == 11
    assert x.quadrature == "soft"
    assert x.grid_vectors.shape == (10, 11, 3)
    assert x.dtype == x.grid_values.dtype
    assert x.shape == x.grid_values.shape


@pytest.mark.parametrize("normalization", ["component", "norm"])
@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
@pytest.mark.parametrize("fft", [False, True])
def test_to_s2grid_dtype(normalization, quadrature, fft):
    jax.config.update("jax_enable_x64", True)

    assert_output_dtype_matches_input_dtype(
        lambda x: e3nn.to_s2grid(x, 4, 5, normalization=normalization, quadrature=quadrature, fft=fft, p_val=1, p_arg=1),
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
def test_spherical_signal_vmap(normalization, quadrature, fft):
    irreps = "0e + 1o"
    coeffs_orig = e3nn.IrrepsArray(irreps, jnp.ones((10, 4)))
    sigs = jax.vmap(lambda x: e3nn.to_s2grid(x, 100, 99, normalization=normalization, quadrature=quadrature))(coeffs_orig)
    coeffs_new = jax.vmap(lambda y: e3nn.from_s2grid(y, irreps, normalization=normalization, fft=fft))(sigs)

    np.testing.assert_allclose(coeffs_orig.array, coeffs_new.array, atol=1e-7, rtol=1e-7)


def test_fft_dtype():
    jax.config.update("jax_enable_x64", True)

    assert_output_dtype_matches_input_dtype(lambda x: _rfft(x, 4), jnp.ones((10, 11)))
    assert_output_dtype_matches_input_dtype(lambda x: _irfft(x, 11), jnp.ones((10, 11)))


@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
@pytest.mark.parametrize("irreps", ["0e + 1e", "1o + 2e"])
def test_to_s2point(keys, irreps, normalization, quadrature):
    jax.config.update("jax_enable_x64", True)

    coeffs = e3nn.normal(irreps, keys[0], ())
    s = e3nn.to_s2grid(coeffs, 20, 19, normalization=normalization, quadrature=quadrature)
    vec = e3nn.IrrepsArray({1: "1e", -1: "1o"}[s.p_arg], s.grid_vectors)
    values = e3nn.to_s2point(coeffs, vec, normalization=normalization)

    np.testing.assert_allclose(values.array[..., 0], s.grid_values, atol=1e-7, rtol=1e-7)

    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("alpha", [0.1, 0.2])
@pytest.mark.parametrize("beta", [0.1, 0.2])
@pytest.mark.parametrize("gamma", [0.1, 0.2])
@pytest.mark.parametrize("irreps", ["0e + 1e", "1o + 2e"])
def test_transform_by_angles(keys, irreps, alpha, beta, gamma):
    irreps = e3nn.Irreps(irreps)

    coeffs = e3nn.normal(irreps, keys[0], ())
    sig = e3nn.to_s2grid(coeffs, 20, 19, quadrature="soft")
    rotated_sig = sig.transform_by_angles(alpha, beta, gamma, lmax=irreps.lmax)
    rotated_coeffs = e3nn.from_s2grid(rotated_sig, irreps)
    expected_rotated_coeffs = coeffs.transform_by_angles(alpha, beta, gamma)

    np.testing.assert_allclose(rotated_coeffs.array, expected_rotated_coeffs.array, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("alpha", [0.1, 0.2])
@pytest.mark.parametrize("beta", [0.1, 0.2])
@pytest.mark.parametrize("gamma", [0.1, 0.2])
@pytest.mark.parametrize("irreps", ["0e + 1e", "1o + 2e"])
def test_transform_by_angles(keys, irreps, alpha, beta, gamma):
    irreps = e3nn.Irreps(irreps)

    coeffs = e3nn.normal(irreps, keys[0], ())
    sig = e3nn.to_s2grid(coeffs, 20, 19, quadrature="soft")
    R = e3nn.angles_to_matrix(alpha, beta, gamma)
    rotated_sig = sig.transform_by_matrix(R, lmax=irreps.lmax)
    rotated_coeffs = e3nn.from_s2grid(rotated_sig, irreps)
    expected_rotated_coeffs = coeffs.transform_by_angles(alpha, beta, gamma)

    np.testing.assert_allclose(rotated_coeffs.array, expected_rotated_coeffs.array, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("alpha", [0.1, 0.2])
@pytest.mark.parametrize("beta", [0.1, 0.2])
@pytest.mark.parametrize("gamma", [0.1, 0.2])
@pytest.mark.parametrize("irreps", ["0e + 1e", "1o + 2e"])
def test_transform_by_axis_angle(keys, irreps, alpha, beta, gamma):
    irreps = e3nn.Irreps(irreps)

    coeffs = e3nn.normal(irreps, keys[0], ())
    sig = e3nn.to_s2grid(coeffs, 20, 19, quadrature="soft")
    axis_angle = e3nn.angles_to_axis_angle(alpha, beta, gamma)
    rotated_sig = sig.transform_by_axis_angle(*axis_angle, lmax=irreps.lmax)
    rotated_coeffs = e3nn.from_s2grid(rotated_sig, irreps)
    expected_rotated_coeffs = coeffs.transform_by_angles(alpha, beta, gamma)

    np.testing.assert_allclose(rotated_coeffs.array, expected_rotated_coeffs.array, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("alpha", [0.1, 0.2])
@pytest.mark.parametrize("beta", [0.1, 0.2])
@pytest.mark.parametrize("gamma", [0.1, 0.2])
@pytest.mark.parametrize("irreps", ["0e + 1e", "1o + 2e"])
def test_transform_by_quaternion(keys, irreps, alpha, beta, gamma):
    irreps = e3nn.Irreps(irreps)
    coeffs = e3nn.normal(irreps, keys[0], ())
    sig = e3nn.to_s2grid(coeffs, 20, 19, quadrature="soft")

    q = e3nn.angles_to_quaternion(alpha, beta, gamma)
    rotated_sig = sig.transform_by_quaternion(q, lmax=irreps.lmax)
    rotated_coeffs = e3nn.from_s2grid(rotated_sig, irreps)
    expected_rotated_coeffs = coeffs.transform_by_angles(alpha, beta, gamma)

    np.testing.assert_allclose(rotated_coeffs.array, expected_rotated_coeffs.array, atol=1e-5, rtol=1e-5)


# TODO (mariogeiger): Uncomment once to_s2point is fixed.
# @pytest.mark.parametrize("lmax", [1, 4, 10])
# def test_sum_of_diracs(lmax):
#     pos = jnp.asarray([
#         [1.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0],
#     ])
#     val = jnp.asarray([
#         -1.0,
#         1.0,
#     ])
#     coeffs = sum_of_diracs(positions=pos, values=val, lmax=lmax, p_val=1, p_arg=-1)
#     points = e3nn.IrrepsArray(irreps="1o", array=jnp.asarray(pos))
#     extracted_val = e3nn.to_s2point(coeffs, point=points)
#     np.testing.assert_allclose(extracted_val.array, val.array, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("lmax", [1, 2, 3, 4])
def test_integrate_constant(lmax):
    coeffs = e3nn.normal(e3nn.s2_irreps(lmax, p_val=1, p_arg=-1), jax.random.PRNGKey(0))
    sig = e3nn.to_s2grid(coeffs, 100, 99, normalization="integral", quadrature="gausslegendre", p_val=1, p_arg=-1)
    integral = sig.integrate().array.squeeze()
    
    scalar_term = coeffs["0e"].array[0]
    expected_integral = 4 * jnp.pi * scalar_term
    np.testing.assert_allclose(integral, expected_integral, atol=1e-5, rtol=1e-5)


# @pytest.mark.parametrize("lmax", [1, 4, 10])
# def test_find_peaks(lmax):
#     pos = jnp.asarray([
#         [1.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0],
#     ])
#     val = jnp.asarray([
#         -1.0,
#         1.0,
#     ])
#     coeffs = sum_of_diracs(positions=pos, values=val, lmax=lmax, p_val=1, p_arg=-1)
#     sig = e3nn.to_s2grid(coeffs, 15, 19, quadrature="gausslegendre")

#     print(sig.find_peaks(lmax))
