import jax
import jax.numpy as jnp
import numpy as np
import pytest

import e3nn_jax as e3nn
from e3nn_jax._src.s2grid import _irfft, _rfft, _spherical_harmonics_s2grid
from e3nn_jax.utils import assert_output_dtype_matches_input_dtype


@pytest.mark.parametrize(
    "irreps", ["0e", "0e + 1o", "1o + 2e", "2e + 0e", e3nn.s2_irreps(4)]
)
@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
@pytest.mark.parametrize("fft_to", [False, True])
@pytest.mark.parametrize("fft_from", [False, True])
def test_s2grid_transforms(keys, irreps, quadrature, fft_to, fft_from):
    @jax.jit
    def f(c):
        res = e3nn.to_s2grid(
            c, 30, 51, quadrature=quadrature, fft=fft_to, p_val=1, p_arg=-1
        )
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
    _, _, sh_y, sh_alpha, _ = _spherical_harmonics_s2grid(
        lmax=1, res_beta=4, res_alpha=5, quadrature=quadrature
    )
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
        lambda x: e3nn.to_s2grid(
            x,
            4,
            5,
            normalization=normalization,
            quadrature=quadrature,
            fft=fft,
            p_val=1,
            p_arg=1,
        ),
        e3nn.IrrepsArray("0e", jnp.array([1.0])),
    )


@pytest.mark.parametrize("normalization", ["component", "norm"])
@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
@pytest.mark.parametrize("fft", [False, True])
def test_from_s2grid_dtype(normalization, quadrature, fft):
    jax.config.update("jax_enable_x64", True)

    assert_output_dtype_matches_input_dtype(
        lambda x: e3nn.from_s2grid(x, "0e + 4e", normalization=normalization, fft=fft),
        e3nn.SphericalSignal(jnp.ones((10, 11)), quadrature=quadrature),
    )


@pytest.mark.parametrize("normalization", ["component", "norm"])
@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
@pytest.mark.parametrize("fft", [False, True])
@pytest.mark.parametrize("irreps", ["0e + 1o", "1o + 2e", e3nn.s2_irreps(4)])
def test_inverse(keys, normalization, quadrature, fft, irreps):
    jax.config.update("jax_enable_x64", True)

    coeffs_orig = e3nn.normal(irreps, keys[0], (12,))
    sigs = jax.vmap(
        lambda x: e3nn.to_s2grid(
            x, 100, 99, normalization=normalization, quadrature=quadrature
        )
    )(coeffs_orig)
    coeffs_new = jax.vmap(
        lambda y: e3nn.from_s2grid(y, irreps, normalization=normalization, fft=fft)
    )(sigs)

    np.testing.assert_allclose(
        coeffs_orig.array, coeffs_new.array, atol=1e-7, rtol=1e-7
    )


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
    s = e3nn.to_s2grid(
        coeffs, 20, 19, normalization=normalization, quadrature=quadrature
    )
    vec = e3nn.IrrepsArray({1: "1e", -1: "1o"}[s.p_arg], s.grid_vectors)
    values = e3nn.to_s2point(coeffs, vec, normalization=normalization)

    np.testing.assert_allclose(
        values.array[..., 0], s.grid_values, atol=1e-7, rtol=1e-7
    )

    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("alpha", [0.1, 0.2])
@pytest.mark.parametrize("beta", [0.1, 0.2])
@pytest.mark.parametrize("gamma", [0.1, 0.2])
@pytest.mark.parametrize("irreps", ["0e + 1e", "1o + 2e"])
def test_transform_by_angles(keys, irreps, alpha, beta, gamma):
    irreps = e3nn.Irreps(irreps)

    coeffs = e3nn.normal(irreps, keys[0], ())
    sig = e3nn.to_s2grid(coeffs, 20, 19, quadrature="gausslegendre")
    rotated_sig = sig.transform_by_angles(alpha, beta, gamma, lmax=irreps.lmax)
    rotated_coeffs = e3nn.from_s2grid(rotated_sig, irreps)
    expected_rotated_coeffs = coeffs.transform_by_angles(alpha, beta, gamma)

    np.testing.assert_allclose(
        rotated_coeffs.array, expected_rotated_coeffs.array, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("alpha", [0.1, 0.2])
@pytest.mark.parametrize("beta", [0.1, 0.2])
@pytest.mark.parametrize("gamma", [0.1, 0.2])
@pytest.mark.parametrize("irreps", ["0e + 1e", "1o + 2e"])
def test_transform_by_matrix(keys, irreps, alpha, beta, gamma):
    irreps = e3nn.Irreps(irreps)

    coeffs = e3nn.normal(irreps, keys[0], ())
    sig = e3nn.to_s2grid(coeffs, 20, 19, quadrature="soft")
    R = e3nn.angles_to_matrix(alpha, beta, gamma)
    rotated_sig = sig.transform_by_matrix(R, lmax=irreps.lmax)
    rotated_coeffs = e3nn.from_s2grid(rotated_sig, irreps)
    expected_rotated_coeffs = coeffs.transform_by_angles(alpha, beta, gamma)

    np.testing.assert_allclose(
        rotated_coeffs.array, expected_rotated_coeffs.array, atol=1e-5, rtol=1e-5
    )


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

    np.testing.assert_allclose(
        rotated_coeffs.array, expected_rotated_coeffs.array, atol=1e-5, rtol=1e-5
    )


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

    np.testing.assert_allclose(
        rotated_coeffs.array, expected_rotated_coeffs.array, atol=1e-5, rtol=1e-5
    )


def test_s2_dirac():
    jax.config.update("jax_enable_x64", True)

    x = e3nn.s2_dirac(jnp.array([0.0, 1.0, 0.0]), lmax=45, p_val=1, p_arg=-1)
    sig = e3nn.to_s2grid(x, 200, 59, quadrature="gausslegendre")

    # The integral of a Dirac delta is 1
    np.testing.assert_allclose(sig.integrate().array, 1.0)

    # All the weight should be located at the north pole
    sig.grid_values = sig.grid_values.at[-60:].set(0.0)
    np.testing.assert_allclose(sig.integrate().array, 0.0, atol=0.05)


@pytest.mark.parametrize("lmax", [1, 2, 3, 4])
@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
def test_integrate_scalar(lmax, quadrature):
    coeffs = e3nn.normal(e3nn.s2_irreps(lmax, p_val=1, p_arg=-1), jax.random.PRNGKey(0))
    sig = e3nn.to_s2grid(
        coeffs,
        100,
        99,
        normalization="integral",
        quadrature=quadrature,
        p_val=1,
        p_arg=-1,
    )
    integral = sig.integrate().array.squeeze()

    scalar_term = coeffs["0e"].array[0]
    expected_integral = 4 * jnp.pi * scalar_term
    np.testing.assert_allclose(integral, expected_integral, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("degree", range(10))
def test_integrate_polynomials(degree):
    sig = e3nn.SphericalSignal(np.empty((26, 17)), "gausslegendre")
    sig.grid_values = (sig.grid_y**degree)[:, None] * jnp.ones_like(sig.grid_values)
    integral = sig.integrate().array.squeeze()

    expected_integral = 4 * jnp.pi / (degree + 1) if degree % 2 == 0 else 0
    np.testing.assert_allclose(integral, expected_integral, atol=1e-5, rtol=1e-5)


def test_sample(keys):
    p = e3nn.to_s2grid(
        0.5 * e3nn.normal(e3nn.s2_irreps(4), keys[0]),
        30,
        51,
        quadrature="gausslegendre",
    ).apply(jnp.exp)
    p: e3nn.SphericalSignal = p / p.integrate()
    keys = jax.random.split(keys[1], 100_000)
    beta_index, alpha_index = jax.vmap(lambda k: p.sample(k))(keys)

    f = jnp.zeros_like(p.grid_values)
    f = f.at[beta_index, alpha_index].add(1.0)

    f = e3nn.SphericalSignal(f / p.quadrature_weights[:, None], p.quadrature)
    f = f / f.integrate()

    err = (p - f).apply(jnp.square).integrate().array[0]
    assert err < 2e-3


@pytest.mark.parametrize("lmax", [2, 4, 10])
def test_find_peaks(lmax):
    pytest.skip(
        "Still has the bug `ValueError: buffer source array is read-only`"
    )  # TODO

    pos = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    val = jnp.asarray(
        [
            1.0,
            -1.0,
        ]
    )
    coeffs = e3nn.s2_sum_of_diracs(
        positions=pos, weights=val, lmax=lmax, p_val=1, p_arg=-1
    )
    sig = e3nn.to_s2grid(coeffs, 50, 49, quadrature="gausslegendre")

    x, f = sig.find_peaks(lmax)
    positive_peak = x[f.argmax()]
    np.testing.assert_allclose(positive_peak, pos[0], atol=4e-1 / lmax)

    x, f = sig.apply(lambda val: -val).find_peaks(lmax)
    negative_peak = x[f.argmax()]
    np.testing.assert_allclose(negative_peak, pos[1], atol=4e-1 / lmax)
