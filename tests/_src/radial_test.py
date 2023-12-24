import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from e3nn_jax._src.radial import u
from e3nn_jax._src.utils.test import assert_output_dtype_matches_input_dtype


def test_sus():
    np.testing.assert_allclose(e3nn.sus(0.0), 0.0)
    np.testing.assert_allclose(e3nn.sus(1e7), 1.0, atol=1e-6)

    x = jnp.linspace(-10.0, 10.0, 100)

    assert np.all(e3nn.sus(x) >= 0.0)
    assert np.all(e3nn.sus(x) <= 1.0)


@pytest.mark.parametrize("end_zero", [True, False])
@pytest.mark.parametrize("start_zero", [True, False])
@pytest.mark.parametrize("basis", ["gaussian", "cosine", "smooth_finite", "fourier"])
def test_soft_one_hot_linspace(basis: str, start_zero: bool, end_zero: bool):
    if basis == "fourier" and start_zero != end_zero:
        pytest.skip()

    x = jnp.linspace(0.2, 0.8, 100)
    y = e3nn.soft_one_hot_linspace(
        x,
        start=0.0,
        end=1.0,
        number=5,
        basis=basis,
        start_zero=start_zero,
        end_zero=end_zero,
    )
    assert y.shape == (100, 5)

    np.testing.assert_allclose(jnp.sum(y**2, axis=1), 1.0, atol=0.4)

    jax.config.update("jax_enable_x64", True)
    assert_output_dtype_matches_input_dtype(
        lambda x: e3nn.soft_one_hot_linspace(
            x,
            start=0.0,
            end=1.0,
            number=5,
            basis=basis,
            start_zero=start_zero,
            end_zero=end_zero,
        ),
        x,
    )


@pytest.mark.parametrize("n", [1, 2, 4])
def test_bessel(n: int):
    x = jnp.linspace(0.0, 1.0, 100)
    y = e3nn.bessel(x, n)
    assert y.shape == (100, n)


def test_u():
    jax.config.update("jax_enable_x64", True)
    x = jnp.linspace(0.0, 0.99, 100)
    p = 6

    np.testing.assert_allclose(u(p, x), e3nn.poly_envelope(p - 1, 2)(x))
    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("n1", [0, 1, 3, 4])
@pytest.mark.parametrize("n0", [0, 1, 2, 5])
def test_poly_envelope(n0, n1):
    jax.config.update("jax_enable_x64", True)

    f = e3nn.poly_envelope(n0, n1)

    np.testing.assert_allclose(f(0), 1.0, atol=1e-9)
    np.testing.assert_allclose(f(1), 0.0, atol=1e-9)

    d = f
    for _ in range(n0):
        d = jax.grad(d)
        np.testing.assert_allclose(d(0.0), 0.0, atol=1e-9)

    d = f
    for _ in range(n1):
        d = jax.grad(d)
        np.testing.assert_allclose(d(1.0), 0.0, atol=1e-9)

    x = jnp.linspace(1.0, 1e6, 100)
    y = f(x)

    np.testing.assert_allclose(y, 0.0, atol=1e-9)

    jax.config.update("jax_enable_x64", False)
