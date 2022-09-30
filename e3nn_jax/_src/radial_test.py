import jax.numpy as jnp
import e3nn_jax as e3nn
import pytest
import numpy as np


def test_sus():
    np.testing.assert_allclose(e3nn.sus(0.0), 0.0)
    np.testing.assert_allclose(e3nn.sus(1e7), 1.0, atol=1e-6)

    x = np.linspace(-10.0, 10.0, 100)

    assert np.all(e3nn.sus(x) >= 0.0)
    assert np.all(e3nn.sus(x) <= 1.0)


@pytest.mark.parametrize("end_zero", [True, False])
@pytest.mark.parametrize("start_zero", [True, False])
@pytest.mark.parametrize("basis", ["gaussian", "cosine", "smooth_finite", "fourier"])
def test_soft_one_hot_linspace(basis: str, start_zero: bool, end_zero: bool):
    if basis == "fourier" and start_zero != end_zero:
        pytest.skip()

    x = np.linspace(0.2, 0.8, 100)
    y = e3nn.soft_one_hot_linspace(x, start=0.0, end=1.0, number=5, basis=basis, start_zero=start_zero, end_zero=end_zero)
    assert y.shape == (100, 5)

    np.testing.assert_allclose(jnp.sum(y**2, axis=1), 1.0, atol=0.4)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_bessel(n: int):
    x = np.linspace(0.0, 1.0, 100)
    y = e3nn.bessel(x, n)
    assert y.shape == (100, n)
