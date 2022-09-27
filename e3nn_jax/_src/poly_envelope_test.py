import jax
import numpy as np
import pytest
from e3nn_jax import poly_envelope
from e3nn_jax._src.poly_envelope import u
from jax import numpy as jnp


def test_u():
    jax.config.update("jax_enable_x64", True)
    x = jnp.linspace(0.0, 0.99, 100)
    p = 6

    np.testing.assert_allclose(u(p, x), poly_envelope(p - 1, 2)(x))
    jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("n1", [0, 1, 3, 4])
@pytest.mark.parametrize("n0", [0, 1, 2, 5])
def test_poly_envelope(n0, n1):
    jax.config.update("jax_enable_x64", True)

    f = poly_envelope(n0, n1)

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
    jax.config.update("jax_enable_x64", False)
