import jax
import numpy as np
from e3nn_jax._src.s2grid import from_s2grid, irfft, rfft, to_s2grid

key = jax.random.PRNGKey(0)


def test_s2grid_transforms():
    res_alpha = 51
    res_beta = 30
    l = 10

    # SOFT quadrature
    quadrature = "soft"
    c = jax.random.uniform(key, shape=(5, (l + 1) ** 2))
    res = to_s2grid(c, (res_beta, res_alpha), quadrature=quadrature)
    c_prime = from_s2grid(res, l, quadrature=quadrature)
    np.testing.assert_allclose(c, c_prime, rtol=1e-5, atol=1e-5)

    # Gauss-Legendre quadrature
    quadrature = "gausslegendre"
    c = jax.random.uniform(key, shape=(5, (l + 1) ** 2))
    res = to_s2grid(c, (res_beta, res_alpha), quadrature=quadrature)
    c_prime = from_s2grid(res, l, quadrature=quadrature)
    np.testing.assert_allclose(c, c_prime, rtol=1e-5, atol=1e-5)


def test_fft():
    res_alpha = 11  # 2l+1
    l = 5
    x = jax.random.uniform(key, shape=(8, res_alpha))
    x_t = rfft(x, l)
    x_p = irfft(x_t, res_alpha)
    np.testing.assert_allclose(x, x_p, rtol=1e-5)


if __name__ == "__main__":
    test_s2grid_transforms()
    test_fft()
