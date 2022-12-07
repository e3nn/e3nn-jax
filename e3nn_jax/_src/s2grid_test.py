import jax
import numpy as np
import pytest
from e3nn_jax._src.s2grid import from_s2grid, irfft, rfft, to_s2grid
from e3nn_jax._src.irreps_array import Irreps, IrrepsArray

key = jax.random.PRNGKey(0)


@pytest.mark.parametrize("quadrature", ["soft", "gausslegendre"])
@pytest.mark.parametrize("fft_to", [False, True])
@pytest.mark.parametrize("fft_from", [False, True])
def test_s2grid_transforms(quadrature, fft_to, fft_from):
    assert quadrature in ["soft", "gausslegendre"], "quadrature must be 'soft' or 'gausslegendre"
    res_alpha = 51
    res_beta = 30
    lmax = 10
    p_val = 1
    p_arg = -1

    c = jax.random.uniform(key, shape=(1, (lmax + 1) ** 2))
    irreps = Irreps([(1, (l, p_val * p_arg**l)) for l in range(lmax + 1)])
    irreps_in = IrrepsArray(irreps, c)

    res = to_s2grid(irreps_in, (res_beta, res_alpha), quadrature=quadrature, fft=fft_to)
    irreps_out = from_s2grid(res, lmax, quadrature=quadrature, fft=fft_from)
    np.testing.assert_allclose(c, irreps_out.array, rtol=1e-5, atol=1e-5)
    assert irreps_in.irreps == irreps_out.irreps


def test_fft():
    res_alpha = 11  # 2l+1
    l = 5
    x = jax.random.uniform(key, shape=(8, res_alpha))
    x_t = rfft(x, l)
    x_p = irfft(x_t, res_alpha)
    np.testing.assert_allclose(x, x_p, rtol=1e-5)
