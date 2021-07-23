import pytest

import jax
import jax.numpy as jnp
from e3nn_jax import fully_connected_tensor_product, Irreps


@pytest.mark.parametrize('specialized_code', [False, True])
@pytest.mark.parametrize('optimize_einsums', [False, True])
@pytest.mark.parametrize('jitted', [False, True])
def test_modes(specialized_code, optimize_einsums, jitted):
    args = (
        Irreps("12x0e + 3x1o + 1x2e"),
        Irreps("10x0e + 2x1o + 1x2o"),
        Irreps("5x0e + 3x1e + 2x2o"),
    )

    _, n, f, _ = fully_connected_tensor_product(*args, specialized_code=specialized_code, optimize_einsums=optimize_einsums)
    if jitted:
        f = jax.jit(f)

    _, n, g, _ = fully_connected_tensor_product(*args)

    key = jax.random.PRNGKey(0)

    key, k = jax.random.split(key)
    w = jax.random.normal(k, (n,))

    key, k = jax.random.split(key)
    x1 = args[0].randn(k, (-1,))

    key, k = jax.random.split(key)
    x2 = args[1].randn(k, (-1,))

    assert jnp.allclose(f(w, x1, x2), g(w, x1, x2))
