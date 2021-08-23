import pytest

import jax
import jax.numpy as jnp
from e3nn_jax import TensorProduct, Irreps


@pytest.mark.parametrize('connection_mode', ['uvw', 'uvu', 'uvv'])
@pytest.mark.parametrize('jitted', [False, True])
@pytest.mark.parametrize('optimize_einsums', [False, True])
@pytest.mark.parametrize('specialized_code', [False, True])
@pytest.mark.parametrize('normalization', ['component', 'norm'])
def test_modes(normalization, specialized_code, optimize_einsums, jitted, connection_mode):
    tp = TensorProduct(
        Irreps("10x0o + 10x1o + 1x2e"),
        Irreps("10x0o + 10x1o + 1x2o"),
        Irreps("10x0e + 10x1e + 2x2o"),
        [
            (0, 0, 0, connection_mode, True),
            (1, 1, 1, connection_mode, True),
            (1, 0, 1, connection_mode, True),
            (2, 2, 2, 'uvw', True),
            (2, 1, 2, 'uvw', True),
        ],
        normalization=normalization,
    )

    def f(ws, x1, x2):
        return tp.left_right(
            ws, x1, x2,
            specialized_code=specialized_code,
            optimize_einsums=optimize_einsums,
            custom_einsum_vjp=optimize_einsums
        )

    if jitted:
        f = jax.jit(f)

    g = tp.left_right

    def k():
        k.key, x = jax.random.split(k.key)
        return x
    k.key = jax.random.PRNGKey(0)

    ws = [jax.random.normal(k(), ins.path_shape) for ins in tp.instructions if ins.has_weight]
    x1 = tp.irreps_in1.randn(k(), (-1,), normalization)
    x2 = tp.irreps_in2.randn(k(), (-1,), normalization)

    a = f(ws, x1, x2)
    b = g(ws, x1, x2)
    assert jnp.allclose(a, b, rtol=1e-4, atol=1e-6), jnp.max(jnp.abs(a - b))
