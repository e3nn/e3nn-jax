import jax
import jax.numpy as jnp


import e3nn_jax as e3nn


def test_gate(keys):
    f = jax.jit(jax.vmap(e3nn.gate))

    x = e3nn.normal("16x0o + 32x0o + 16x1e + 16x1o", next(keys), (10,))
    y = f(x)

    assert jnp.abs(jnp.log(jnp.mean(y.array**2))) < 0.2

    R = -e3nn.rand_matrix(next(keys), ())
    y1 = y.transform_by_matrix(R)
    y2 = f(x.transform_by_matrix(R))

    assert jnp.allclose(y1.array, y2.array, atol=1e-6)
