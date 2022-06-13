import jax
import jax.numpy as jnp

from e3nn_jax import IrrepsData, gate, rand_matrix


def test_gate(keys):
    f = jax.jit(jax.vmap(gate))

    x = IrrepsData.randn("16x0o + 32x0o + 16x1e + 16x1o", next(keys), (10,))
    y = f(x)

    assert jnp.abs(jnp.log(jnp.mean(y.contiguous**2))) < 0.2

    R = -rand_matrix(next(keys), ())
    y1 = y.transform_by_matrix(R)
    y2 = f(x.transform_by_matrix(R))

    assert jnp.allclose(y1.contiguous, y2.contiguous, atol=1e-6)
