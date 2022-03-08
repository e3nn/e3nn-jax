import jax
import jax.numpy as jnp

from e3nn_jax import Irreps, Gate, rand_matrix


def test_gate(keys):
    irreps_scalars, act_scalars = Irreps("16x0o"), [jnp.tanh]
    irreps_gates, act_gates, irreps_gated = Irreps("32x0o"), [jnp.tanh], Irreps("16x1e+16x1o")

    g = Gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated)
    f = jax.jit(lambda x: jax.vmap(g)(x).contiguous)

    x = g.irreps_in.randn(next(keys), (10, -1,))
    y = f(x)

    assert jnp.abs(jnp.log(jnp.mean(y**2))) < 0.2

    R = -rand_matrix(next(keys), ())
    y1 = g.irreps_out.transform_by_matrix(y, R)
    y2 = f(g.irreps_in.transform_by_matrix(x, R))

    assert jnp.allclose(y1, y2, atol=1e-6)
