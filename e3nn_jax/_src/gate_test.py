import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import pytest
from e3nn_jax.utils import assert_equivariant

gate = jax.jit(jax.vmap(e3nn.gate))


@pytest.mark.parametrize(
    "irreps",
    [
        "0e + 0e + 1e",  # simple case: one extra scalar, one gate scalar, one vector
        "0e + 1e",  # no extra scalars
        "0e + 0o",  # no vectors
        "3x0o + 1e + 2e",  # extra scalars and gates are together
        "2x2e + 0e + 4x0o + 0e + 1e",  # vectors are all around
    ],
)
def test_gate(keys, irreps: e3nn.Irreps):
    x = e3nn.normal(irreps, next(keys), (128,))
    assert jnp.exp(jnp.abs(jnp.log(jnp.mean(gate(x).array ** 2)))) < 1.2

    assert_equivariant(gate, next(keys), x)
