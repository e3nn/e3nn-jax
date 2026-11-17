import jax
import jax.numpy as jnp
import numpy as np
import pytest

import e3nn_jax as e3nn


def test_errors(keys):
    x = e3nn.normal("0e + 1e", keys[0], ())
    with pytest.raises(ValueError):
        e3nn.scalar_activation(x, [None, jnp.tanh])


def test_zero_in_zero():
    x = e3nn.from_chunks("0e + 0o + 0o + 0e", [jnp.ones((1, 1)), None, None, None], ())
    y = e3nn.scalar_activation(x, [jnp.tanh, jnp.tanh, lambda x: x**2, jnp.cos])

    assert y.irreps == e3nn.Irreps("0e + 0o + 0e + 0e")
    assert y.chunks[1] is None
    assert y.chunks[2] is None
    assert y.chunks[3] is not None


def test_irreps_argument():
    assert e3nn.scalar_activation(
        "0e + 0o + 0o + 0e", [jnp.tanh, jnp.tanh, lambda x: x**2, jnp.cos]
    ) == e3nn.Irreps("0e + 0o + 0e + 0e")


@pytest.mark.parametrize("jac", (jax.jacrev, jax.jacfwd))
def test_norm_act(jac):
    def phi(n):
        return 1.0 / (1.0 + n * e3nn.sus(n))

    def f(x):
        return e3nn.norm_activation(e3nn.IrrepsArray("1o", x), [phi]).array

    J = jac(f)(jnp.array([0.0, 0.0, 1e-9]))
    np.testing.assert_allclose(J, np.diag([1.0, 1.0, 1.0]))
