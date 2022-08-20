import e3nn_jax as e3nn
import numpy as np
from e3nn_jax.util.test import assert_equivariant
from jax import random


def test_equivariance():
    assert_equivariant(e3nn.grad(lambda x: e3nn.tensor_product(x, x)), random.PRNGKey(0), irreps_in=("2x0e + 1e",))
    assert_equivariant(e3nn.grad(lambda x: e3nn.norm(x)), random.PRNGKey(1), irreps_in=("2x0e + 1e",))
    assert_equivariant(e3nn.grad(lambda x: e3nn.sum(x)), random.PRNGKey(2), irreps_in=("2x0e + 1e",))


def test_simple_grad():
    def fn(x):
        return e3nn.sum(0.5 * e3nn.norm(x, squared=True).simplify())

    x = e3nn.normal("2e + 0e + 2x1o", random.PRNGKey(0), ())
    np.testing.assert_allclose(e3nn.grad(fn)(x).array, x.array, atol=1e-6, rtol=1e-6)
