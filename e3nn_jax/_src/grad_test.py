import e3nn_jax as e3nn
import numpy as np
from e3nn_jax.utils import assert_equivariant
from jax import random


def test_equivariance():
    assert_equivariant(
        e3nn.grad(lambda x: e3nn.tensor_product(x, x)),
        random.PRNGKey(0),
        "2x0e + 1e",
    )
    assert_equivariant(
        e3nn.grad(lambda x: e3nn.norm(x)), random.PRNGKey(1), "2x0e + 1e"
    )
    assert_equivariant(e3nn.grad(lambda x: e3nn.sum(x)), random.PRNGKey(2), "2x0e + 1e")


def test_simple_grad():
    def fn(x):
        return e3nn.sum(0.5 * e3nn.norm(x, squared=True).simplify())

    x = e3nn.normal("2e + 0e + 2x1o", random.PRNGKey(0), ())
    np.testing.assert_allclose(
        e3nn.grad(fn, regroup_output=False)(x).array, x.array, atol=1e-6, rtol=1e-6
    )


def test_grad_in_zero():
    def fn(x):
        return e3nn.sum(x)

    x = e3nn.zeros("0e", ())
    np.testing.assert_allclose(e3nn.grad(fn)(x).array, 1.0, atol=1e-6, rtol=1e-6)


def test_aux():
    def fn(x):
        return e3nn.sum(0.5 * e3nn.norm(x, squared=True).simplify()), x.irreps

    x = e3nn.normal("2e + 0e + 2x1o", random.PRNGKey(0), ())
    _, irreps = e3nn.grad(fn, has_aux=True)(x)
    assert irreps == x.irreps


def proportional(x, y, atol=1e-6, rtol=1e-6):
    x_zero = np.allclose(x, 0.0, atol=atol, rtol=rtol)
    y_zero = np.allclose(y, 0.0, atol=atol, rtol=rtol)
    if x_zero and y_zero:
        return True
    if x_zero or y_zero:
        return False
    i = np.argmax(np.abs(x))
    return np.allclose(y[i] / x[i] * x, y, atol=atol, rtol=rtol)


def test_argnums():
    def fn(x, y):
        return e3nn.tensor_product(x, y)["0e"]

    x = e3nn.normal("1o", random.PRNGKey(0), ())
    y = e3nn.normal("1o", random.PRNGKey(1), ())
    assert proportional(e3nn.grad(fn, argnums=1)(x, y).array, x.array)
