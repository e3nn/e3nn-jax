import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp
import numpy as np


def test_tensor_product():
    x1 = e3nn.IrrepsArray("1o", jnp.array([1.0, 0.0, 0.0]))
    x2 = e3nn.IrrepsArray("1o", jnp.array([0.0, 1.0, 0.0]))
    x3 = e3nn.tensor_product(x1, x2, filter_ir_out=("1e",))
    assert x3.irreps == e3nn.Irreps("1e")
    np.testing.assert_allclose(x3.array, jnp.array([0.0, 0.0, 1 / 2**0.5]))


def test_tensor_product_irreps():
    irreps = e3nn.tensor_product("1o", "1o", filter_ir_out=("1e",))
    assert irreps == e3nn.Irreps("1e")


def test_elementwise_tensor_product(keys):
    x1 = e3nn.normal("0e + 1o", next(keys), (10,))
    x2 = e3nn.normal("1o + 0o", next(keys), (20, 1))

    x3 = e3nn.elementwise_tensor_product(x1, x2)
    assert x3.irreps == e3nn.Irreps("1o + 1e")
    assert x3.shape == (20, 10, 6)


def test_fully_connected_tensor_product(keys):
    @hk.without_apply_rng
    @hk.transform
    def f(x1, x2):
        return e3nn.FullyConnectedTensorProduct("10x0e + 1e")(x1, x2)

    x1 = e3nn.normal("5x0e + 1e", next(keys), (10,))
    x2 = e3nn.normal("3x1e + 2x0e", next(keys), (20, 1))

    w = f.init(next(keys), x1, x2)
    x3 = f.apply(w, x1, x2)
    assert x3.irreps == e3nn.Irreps("10x0e + 1e")
    assert x3.shape[:-1] == (20, 10)
