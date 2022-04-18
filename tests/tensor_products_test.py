import haiku as hk
import jax.numpy as jnp
from e3nn_jax import FullyConnectedTensorProduct, Irreps, IrrepsData, elementwise_tensor_product, full_tensor_product


def test_full_tensor_product():
    x1 = IrrepsData.from_contiguous("1o", jnp.array([1.0, 0.0, 0.0]))
    x2 = IrrepsData.from_contiguous("1o", jnp.array([0.0, 1.0, 0.0]))
    x3 = full_tensor_product(x1, x2, filter_ir_out=("1e",))
    assert x3.irreps == Irreps("1e")
    assert jnp.allclose(x3.contiguous, jnp.array([0.0, 0.0, 1 / 2 ** 0.5]))


def test_full_tensor_product_irreps():
    irreps = full_tensor_product("1o", "1o", filter_ir_out=("1e",))
    assert irreps == Irreps("1e")


def test_elementwise_tensor_product(keys):
    x1 = IrrepsData.randn("0e + 1o", next(keys), (10,))
    x2 = IrrepsData.randn("1o + 0o", next(keys), (20, 1))

    x3 = elementwise_tensor_product(x1, x2)
    assert x3.irreps == Irreps("1o + 1e")
    assert x3.shape == (20, 10)


def test_fully_connected_tensor_product(keys):
    @hk.without_apply_rng
    @hk.transform
    def f(x1, x2):
        return FullyConnectedTensorProduct("10x0e + 1e")(x1, x2)

    x1 = IrrepsData.randn("5x0e + 1e", next(keys), (10,))
    x2 = IrrepsData.randn("3x1e + 2x0e", next(keys), (20, 1))

    w = f.init(next(keys), x1, x2)
    x3 = f.apply(w, x1, x2)
    assert x3.irreps == Irreps("10x0e + 1e")
    assert x3.shape == (20, 10)
