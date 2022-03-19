import jax.numpy as jnp
from e3nn_jax import full_tensor_product, IrrepsData, Irreps


def test_full_tensor_product():
    x1 = IrrepsData.from_contiguous("1o", jnp.array([1.0, 0.0, 0.0]))
    x2 = IrrepsData.from_contiguous("1o", jnp.array([0.0, 1.0, 0.0]))
    x3 = full_tensor_product(x1, x2, filter_ir_out=("1e",))
    assert x3.irreps == Irreps("1e")
    assert jnp.allclose(x3.contiguous, jnp.array([0.0, 0.0, 1/2**0.5]))


def test_full_tensor_product_irreps():
    irreps = full_tensor_product("1o", "1o", filter_ir_out=("1e",))
    assert irreps == Irreps("1e")
