import haiku as hk
import jax.numpy as jnp
import numpy as np

import e3nn_jax as e3nn


def test_tensor_product():
    x1 = e3nn.IrrepsArray("1o", jnp.array([1.0, 0.0, 0.0]))
    x2 = e3nn.IrrepsArray("1o", jnp.array([0.0, 1.0, 0.0]))
    x3 = e3nn.tensor_product(x1, x2, filter_ir_out=("1e",))
    assert x3.irreps == e3nn.Irreps("1e")
    np.testing.assert_allclose(x3.array, jnp.array([0.0, 0.0, 1 / 2**0.5]))


def test_tensor_product_with_zeros():
    x1 = e3nn.from_chunks("1o", [None], (), jnp.float32)
    x2 = e3nn.IrrepsArray("1o", jnp.array([0.0, 1.0, 0.0]))
    x3 = e3nn.tensor_product(x1, x2)
    assert x3.irreps == "0e + 1e + 2e"
    assert x3.zero_flags == (True, True, True)


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
        return e3nn.haiku.FullyConnectedTensorProduct("10x0e + 1e")(x1, x2)

    x1 = e3nn.normal("5x0e + 1e", next(keys), (10,))
    x2 = e3nn.normal("3x1e + 2x0e", next(keys), (20, 1))

    w = f.init(next(keys), x1, x2)
    x3 = f.apply(w, x1, x2)

    assert x3.irreps == e3nn.Irreps("10x0e + 1e")
    assert x3.shape[:-1] == (20, 10)


def test_square_normalization(keys):
    x = e3nn.normal("2x0e + 3x1e + 2x2e + 3e", keys[1], (100_000,))
    y = e3nn.tensor_square(x)
    assert jnp.all(jnp.exp(jnp.abs(jnp.log(jnp.mean(y.array**2, 0)))) < 1.1)


def test_tensor_square_normalization(keys):
    x = e3nn.normal("2x0e + 2x0o + 1o + 1e", keys[0], (10_000,))
    y = e3nn.tensor_square(x, irrep_normalization="component")
    np.testing.assert_allclose(
        e3nn.mean(e3nn.norm(y, squared=True), axis=0).array,
        np.array([ir.dim for mul, ir in y.irreps for _ in range(mul)]),
        rtol=0.1,
    )

    x = e3nn.normal("2x0e + 2x0o + 1o + 1e", keys[1], (10_000,), normalize=True)
    y = e3nn.tensor_square(x, normalized_input=True, irrep_normalization="norm")
    np.testing.assert_allclose(
        e3nn.mean(e3nn.norm(y, squared=True), axis=0).array, 1.0, rtol=0.1
    )

    x = e3nn.normal("2x0e + 2x0o + 1o + 1e", keys[1], (10_000,), normalize=True)
    y = e3nn.tensor_square(x, normalized_input=True, irrep_normalization="component")
    np.testing.assert_allclose(
        e3nn.mean(e3nn.norm(y, squared=True), axis=0).array,
        np.array([ir.dim for mul, ir in y.irreps for _ in range(mul)]),
        rtol=0.1,
    )

    x = e3nn.normal("2x0e + 2x0o + 1o + 1e", keys[1], (10_000,), normalization="norm")
    y = e3nn.tensor_square(x, irrep_normalization="norm")
    np.testing.assert_allclose(
        e3nn.mean(e3nn.norm(y, squared=True), axis=0).array, 1.0, rtol=0.1
    )


def test_tensor_square_and_spherical_harmonics(keys):
    x = e3nn.normal("1o", keys[0])

    y1 = e3nn.tensor_square(x, normalized_input=True, irrep_normalization="norm")["2e"]
    y2 = e3nn.spherical_harmonics("2e", x, normalize=False, normalization="norm")
    np.testing.assert_allclose(y1.array, y2.array)

    y1 = e3nn.tensor_square(x, normalized_input=True, irrep_normalization="component")[
        "2e"
    ]
    y2 = e3nn.spherical_harmonics("2e", x, normalize=False, normalization="component")
    np.testing.assert_allclose(y1.array, y2.array, atol=1e-5)

    # normalize the input
    y1 = e3nn.tensor_square(
        x / e3nn.norm(x), normalized_input=True, irrep_normalization="component"
    )["2e"]
    y2 = e3nn.spherical_harmonics("2e", x, normalize=True, normalization="component")
    np.testing.assert_allclose(y1.array, y2.array, atol=1e-5)
