import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_convert():
    id = e3nn.IrrepsArray.from_any("10x0e + 10x0e", [None, jnp.ones((1, 10, 1))])
    assert jax.tree_util.tree_map(lambda x: x.shape, id.convert("0x0e + 20x0e + 0x0e")).list == [None, (1, 20, 1), None]
    assert jax.tree_util.tree_map(lambda x: x.shape, id.convert("7x0e + 4x0e + 9x0e")).list == [None, (1, 4, 1), (1, 9, 1)]

    id = e3nn.IrrepsArray.from_any("10x0e + 10x1e", [None, jnp.ones((1, 10, 3))])
    assert jax.tree_util.tree_map(lambda x: x.shape, id.convert("5x0e + 5x0e + 5x1e + 5x1e")).list == [
        None,
        None,
        (1, 5, 3),
        (1, 5, 3),
    ]

    id = e3nn.IrrepsArray.zeros("10x0e + 10x1e", ())
    id = id.convert("5x0e + 0x2e + 5x0e + 0x2e + 5x1e + 5x1e")

    a = e3nn.IrrepsArray.from_list(
        "            10x0e  +  0x0e +1x1e  +     0x0e    +          9x1e           + 0x0e",
        [jnp.ones((2, 10, 1)), None, None, jnp.ones((2, 0, 1)), jnp.ones((2, 9, 3)), None],
        (2,),
    )
    b = a.convert("5x0e + 0x2e + 5x0e + 0x2e + 5x1e + 5x1e")
    b = e3nn.IrrepsArray.from_list(b.irreps, b.list, b.shape[:-1])

    np.testing.assert_allclose(a.array, b.array)


def test_indexing():
    x = e3nn.IrrepsArray("2x0e + 1x0e", jnp.array([[1.0, 2, 3], [4.0, 5, 6]]))
    assert x.shape == (2, 3)
    assert jnp.allclose(x[0].array, jnp.array([1.0, 2, 3]))
    assert jnp.allclose(x[1, "1x0e"].array, jnp.array([6.0]))
    assert jnp.allclose(x[:, "1x0e"].array, jnp.array([[3.0], [6.0]]))
    assert jnp.allclose(x[..., "1x0e"].array, jnp.array([[3.0], [6.0]]))
    assert jnp.allclose(x[..., 1, "1x0e"].array, jnp.array([6.0]))


def test_reductions():
    x = e3nn.IrrepsArray("2x0e + 1x1e", jnp.array([[1.0, 2, 3, 4, 5], [4.0, 5, 6, 6, 6]]))
    assert e3nn.sum(x).irreps == "0e + 1e"
    np.testing.assert_allclose(e3nn.sum(x).array, jnp.array([12.0, 9, 10, 11]))
    np.testing.assert_allclose(e3nn.sum(x, axis=0).array, jnp.array([5.0, 7, 9, 10, 11]))
    np.testing.assert_allclose(e3nn.sum(x, axis=1).array, jnp.array([[3.0, 3, 4, 5], [9.0, 6, 6, 6]]))

    np.testing.assert_allclose(e3nn.mean(x, axis=1).array, jnp.array([[1.5, 3, 4, 5], [4.5, 6, 6, 6]]))


def test_operators():
    x = e3nn.IrrepsArray("2x0e + 1x1e", jnp.array([[1.0, 2, 3, 4, 5], [4.0, 5, 6, 6, 6]]))
    y = e3nn.IrrepsArray("2x0e + 1x1o", jnp.array([[1.0, 2, 3, 4, 5], [4.0, 5, 6, 6, 6]]))

    with pytest.raises(ValueError):
        x + 1

    e3nn.norm(x) + 1

    assert (x + x).shape == x.shape

    with pytest.raises(ValueError):
        x + y

    assert (x - x).shape == x.shape

    with pytest.raises(ValueError):
        x - y

    assert (x * 2.0).shape == x.shape
    assert (x / 2.0).shape == x.shape
    assert (x * jnp.array([[2], [3.0]])).shape == x.shape

    with pytest.raises(ValueError):
        x * jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    with pytest.raises(ValueError):
        1.0 / x

    1.0 / e3nn.norm(x)

    np.testing.assert_allclose(e3nn.norm(x / e3nn.norm(x)).array, 1)
