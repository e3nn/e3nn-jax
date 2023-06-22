from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import e3nn_jax as e3nn


def test_empty():
    x = e3nn.from_chunks("", [], (2, 2), jnp.float32)
    assert x.irreps == e3nn.Irreps([])
    assert x.shape == (2, 2, 0)


def test_convert():
    id = e3nn.from_chunks("10x0e + 10x0e", [None, jnp.ones((1, 10, 1))], (1,))
    assert jax.tree_util.tree_map(
        jnp.shape, id.rechunk("0x0e + 20x0e + 0x0e").chunks
    ) == [None, (1, 20, 1), None]
    assert jax.tree_util.tree_map(
        jnp.shape, id.rechunk("7x0e + 4x0e + 9x0e").chunks
    ) == [None, (1, 4, 1), (1, 9, 1)]

    id = e3nn.from_chunks("10x0e + 10x1e", [None, jnp.ones((1, 10, 3))], (1,))
    assert jax.tree_util.tree_map(
        jnp.shape, id.rechunk("5x0e + 5x0e + 5x1e + 5x1e").chunks
    ) == [
        None,
        None,
        (1, 5, 3),
        (1, 5, 3),
    ]

    id = e3nn.zeros("10x0e + 10x1e", ())
    id = id.rechunk("5x0e + 0x2e + 5x0e + 0x2e + 5x1e + 5x1e")

    a = e3nn.from_chunks(
        "            10x0e  +  0x0e +1x1e  +     0x0e    +          9x1e           + 0x0e",
        [
            jnp.ones((2, 10, 1)),
            None,
            None,
            jnp.ones((2, 0, 1)),
            jnp.ones((2, 9, 3)),
            None,
        ],
        (2,),
    )
    b = a.rechunk("5x0e + 0x2e + 5x0e + 0x2e + 5x1e + 5x1e")
    b = e3nn.from_chunks(b.irreps, b.chunks, b.shape[:-1])

    np.testing.assert_allclose(a.array, b.array)


def test_indexing():
    x = e3nn.IrrepsArray("2x0e + 1x0e", jnp.array([[1.0, 2, 3], [4.0, 5, 6]]))
    assert x.shape == (2, 3)
    np.testing.assert_allclose(x[0].array, jnp.array([1.0, 2, 3]))
    np.testing.assert_allclose(x[1, "1x0e"].array, jnp.array([6.0]))
    np.testing.assert_allclose(x[:, "1x0e"].array, jnp.array([[3.0], [6.0]]))
    np.testing.assert_allclose(x[..., "1x0e"].array, jnp.array([[3.0], [6.0]]))
    np.testing.assert_allclose(x[..., 1, "1x0e"].array, jnp.array([6.0]))
    np.testing.assert_allclose(x[..., 1, 2:].array, jnp.array([6.0]))
    np.testing.assert_allclose(x[..., 1, 2:3].array, jnp.array([6.0]))
    np.testing.assert_allclose(x[..., 1, -1:50].array, jnp.array([6.0]))
    np.testing.assert_allclose(x[..., 1, "2x0e + 1x0e"].array, jnp.array([4, 5, 6.0]))
    np.testing.assert_allclose(x[..., :1].array, jnp.array([[1.0], [4.0]]))
    np.testing.assert_allclose(x[..., 1:].array, jnp.array([[2, 3], [5.0, 6]]))

    x = e3nn.IrrepsArray("2x0e + 1x2e", jnp.arange(3 * 4 * 7).reshape((3, 4, 7)))
    np.testing.assert_allclose(x[..., 1, -5:].array, x[:3, 1, "2e"].array)

    x = e3nn.zeros("2x1e + 2x1e", (3, 3))
    with pytest.raises(IndexError):
        x[..., "2x1e"]

    with pytest.raises(IndexError):
        x[..., :2]

    x = e3nn.IrrepsArray(
        "2x1e + 2x1e",
        jnp.array([0.1, 0.2, 0.3, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3]),
    )
    assert x[3:-3].irreps == "1e + 1e"
    np.testing.assert_allclose(x[3:-3].array, jnp.array([1.1, 1.2, 1.3, 2.1, 2.2, 2.3]))


def test_reductions():
    x = e3nn.IrrepsArray(
        "2x0e + 1x1e", jnp.array([[1.0, 2, 3, 4, 5], [4.0, 5, 6, 6, 6]])
    )
    assert e3nn.sum(x).irreps == "0e + 1e"
    np.testing.assert_allclose(e3nn.sum(x).array, jnp.array([12.0, 9, 10, 11]))
    np.testing.assert_allclose(
        e3nn.sum(x, axis=0).array, jnp.array([5.0, 7, 9, 10, 11])
    )
    np.testing.assert_allclose(
        e3nn.sum(x, axis=1).array, jnp.array([[3.0, 3, 4, 5], [9.0, 6, 6, 6]])
    )

    np.testing.assert_allclose(
        e3nn.mean(x, axis=1).array, jnp.array([[1.5, 3, 4, 5], [4.5, 6, 6, 6]])
    )


def test_operators():
    x = e3nn.IrrepsArray(
        "2x0e + 1x1e", jnp.array([[1.0, 2, 3, 4, 5], [4.0, 5, 6, 6, 6]])
    )
    y = e3nn.IrrepsArray(
        "2x0e + 1x1o", jnp.array([[1.0, 2, 3, 4, 5], [4.0, 5, 6, 6, 6]])
    )

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

    x * jnp.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        1.0 / x

    1.0 / e3nn.norm(x)

    jax.config.update("jax_enable_x64", True)
    np.testing.assert_allclose(e3nn.norm(x / e3nn.norm(x)).array, 1)
    jax.config.update("jax_enable_x64", False)


def test_at_set():
    x = e3nn.IrrepsArray("0e + 1e", jnp.arange(3 * 4 * 4).reshape((3, 4, 4)))

    y = x.at[0, 1].set(0)
    assert y.shape == x.shape
    np.testing.assert_allclose(y[0, 1].array, 0)
    np.testing.assert_allclose(y[0, 1].chunks[0], 0)
    np.testing.assert_allclose(y[0, 1].chunks[1], 0)
    np.testing.assert_allclose(y[0, 2].array, x[0, 2].array)
    np.testing.assert_allclose(y[0, 2].chunks[0], x[0, 2].chunks[0])
    np.testing.assert_allclose(y[0, 2].chunks[1], x[0, 2].chunks[1])

    v = e3nn.IrrepsArray("0e + 1e", jnp.arange(4 * 4).reshape((4, 4)))
    y = x.at[1].set(v)
    assert y.shape == x.shape
    np.testing.assert_allclose(y[1].array, v.array)
    np.testing.assert_allclose(y[1].chunks[0], v.chunks[0])
    np.testing.assert_allclose(y[1].chunks[1], v.chunks[1])
    np.testing.assert_allclose(y[0].array, x[0].array)
    np.testing.assert_allclose(y[0].chunks[0], x[0].chunks[0])
    np.testing.assert_allclose(y[0].chunks[1], x[0].chunks[1])


def test_at_add():
    def f(*shape):
        return 1.0 + jnp.arange(prod(shape)).reshape(shape)

    x = e3nn.from_chunks(
        "1e + 0e + 0e + 0e",
        [None, None, f(2, 1, 1), f(2, 1, 1)],
        (2,),
    )
    v = e3nn.from_chunks("1e + 0e + 0e + 0e", [None, f(1, 1), None, f(1, 1)], ())
    y1 = x.at[0].add(v)
    y2 = e3nn.IrrepsArray(x.irreps, x.array.at[0].add(v.array))
    np.testing.assert_array_equal(y1.array, y2.array)
    assert y1.chunks[0] is None
    assert y1.chunks[1] is not None
    assert y1.chunks[2] is not None
    assert y1.chunks[3] is not None
    np.testing.assert_allclose(0, y2.chunks[0])
    np.testing.assert_array_equal(y1.chunks[1], y2.chunks[1])
    np.testing.assert_array_equal(y1.chunks[2], y2.chunks[2])
    np.testing.assert_array_equal(y1.chunks[3], y2.chunks[3])


def test_slice_by_mul():
    x = e3nn.IrrepsArray("3x0e + 4x1e", jnp.arange(3 + 4 * 3))
    y = x.slice_by_mul[2:4]
    assert y.irreps == "0e + 1e"
    np.testing.assert_allclose(y.array, jnp.array([2.0, 3.0, 4.0, 5.0]))

    y = x.slice_by_mul[:0]
    assert y.irreps == ""
    assert y.array.shape == (0,)
    assert len(y.chunks) == 0


def test_norm():
    x = e3nn.IrrepsArray(
        "2x0e + 1x1e", jnp.array([[1.0, 2, 3, 4, 5], [4.0, 5, 6, 6, 6]])
    )

    assert e3nn.norm(x).shape == (2, 3)
    assert e3nn.norm(x, per_irrep=True).shape == (2, 3)
    assert e3nn.norm(x, per_irrep=False).shape == (2, 1)

    x = e3nn.from_chunks("2x0e + 1x1e", [None, None], (2,), dtype=jnp.complex64)

    assert e3nn.norm(x).shape == (2, 3)


def test_dot():
    x = e3nn.IrrepsArray(
        "2x0e + 1x1e", jnp.array([[1.0, 2, 3, 4, 5], [4.0, 5, 6, 6, 6]])
    )
    y = e3nn.IrrepsArray(
        "2x0e + 1x1e", jnp.array([[1.0j, 2, 3, 4, 5], [4.0, 5, 6, 6, 6]])
    )

    assert e3nn.dot(x, y).shape == (2, 1)
    assert e3nn.dot(x, y, per_irrep=True).shape == (2, 3)
    assert e3nn.dot(x, y, per_irrep=False).shape == (2, 1)

    y = e3nn.from_chunks("2x0e + 1x1e", [None, None], (2,), dtype=jnp.complex64)

    assert e3nn.dot(x, y).shape == (2, 1)
