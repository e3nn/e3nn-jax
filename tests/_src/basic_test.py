import jax.numpy as jnp
import numpy as np

import e3nn_jax as e3nn


def assert_array_equals_chunks(x: e3nn.IrrepsArray):
    y = e3nn.from_chunks(x.irreps, x.chunks, x.shape[:-1])
    np.testing.assert_array_equal(x.array, y.array)


def test_concatenate1(keys):
    x1 = e3nn.normal("0e + 1e", keys[0], (3,))
    x2 = e3nn.normal("0e + 1e", keys[0], (2,))
    x3 = e3nn.normal("0e + 1e", keys[0], (1,))

    y = e3nn.concatenate([x1, x2, x3], axis=0)
    assert y.shape == (6, 4)
    assert y.irreps == "0e + 1e"
    assert_array_equals_chunks(y)


def test_concatenate2(keys):
    x1 = e3nn.normal("0e + 1e", keys[0], (2,))
    x2 = e3nn.normal("0e", keys[0], (2,))
    x3 = e3nn.normal("0e + 1e", keys[0], (2,))

    y = e3nn.concatenate([x1, x2, x3], axis=1)
    assert y.shape == (2, 9)
    assert y.irreps == "0e + 1e + 0e + 0e + 1e"
    assert_array_equals_chunks(y)


def test_concatenate3():
    x1 = e3nn.from_chunks("0e + 1e", [None, jnp.zeros((1, 1, 3))], (1,))
    x2 = e3nn.from_chunks("0e + 1e", [None, jnp.ones((1, 1, 3))], (1,))
    x3 = e3nn.from_chunks("0e + 1e", [None, jnp.zeros((2, 1, 3))], (2,))

    y = e3nn.concatenate([x1, x2, x3], axis=0)
    assert y.shape == (4, 4)
    assert y.irreps == "0e + 1e"
    assert_array_equals_chunks(y)
    assert y.zero_flags == (True, False)


def test_stack1(keys):
    x1 = e3nn.normal("0e + 1e", keys[0], (2,))
    x2 = e3nn.normal("0e + 1e", keys[0], (2,))
    x3 = e3nn.normal("0e + 1e", keys[0], (2,))

    y = e3nn.stack([x1, x2, x3], axis=0)
    assert y.shape == (3, 2, 4)
    assert y.irreps == "0e + 1e"
    assert_array_equals_chunks(y)


def test_stack2():
    x1 = e3nn.from_chunks("0e + 1e", [None, jnp.zeros((1, 1, 3))], (1,))
    x2 = e3nn.from_chunks("0e + 1e", [None, jnp.ones((1, 1, 3))], (1,))
    x3 = e3nn.from_chunks("0e + 1e", [None, jnp.zeros((1, 1, 3))], (1,))

    y = e3nn.stack([x1, x2, x3], axis=0)
    assert y.shape == (3, 1, 4)
    assert y.irreps == "0e + 1e"
    assert_array_equals_chunks(y)
    assert y.zero_flags == (True, False)
