import jax
import jax.numpy as jnp
import numpy as np

import e3nn_jax as e3nn


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
