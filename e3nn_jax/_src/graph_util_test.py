import e3nn_jax as e3nn
import jax.numpy as jnp
import numpy as np


def test_scatter_sum():
    i = jnp.array([0, 2, 2, 0])
    x = jnp.array([1.0, 2.0, 3.0, -10.0])
    np.testing.assert_allclose(
        e3nn.scatter_sum(x, dst=i, output_size=3),
        jnp.array([-9.0, 0.0, 5.0]),
    )

    np.testing.assert_allclose(  # map_back
        e3nn.scatter_sum(x, dst=i, map_back=True),
        jnp.array([-9.0, 5.0, 5.0, -9.0]),
    )

    np.testing.assert_allclose(  # nel
        e3nn.scatter_sum(jnp.array([1.0, 2.0, 1.0, 0.5, 0.5, 0.7, 0.2, 0.1]), nel=jnp.array([3, 2, 3])),
        jnp.array([4.0, 1.0, 1.0]),
    )


def test_radius_graph():
    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    r_max = 1.01
    batch = jnp.array([0, 0, 0, 0])

    src, dst = e3nn.radius_graph(pos, r_max, batch=batch)
    assert src.shape == (6,)
    assert dst.shape == (6,)

    src, dst = e3nn.radius_graph(pos, r_max, batch=batch, loop=True)
    assert src.shape == (6 + 4,)
    assert dst.shape == (6 + 4,)

    src, dst = e3nn.radius_graph(pos, r_max, batch=batch, size=12, fill_src=-1, fill_dst=-1)
    assert src.shape == (12,)
    assert dst.shape == (12,)
