import e3nn_jax as e3nn
import jax.numpy as jnp


def test_radius_graph():
    pos = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    r_max = 1.01
    batch = jnp.array([0, 0, 0, 0])

    src, dst = e3nn.radius_graph(pos, r_max, batch=batch)
    assert src.shape == (6,)
    assert dst.shape == (6,)

    src, dst = e3nn.radius_graph(pos, r_max, batch=batch, loop=True)
    assert src.shape == (6 + 4,)
    assert dst.shape == (6 + 4,)

    src, dst = e3nn.radius_graph(
        pos, r_max, batch=batch, size=12, fill_src=-1, fill_dst=-1
    )
    assert src.shape == (12,)
    assert dst.shape == (12,)
