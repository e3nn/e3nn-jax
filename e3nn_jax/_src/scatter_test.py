import jax
import jax.numpy as jnp
import numpy as np

import e3nn_jax as e3nn


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

    x = e3nn.IrrepsArray("0e", x[:, None])
    np.testing.assert_allclose(  # map_back
        e3nn.scatter_sum(x, dst=i, map_back=True).array,
        jnp.array([-9.0, 5.0, 5.0, -9.0])[:, None],
    )

    np.testing.assert_allclose(  # nel
        e3nn.scatter_sum(
            jnp.array([1.0, 2.0, 1.0, 0.5, 0.5, 0.7, 0.2, 0.1]),
            nel=jnp.array([3, 2, 3]),
        ),
        jnp.array([4.0, 1.0, 1.0]),
    )


def test_scatter_max():
    jax.config.update("jax_debug_infs", False)

    i = jnp.array([0, 2, 2, 0])
    x = jnp.array([1.0, 2.0, 3.0, -10.0])
    np.testing.assert_allclose(
        e3nn.scatter_max(x, dst=i, output_size=3),
        jnp.array([1.0, -np.inf, 3.0]),
    )

    np.testing.assert_allclose(  # map_back
        e3nn.scatter_max(x, dst=i, map_back=True),
        jnp.array([1.0, 3.0, 3.0, 1.0]),
    )

    np.testing.assert_allclose(  # nel
        e3nn.scatter_max(
            jnp.array([-1.0, -2.0, -1.0, 0.5, 0.5, 0.7, 0.2, 0.1]),
            nel=jnp.array([3, 2, 3]),
        ),
        jnp.array([-1.0, 0.5, 0.7]),
    )
