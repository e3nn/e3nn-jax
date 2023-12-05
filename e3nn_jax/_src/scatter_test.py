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

    x = jnp.array([1.0, 2.0, 1.0, 0.5, 0.5, 0.7, 0.2, 0.1])
    nel = jnp.array([3, 2, 3])
    np.testing.assert_allclose(  # nel
        e3nn.scatter_sum(x, nel=nel),
        jnp.array([4.0, 1.0, 1.0]),
    )

    np.testing.assert_allclose(  # nel + map_back
        e3nn.scatter_sum(x, nel=nel, map_back=True),
        jnp.array([4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )

    i = jnp.array([[0, 2], [2, 0]])
    x = jnp.array([[[1.0, 0.0], [2.0, 1.0]], [[3.0, 0.0], [-10.0, -1.0]]])
    np.testing.assert_allclose(
        e3nn.scatter_sum(x, dst=i, output_size=3),
        jnp.array([[-9.0, -1.0], [0.0, 0.0], [5.0, 1.0]]),
    )


def test_scatter_mean():
    x = jnp.array([[2.0, 3.0], [0.0, 3.0], [-10.0, 42.0]])
    dst = jnp.array([[0, 2], [2, 2], [0, 1]])

    np.testing.assert_allclose(  # dst
        e3nn.scatter_mean(x, dst=dst, output_size=3),
        jnp.array([-4.0, 42.0, 2.0]),
    )

    np.testing.assert_allclose(  # map_back
        e3nn.scatter_mean(x, dst=dst, map_back=True),
        jnp.array([[-4.0, 2.0], [2.0, 2.0], [-4.0, 42.0]]),
    )

    x = jnp.array([10.0, 1.0, 2.0, 3.0])
    nel = jnp.array([1, 0, 3])
    np.testing.assert_allclose(  # nel
        e3nn.scatter_mean(x, nel=nel),
        jnp.array([10.0, 0.0, 2.0]),
    )

    np.testing.assert_allclose(  # nel + map_back
        e3nn.scatter_mean(x, nel=nel, map_back=True),
        jnp.array([10.0, 2.0, 2.0, 2.0]),
    )


def test_scatter_mean_irreps_array():
    x = e3nn.IrrepsArray(
        "0e", jnp.array([[[2.0], [3.0]], [[0.0], [3.0]], [[-10.0], [42.0]]])
    )
    dst = jnp.array([[0, 2], [2, 2], [0, 1]])

    np.testing.assert_allclose(  # dst
        e3nn.scatter_mean(x, dst=dst, output_size=3).array,
        jnp.array([-4.0, 42.0, 2.0])[..., None],
    )

    np.testing.assert_allclose(  # map_back
        e3nn.scatter_mean(x, dst=dst, map_back=True).array,
        jnp.array([[-4.0, 2.0], [2.0, 2.0], [-4.0, 42.0]])[..., None],
    )

    x = e3nn.IrrepsArray("0e", jnp.array([[10.0], [1.0], [2.0], [3.0]]))
    nel = jnp.array([1, 0, 3])
    np.testing.assert_allclose(  # nel
        e3nn.scatter_mean(x, nel=nel).array, jnp.array([10.0, 0.0, 2.0])[..., None]
    )

    np.testing.assert_allclose(  # nel + map_back
        e3nn.scatter_mean(x, nel=nel, map_back=True).array,
        jnp.array([10.0, 2.0, 2.0, 2.0])[..., None],
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
