import jax.numpy as jnp
import numpy as np
import pytest
from e3nn_jax.experimental.voxel_pooling import zoom


@pytest.mark.parametrize("interpolation", ["nearest", "linear"])
def test_zoom(interpolation):
    x = jnp.ones((16, 16, 16))
    y = zoom(x, output_size=(8, 8, 8), interpolation=interpolation)
    np.testing.assert_allclose(y, jnp.ones((8, 8, 8)))


@pytest.mark.parametrize("interpolation", ["nearest", "linear"])
def test_zoom_identity(interpolation):
    x = jnp.arange(8**3).reshape((8, 8, 8))
    y = zoom(x, output_size=(8, 8, 8), interpolation=interpolation)
    np.testing.assert_allclose(x, y)

    x = jnp.arange(2**3).reshape((2, 2, 2))
    y = zoom(x, output_size=(32, 24, 18), interpolation=interpolation)
    y = zoom(y, output_size=(2, 2, 2), interpolation=interpolation)
    np.testing.assert_allclose(x, y, rtol=0.1, atol=0.1)
