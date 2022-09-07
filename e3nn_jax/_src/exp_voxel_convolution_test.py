import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from e3nn_jax.experimental.voxel_convolution import Convolution


def test_convolution(keys):
    irreps_in = e3nn.Irreps("2x0e + 3x1e + 2x2e")
    irreps_out = e3nn.Irreps("0e + 2x1e + 2e")
    irreps_sh = e3nn.Irreps("0e + 1e + 2e")

    @hk.without_apply_rng
    @hk.transform
    def c(x, z):
        x = Convolution(
            irreps_out=irreps_out,
            irreps_sh=irreps_sh,
            diameter=3.9,
            num_radial_basis={0: 3, 1: 2, 2: 1},
            relative_starts={0: 0.0, 1: 0.0, 2: 0.5},
            steps=((1.0, 1.0, 1.0), z),
        )(x)
        return x

    f = jax.jit(c.apply)

    x0 = e3nn.normal(irreps_in, next(keys), (3, 8, 8, 8))
    x0 = jax.tree_util.tree_map(lambda x: jnp.pad(x, ((0, 0), (4, 4), (4, 4), (4, 4)) + ((0, 0),) * (x.ndim - 4)), x0)

    w = c.init(next(keys), x0, jnp.array([1.0, 1.0, 1.0]))

    y0 = f(w, x0, jnp.array([1.0, 1.02, 0.98]))
    y2 = jax.tree_util.tree_map(lambda x: jnp.rot90(x, axes=(2, 3)), y0)
    y2 = y2.transform_by_angles(0.0, jnp.pi / 2, 0.0)

    x1 = jax.tree_util.tree_map(lambda x: jnp.rot90(x, axes=(2, 3)), x0)
    x1 = x1.transform_by_angles(0.0, jnp.pi / 2, 0.0)
    y1 = f(w, x1, jnp.array([1.0, 0.98, 1.02]))

    np.testing.assert_allclose(y1.array, y2.array, atol=1e-5)

    y1 = f(w, x1, jnp.array([1.0, 1.02, 0.98]))
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(y1.array, y2.array, atol=1e-5)


def test_convolution_defaults(keys):
    irreps_in = e3nn.Irreps("2x0e + 3x1e + 2x2e")
    irreps_out = e3nn.Irreps("0e + 2x1e + 2e")
    irreps_sh = e3nn.Irreps("0e + 1e + 2e")

    @hk.without_apply_rng
    @hk.transform
    def c(x):
        x = Convolution(
            irreps_out=irreps_out,
            irreps_sh=irreps_sh,
            diameter=3.9,
            num_radial_basis=3,
            steps=(1.0, 1.0, 1.0),
        )(x)
        return x

    f = jax.jit(c.apply)

    x0 = e3nn.normal(irreps_in, next(keys), (3, 8, 8, 8))
    x0 = jax.tree_util.tree_map(lambda x: jnp.pad(x, ((0, 0), (4, 4), (4, 4), (4, 4)) + ((0, 0),) * (x.ndim - 4)), x0)

    w = c.init(next(keys), x0)
    y0 = f(w, x0)

    x1 = jax.tree_util.tree_map(lambda x: jnp.rot90(x, axes=(2, 3)), x0)
    x1 = x1.transform_by_angles(0.0, jnp.pi / 2, 0.0)
    y1 = f(w, x1)

    y2 = jax.tree_util.tree_map(lambda x: jnp.rot90(x, axes=(2, 3)), y0)
    y2 = y2.transform_by_angles(0.0, jnp.pi / 2, 0.0)

    assert jnp.allclose(y1.array, y2.array, atol=1e-5)
