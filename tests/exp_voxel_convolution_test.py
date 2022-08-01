import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from e3nn_jax import Irreps, IrrepsArray
from e3nn_jax.experimental.voxel_convolution import Convolution


def test_convolution(keys):
    irreps_in = Irreps("2x0e + 3x1e + 2x2e")
    irreps_out = Irreps("0e + 2x1e + 2e")
    irreps_sh = Irreps("0e + 1e + 2e")

    @hk.without_apply_rng
    @hk.transform
    def c(x, z):
        x = IrrepsArray.from_contiguous(irreps_in, x)
        x = Convolution(
            irreps_out=irreps_out,
            irreps_sh=irreps_sh,
            diameter=3.9,
            num_radial_basis={0: 3, 1: 2, 2: 1},
            relative_starts={0: 0.0, 1: 0.0, 2: 0.5},
            steps=((1.0, 1.0, 1.0), z),
        )(x)
        return x.contiguous

    f = jax.jit(c.apply)

    x0 = irreps_in.randn(next(keys), (3, 8, 8, 8, -1))
    x0 = jnp.pad(x0, ((0, 0), (4, 4), (4, 4), (4, 4), (0, 0)))

    w = c.init(next(keys), x0, jnp.array([1.0, 1.0, 1.0]))

    y0 = f(w, x0, jnp.array([1.0, 1.02, 0.98]))
    y2 = jnp.rot90(y0, axes=(2, 3))
    y2 = irreps_out.transform_by_angles(y2, 0.0, jnp.pi / 2, 0.0)

    x1 = jnp.rot90(x0, axes=(2, 3))
    x1 = irreps_in.transform_by_angles(x1, 0.0, jnp.pi / 2, 0.0)
    y1 = f(w, x1, jnp.array([1.0, 0.98, 1.02]))

    np.testing.assert_allclose(y1, y2, atol=1e-5)

    y1 = f(w, x1, jnp.array([1.0, 1.02, 0.98]))
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(y1, y2, atol=1e-5)


def test_convolution_defaults(keys):
    irreps_in = Irreps("2x0e + 3x1e + 2x2e")
    irreps_out = Irreps("0e + 2x1e + 2e")
    irreps_sh = Irreps("0e + 1e + 2e")

    @hk.without_apply_rng
    @hk.transform
    def c(x):
        x = IrrepsArray.from_contiguous(irreps_in, x)
        x = Convolution(
            irreps_out=irreps_out,
            irreps_sh=irreps_sh,
            diameter=3.9,
            num_radial_basis=3,
            steps=(1.0, 1.0, 1.0),
        )(x)
        return x.contiguous

    f = jax.jit(c.apply)

    x0 = irreps_in.randn(next(keys), (3, 8, 8, 8, -1))
    x0 = jnp.pad(x0, ((0, 0), (4, 4), (4, 4), (4, 4), (0, 0)))

    w = c.init(next(keys), x0)
    y0 = f(w, x0)

    x1 = jnp.rot90(x0, axes=(2, 3))
    x1 = irreps_in.transform_by_angles(x1, 0.0, jnp.pi / 2, 0.0)
    y1 = f(w, x1)

    y2 = jnp.rot90(y0, axes=(2, 3))
    y2 = irreps_out.transform_by_angles(y2, 0.0, jnp.pi / 2, 0.0)

    assert jnp.allclose(y1, y2, atol=1e-5)
