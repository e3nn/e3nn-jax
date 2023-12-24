import jax.numpy as jnp
import numpy as np
import pytest

import e3nn_jax as e3nn
from e3nn_jax.experimental.linear_shtp import LinearSHTP, shtp
from e3nn_jax.utils import equivariance_test


@pytest.mark.parametrize("mix", [True, False])
def test_equivariance_linear(keys, mix: bool):
    conv = LinearSHTP("0e + 0o + 4x1o + 1e + 2e + 2o", mix)

    x = e3nn.normal("0e + 3x2o + 3x1e + 2x1o + 3e", next(keys))
    d = e3nn.IrrepsArray("1o", jnp.array([0.0, 1.0, 0.0]))

    w = conv.init(next(keys), x, d)

    def f(x, d):
        return conv.apply(w, x, d)

    z1, z2 = equivariance_test(
        f, next(keys), x, e3nn.IrrepsArray("1o", jnp.array([0.1, -0.2, -0.4]))
    )
    np.testing.assert_allclose(z1.array, z2.array, atol=1e-5)

    # Test Gimbal Lock
    z1, z2 = equivariance_test(
        e3nn.grad(f, 1),
        next(keys),
        x,
        e3nn.IrrepsArray("1o", jnp.array([0.0, 1.0, 0.0])),
    )
    np.testing.assert_allclose(z1.array, z2.array, atol=1e-5)

    if mix:
        assert f(x, d).irreps == conv.irreps_out


def test_equivariance(keys):
    x = e3nn.normal("0e + 3x2o + 3x1e + 2x1o + 3e", next(keys))

    def f(x, d):
        return shtp(x, d, "0e + 0o + 4x1o + 1e + 2e + 2o")

    z1, z2 = equivariance_test(
        f, next(keys), x, e3nn.IrrepsArray("1o", jnp.array([0.1, -0.2, -0.4]))
    )
    np.testing.assert_allclose(z1.array, z2.array, atol=1e-5)

    # Test Gimbal Lock
    z1, z2 = equivariance_test(
        e3nn.grad(f, 1),
        next(keys),
        x,
        e3nn.IrrepsArray("1o", jnp.array([0.0, 1.0, 0.0])),
    )
    np.testing.assert_allclose(z1.array, z2.array, atol=1e-5)
