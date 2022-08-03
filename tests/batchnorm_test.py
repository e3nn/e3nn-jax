import jax
import jax.numpy as jnp
import haiku as hk
from e3nn_jax import BatchNorm, Irreps
from e3nn_jax.util.test import assert_equivariant
import pytest


@pytest.mark.parametrize("irreps", [Irreps("3x0e + 3x0o + 4x1e"), Irreps("3x0o + 3x0e + 4x1e")])
def test_equivariant(keys, irreps):
    @hk.without_apply_rng
    @hk.transform_with_state
    def b(x, is_training=True):
        m = BatchNorm(irreps=irreps)
        return m(x, is_training).array

    params, state = b.init(next(keys), irreps.randn(next(keys), (16, -1)))
    _, state = b.apply(params, state, irreps.randn(next(keys), (16, -1)))
    _, state = b.apply(params, state, irreps.randn(next(keys), (16, -1)))

    m_train = lambda x: b.apply(params, state, x)[0]
    assert_equivariant(m_train, next(keys), irreps_in=[irreps], irreps_out=[irreps])
    m_eval = lambda x: b.apply(params, state, x, is_training=False)[0]
    assert_equivariant(m_eval, next(keys), irreps_in=[irreps], irreps_out=[irreps])


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("reduce", ["mean", "max"])
@pytest.mark.parametrize("normalization", ["norm", "component"])
@pytest.mark.parametrize("instance", [True, False])
def test_modes(keys, affine, reduce, normalization, instance):
    irreps = Irreps("10x0e + 5x1e")

    @hk.without_apply_rng
    @hk.transform_with_state
    def b(x, is_training=True):
        m = BatchNorm(irreps=irreps, affine=affine, reduce=reduce, normalization=normalization, instance=instance)
        return m(x, is_training)

    params, state = b.init(next(keys), irreps.randn(next(keys), (16, -1)))

    m_train = lambda x: b.apply(params, state, x)[0]
    m_eval = lambda x: b.apply(params, state, x, is_training=False)[0]

    m_train(irreps.randn(next(keys), (20, 20, -1)))

    m_eval(irreps.randn(next(keys), (20, 20, -1)))


@pytest.mark.parametrize("instance", [True, False])
def test_normalization(keys, instance):
    float_tolerance = 1e-3
    sqrt_float_tolerance = jnp.sqrt(float_tolerance)

    batch, n = 20, 20
    irreps = Irreps("3x0e + 4x1e")

    @hk.without_apply_rng
    @hk.transform_with_state
    def b(x, is_training=True):
        m = BatchNorm(irreps=irreps, normalization="norm", instance=instance)
        return m(x, is_training)

    params, state = b.init(next(keys), irreps.randn(next(keys), (16, -1)))

    x = jax.random.normal(next(keys), (batch, n, irreps.dim)) * 5 + 10
    x, state = b.apply(params, state, x)

    a = x.list[0]  # [batch, space, mul, 1]
    assert jnp.max(jnp.abs(a.mean([0, 1]))) < float_tolerance
    assert jnp.max(jnp.abs(jnp.square(a).mean([0, 1]) - 1)) < sqrt_float_tolerance

    a = x.list[1]  # [batch, space, mul, repr]
    assert jnp.max(jnp.abs(jnp.square(a).sum(3).mean([0, 1]) - 1)) < sqrt_float_tolerance

    @hk.without_apply_rng
    @hk.transform_with_state
    def b(x, is_training=True):
        m = BatchNorm(irreps=irreps, normalization="component", instance=instance)
        return m(x, is_training)

    params, state = b.init(next(keys), irreps.randn(next(keys), (16, -1)))

    x = jax.random.normal(next(keys), (batch, n, irreps.dim)) * 5 + 10.0
    x, state = b.apply(params, state, x)

    a = x.list[0]  # [batch, space, mul, 1]
    assert jnp.max(jnp.abs(a.mean([0, 1]))) < float_tolerance
    assert jnp.max(jnp.abs(jnp.square(a).mean([0, 1]) - 1)) < sqrt_float_tolerance

    a = x.list[1]  # [batch, space, mul, repr]
    assert jnp.max(jnp.abs(jnp.square(a).mean(3).mean([0, 1]) - 1)) < sqrt_float_tolerance
