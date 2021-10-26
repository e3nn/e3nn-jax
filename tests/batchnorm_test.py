import jax
import jax.numpy as jnp
import random
from sys import maxsize
import haiku as hk
from e3nn_jax import BatchNorm, Irreps
from e3nn_jax.util.test import assert_equivariant
import pytest


def test_equivariant():
    rng_key = jax.random.PRNGKey(random.randint(0,maxsize))
    rng_key, *sub_keys = jax.random.split(rng_key, num=6)
    irreps = Irreps("3x0e + 3x0o + 4x1e")

    @hk.without_apply_rng
    @hk.transform_with_state
    def b(x, is_training=True):
        m = BatchNorm(irreps)
        return m(x, is_training)

    params, state = b.init(rng_key, irreps.randn(sub_keys[0], (16, -1)))
    out, state = b.apply(params, state, irreps.randn(sub_keys[1], (16, -1)))
    out, state = b.apply(params, state, irreps.randn(sub_keys[2], (16, -1)))

    m_train = lambda x: b.apply(params, state, x)[0]
    assert_equivariant(m_train, sub_keys[3], irreps_in=[irreps], irreps_out=[irreps])
    m_eval = lambda x: b.apply(params, state, x, is_training=False)[0]
    assert_equivariant(m_eval, sub_keys[4], irreps_in=[irreps], irreps_out=[irreps])


@pytest.mark.parametrize('affine', [True, False])
@pytest.mark.parametrize('reduce', ['mean', 'max'])
@pytest.mark.parametrize('normalization', ['norm', 'component'])
@pytest.mark.parametrize('instance', [True, False])
def test_modes(affine, reduce, normalization, instance):
    irreps = Irreps("10x0e + 5x1e")
    rng_key = jax.random.PRNGKey(random.randint(0, maxsize))
    rng_key, *sub_keys = jax.random.split(rng_key, num=4)

    @hk.without_apply_rng
    @hk.transform_with_state
    def b(x, is_training=True):
        m = BatchNorm(irreps, affine=affine, reduce=reduce, normalization=normalization, instance=instance)
        return m(x, is_training)

    params, state = b.init(rng_key, irreps.randn(sub_keys[0], (16, -1)))

    m_train = lambda x: b.apply(params, state, x)[0]
    m_eval = lambda x: b.apply(params, state, x, is_training=False)[0]

    m_train(irreps.randn(sub_keys[1], (20, 20, -1)))

    m_eval(irreps.randn(sub_keys[2], (20, 20, -1)))


@pytest.mark.parametrize('float_tolerance,instance', [(1e-3,True), (1e-3,False)])
def test_normalization(float_tolerance, instance):
    sqrt_float_tolerance = jnp.sqrt(float_tolerance)
    rng_key = jax.random.PRNGKey(random.randint(0, maxsize))
    rng_key, *sub_keys = jax.random.split(rng_key, num=6)

    batch, n = 20, 20
    irreps = Irreps("3x0e + 4x1e")

    @hk.without_apply_rng
    @hk.transform_with_state
    def b(x, is_training=True):
        m = BatchNorm(irreps, normalization='norm', instance=instance)
        return m(x, is_training)

    params, state = b.init(rng_key, irreps.randn(sub_keys[0], (16, -1)))

    x = jax.random.normal(sub_keys[1], (batch, n, irreps.dim)) * 5 + 10
    x, state = b.apply(params, state, x)

    a = x[..., :3]  # [batch, space, mul]
    assert jnp.max(jnp.abs(a.mean([0, 1]))) < float_tolerance
    assert jnp.max(jnp.abs(jnp.square(a).mean([0, 1]) - 1)) < sqrt_float_tolerance

    a = x[..., 3:].reshape(batch, n, 4, 3)  # [batch, space, mul, repr]
    assert jnp.max(jnp.abs(jnp.square(a).sum(3).mean([0, 1]) - 1)) < sqrt_float_tolerance

    @hk.without_apply_rng
    @hk.transform_with_state
    def b(x, is_training=True):
        m = BatchNorm(irreps, normalization='component', instance=instance)
        return m(x, is_training)

    params, state = b.init(sub_keys[2], irreps.randn(sub_keys[3], (16, -1)))

    x = jax.random.normal(sub_keys[4], (batch, n, irreps.dim)) * 5 + 10.0
    x, state = b.apply(params, state, x)

    a = x[..., :3]  # [batch, space, mul]
    assert jnp.max(jnp.abs(a.mean([0, 1]))) < float_tolerance
    assert jnp.max(jnp.abs(jnp.square(a).mean([0, 1]) - 1)) < sqrt_float_tolerance

    a = x[..., 3:].reshape(batch, n, 4, 3)  # [batch, space, mul, repr]
    assert jnp.max(jnp.abs(jnp.square(a).mean(3).mean([0, 1]) - 1)) < sqrt_float_tolerance
