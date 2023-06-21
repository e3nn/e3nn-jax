import haiku as hk
import jax.numpy as jnp
import pytest

import e3nn_jax as e3nn
from e3nn_jax.utils import assert_equivariant


@pytest.mark.parametrize(
    "irreps", [e3nn.Irreps("3x0e + 3x0o + 4x1e"), e3nn.Irreps("3x0o + 3x0e + 4x1e")]
)
def test_equivariant(keys, irreps):
    @hk.without_apply_rng
    @hk.transform_with_state
    def b(x, is_training=True):
        m = e3nn.haiku.BatchNorm(irreps=irreps)
        return m(x, is_training)

    params, state = b.init(next(keys), e3nn.normal(irreps, next(keys), (16,)))
    _, state = b.apply(params, state, e3nn.normal(irreps, next(keys), (16,)))
    _, state = b.apply(params, state, e3nn.normal(irreps, next(keys), (16,)))

    m_train = lambda x: b.apply(params, state, x)[0]
    assert_equivariant(m_train, next(keys), e3nn.normal(irreps, next(keys), (16,)))
    m_eval = lambda x: b.apply(params, state, x, is_training=False)[0]
    assert_equivariant(m_eval, next(keys), e3nn.normal(irreps, next(keys), (16,)))


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("reduce", ["mean", "max"])
@pytest.mark.parametrize("normalization", ["norm", "component"])
@pytest.mark.parametrize("instance", [True, False])
def test_modes(keys, affine, reduce, normalization, instance):
    irreps = e3nn.Irreps("10x0e + 5x1e")

    @hk.without_apply_rng
    @hk.transform_with_state
    def b(x, is_training=True):
        m = e3nn.haiku.BatchNorm(
            irreps=irreps,
            affine=affine,
            reduce=reduce,
            normalization=normalization,
            instance=instance,
        )
        return m(x, is_training)

    params, state = b.init(next(keys), e3nn.normal(irreps, next(keys), (20, 20)))

    m_train = lambda x: b.apply(params, state, x)[0]
    m_eval = lambda x: b.apply(params, state, x, is_training=False)[0]

    m_train(e3nn.normal(irreps, next(keys), (20, 20)))

    m_eval(e3nn.normal(irreps, next(keys), (20, 20)))


@pytest.mark.parametrize("instance", [True, False])
def test_normalization(keys, instance):
    float_tolerance = 1e-3
    sqrt_float_tolerance = jnp.sqrt(float_tolerance)

    batch, n = 20, 20
    irreps = e3nn.Irreps("3x0e + 4x1e")

    @hk.without_apply_rng
    @hk.transform_with_state
    def b(x, is_training=True):
        m = e3nn.haiku.BatchNorm(irreps=irreps, normalization="norm", instance=instance)
        return m(x, is_training)

    params, state = b.init(next(keys), e3nn.normal(irreps, next(keys), (16,)))

    x = e3nn.normal(irreps, next(keys), (batch, n)) * 5
    x, state = b.apply(params, state, x)

    a = x.chunks[0]  # [batch, space, mul, 1]
    assert jnp.max(jnp.abs(a.mean([0, 1]))) < float_tolerance
    assert jnp.max(jnp.abs(jnp.square(a).mean([0, 1]) - 1)) < sqrt_float_tolerance

    a = x.chunks[1]  # [batch, space, mul, repr]
    assert (
        jnp.max(jnp.abs(jnp.square(a).sum(3).mean([0, 1]) - 1)) < sqrt_float_tolerance
    )

    @hk.without_apply_rng
    @hk.transform_with_state
    def b(x, is_training=True):
        m = e3nn.haiku.BatchNorm(
            irreps=irreps, normalization="component", instance=instance
        )
        return m(x, is_training)

    params, state = b.init(next(keys), e3nn.normal(irreps, next(keys), (16,)))

    x = e3nn.normal(irreps, next(keys), (batch, n)) * 5
    x, state = b.apply(params, state, x)

    a = x.chunks[0]  # [batch, space, mul, 1]
    assert jnp.max(jnp.abs(a.mean([0, 1]))) < float_tolerance
    assert jnp.max(jnp.abs(jnp.square(a).mean([0, 1]) - 1)) < sqrt_float_tolerance

    a = x.chunks[1]  # [batch, space, mul, repr]
    assert (
        jnp.max(jnp.abs(jnp.square(a).mean(3).mean([0, 1]) - 1)) < sqrt_float_tolerance
    )
