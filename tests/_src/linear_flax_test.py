from typing import Tuple

import jax
import jax.numpy as jnp
import flax
import pytest

import e3nn_jax as e3nn
from e3nn_jax.utils import assert_output_dtype_matches_input_dtype


@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"]
)
def test_linear_vanilla(keys, irreps_in, irreps_out):
    linear = e3nn.flax.Linear(irreps_in=irreps_in, irreps_out=irreps_out)
    x = e3nn.normal(irreps_in, next(keys), (128,))
    w = linear.init(next(keys), x)
    y = linear.apply(w, x)
    assert jnp.all(y.array != 0.0)  # unaccessible irreps are discarded
    assert y.shape == (128, y.irreps.dim)


@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "initializer", ["uniform", "custom"]
)
def test_linear_vanilla_custom_initializer(keys, irreps_in, irreps_out, initializer):
    if initializer == "uniform":
        def parameter_initializer(
            weight_std: float,
        ) -> jax.nn.initializers.Initializer:
            del weight_std
            return flax.linen.initializers.uniform(0.1)
    
    elif initializer == "custom":
        def parameter_initializer(
            weight_std: float,
        ) -> jax.nn.initializers.Initializer:
            del weight_std
            def custom_initializer(rng, shape, dtype=jnp.float32):
                return 5 + jax.random.normal(rng, shape, dtype=dtype) * 0.1
            return custom_initializer

    linear = e3nn.flax.Linear(irreps_in=irreps_in, irreps_out=irreps_out, parameter_initializer=parameter_initializer)
    x = e3nn.normal(irreps_in, next(keys), (128,))
    w = linear.init(next(keys), x)
    y = linear.apply(w, x)
    assert jnp.all(y.array != 0.0)  # unaccessible irreps are discarded
    assert y.shape == (128, y.irreps.dim)


@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"]
)
def test_linear_vanilla_with_forced_irreps_out(keys, irreps_in, irreps_out):
    linear = e3nn.flax.Linear(
        irreps_in=irreps_in, irreps_out=irreps_out, force_irreps_out=True
    )
    x = e3nn.normal(irreps_in, next(keys), (128,))
    w = linear.init(next(keys), x)
    y = linear.apply(w, x)
    assert y.irreps == irreps_out
    assert y.shape == (128, e3nn.Irreps(irreps_out).dim)


@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"]
)
def test_linear_indexed(keys, irreps_in, irreps_out):
    linear = e3nn.flax.Linear(
        irreps_in=irreps_in, irreps_out=irreps_out, num_indexed_weights=10
    )
    x = e3nn.normal(irreps_in, next(keys), (128,))
    i = jnp.arange(128) % 10
    w = linear.init(next(keys), i, x)
    y = linear.apply(w, i, x)
    assert jnp.all(y.array != 0.0)  # unaccessible irreps are discarded
    assert y.shape == (128, y.irreps.dim)


@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"]
)
def test_linear_mixed(keys, irreps_in, irreps_out):
    linear = e3nn.flax.Linear(irreps_in=irreps_in, irreps_out=irreps_out)
    x = e3nn.normal(irreps_in, next(keys), (128,))
    e = jax.random.normal(next(keys), (128, 10))
    w = linear.init(next(keys), e, x)
    y = linear.apply(w, e, x)
    assert jnp.all(y.array != 0.0)  # unaccessible irreps are discarded
    assert y.shape == (128, y.irreps.dim)


@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"]
)
def test_linear_mixed_per_channel(keys, irreps_in, irreps_out):
    linear = e3nn.flax.Linear(
        irreps_in=irreps_in, irreps_out=irreps_out, weights_per_channel=True
    )
    x = e3nn.normal(irreps_in, next(keys), (128,))
    e = jax.random.normal(next(keys), (10,))
    w = jax.jit(linear.init)(next(keys), e, x)
    y = linear.apply(w, e, x)
    assert jnp.all(y.array != 0.0)  # unaccessible irreps are discarded
    assert y.shape == (128, y.irreps.dim)


@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"]
)
def test_linear_dtype(keys, irreps_in, irreps_out):
    jax.config.update("jax_enable_x64", True)

    linear = e3nn.flax.Linear(irreps_out)
    x = e3nn.normal(irreps_in, next(keys), (128,))
    w = linear.init(next(keys), x)

    assert_output_dtype_matches_input_dtype(linear.apply, w, x)
