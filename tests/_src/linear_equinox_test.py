import jax
import jax.numpy as jnp
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
    linear = e3nn.equinox.Linear(
        irreps_in=irreps_in, irreps_out=irreps_out, key=next(keys)
    )
    x = e3nn.normal(irreps_in, next(keys), (128,))
    y = linear(x)
    assert jnp.all(y.array != 0.0)  # unaccessible irreps are discarded
    assert y.shape == (128, y.irreps.dim)


@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize("num_channels", [1, 2, 8])
def test_linear_vanilla_with_channels(keys, irreps_in, irreps_out, num_channels):
    linear = e3nn.equinox.Linear(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        key=next(keys),
        channel_out=num_channels,
    )
    x = e3nn.normal(irreps_in, next(keys), (128,))
    y = linear(x)
    assert jnp.all(y.array != 0.0)  # unaccessible irreps are discarded
    assert y.shape == (128, num_channels, y.irreps.dim)


@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"]
)
def test_linear_vanilla_with_forced_irreps_out(keys, irreps_in, irreps_out):
    linear = e3nn.equinox.Linear(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        key=next(keys),
        force_irreps_out=True,
    )
    x = e3nn.normal(irreps_in, next(keys), (128,))
    y = linear(x)
    assert y.irreps == irreps_out
    assert y.shape == (128, e3nn.Irreps(irreps_out).dim)


@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"]
)
def test_linear_indexed(keys, irreps_in, irreps_out):
    linear = e3nn.equinox.Linear(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        num_indexed_weights=10,
        key=next(keys),
        linear_type="indexed",
    )
    x = e3nn.normal(irreps_in, next(keys), (128,))
    i = jnp.arange(128) % 10
    y = linear(i, x)
    assert jnp.all(y.array != 0.0)  # unaccessible irreps are discarded
    assert y.shape == (128, y.irreps.dim)


@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"]
)
def test_linear_mixed(keys, irreps_in, irreps_out):
    linear = e3nn.equinox.Linear(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        key=next(keys),
        linear_type="mixed",
        weights_dim=10,
    )
    x = e3nn.normal(irreps_in, next(keys), (128,))
    e = jax.random.normal(next(keys), (128, 10))
    y = linear(e, x)
    assert jnp.all(y.array != 0.0)  # unaccessible irreps are discarded
    assert y.shape == (128, y.irreps.dim)


@pytest.mark.parametrize(
    "irreps_in", ["5x0e", "1e + 2e + 4x1e + 3x3o", "2x1o + 0x3e", "0x0e"]
)
@pytest.mark.parametrize(
    "irreps_out", ["5x0e", "1e + 2e + 3x3o + 3x1e", "2x1o + 0x3e", "0x0e"]
)
def test_linear_mixed_per_channel(keys, irreps_in, irreps_out):
    linear = e3nn.equinox.Linear(
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        weights_per_channel=True,
        weights_dim=10,
        channel_in=128,
        key=next(keys),
        linear_type="mixed_per_channel",
    )
    x = e3nn.normal(irreps_in, next(keys), (128,))
    e = jax.random.normal(next(keys), (10,))
    y = linear(e, x)
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

    linear = e3nn.equinox.Linear(
        irreps_in=irreps_in, irreps_out=irreps_out, key=next(keys)
    )
    x = e3nn.normal(irreps_in, next(keys), (128,))

    assert_output_dtype_matches_input_dtype(linear, x)
