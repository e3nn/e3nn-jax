import jax.numpy as jnp
import pytest
import e3nn_jax as e3nn


def test_config(keys):
    e3nn.config("irrep_normalization", "component")
    assert e3nn.config("irrep_normalization") == "component"

    inputs = e3nn.normal("10x1e", keys[0], (2,))
    s, p, d = e3nn.elementwise_tensor_product(inputs[0], inputs[1]).chunks

    assert jnp.exp(jnp.abs(jnp.log(jnp.mean(p**2)))) < 2.0
    assert jnp.exp(jnp.abs(jnp.log(jnp.mean(d**2)))) < 2.0

    e3nn.config("irrep_normalization", "norm")
    assert e3nn.config("irrep_normalization") == "norm"

    inputs = e3nn.normal("10x1e", keys[0], (2,))
    s, p, d = e3nn.elementwise_tensor_product(inputs[0], inputs[1]).chunks

    assert jnp.exp(jnp.abs(jnp.log(jnp.mean(jnp.sum(p**2, axis=1))))) < 2.0
    assert jnp.exp(jnp.abs(jnp.log(jnp.mean(jnp.sum(d**2, axis=1))))) < 2.0

    with pytest.raises(ValueError):
        e3nn.config("this is not a valid name", 1)
