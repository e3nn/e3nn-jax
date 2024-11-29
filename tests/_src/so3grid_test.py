import math

import jax
import jax.numpy as jnp
import pytest
from e3nn_jax import SO3Signal


def test_integrate_ones():
    sig = SO3Signal.from_function(
        lambda R: 1.0,
        res_beta=40,
        res_alpha=39,
        res_theta=40,
        quadrature="gausslegendre",
    )
    integral = sig.integrate()
    assert jnp.isclose(integral, 1.0)


@pytest.mark.parametrize("x", [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
def test_integrate_vector(x):
    x = jnp.array(x)
    sig = SO3Signal.from_function(
        lambda R: R @ x,
        res_beta=40,
        res_alpha=39,
        res_theta=40,
        quadrature="gausslegendre",
    )
    integral = sig.integrate()
    assert jnp.allclose(integral, 0.0, atol=1e-6)
