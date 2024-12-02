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


def test_sampling(num_seeds: int = 10):
    sig = SO3Signal.from_function(
        lambda R: 1.0,
        res_beta=40,
        res_alpha=39,
        res_theta=40,
        quadrature="gausslegendre",
    )
    seeds = jax.random.split(jax.random.PRNGKey(0), num_seeds)
    Rs = jax.vmap(sig.sample)(seeds)
    assert Rs.shape == (num_seeds, 3, 3)

    # Check that the samples are orthogonal.
    for R in Rs:
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-6)


def test_multiplication_scalar():
    sig1 = SO3Signal.from_function(
        lambda R: jnp.trace(R @ R),
        res_beta=40,
        res_alpha=39,
        res_theta=40,
        quadrature="gausslegendre",
    )
    integral1 = sig1.integrate()
    sig2 = sig1 * 2.7
    integral2 = sig2.integrate()
    assert jnp.isclose(integral2, 2.7 * integral1)


def test_multiplication_signal():
    sig1 = SO3Signal.from_function(
        lambda R: jnp.trace(R @ R),
        res_beta=40,
        res_alpha=39,
        res_theta=40,
        quadrature="gausslegendre",
    )
    integral1 = sig1.integrate()

    ones = SO3Signal.from_function(
        lambda R: 1.0,
        res_beta=40,
        res_alpha=39,
        res_theta=40,
        quadrature="gausslegendre",
    )
    sig2 = sig1 * ones
    integral2 = sig2.integrate()
    assert jnp.isclose(integral2, integral1)


def test_division_scalar():
    sig1 = SO3Signal.from_function(
        lambda R: jnp.trace(R @ R),
        res_beta=40,
        res_alpha=39,
        res_theta=40,
        quadrature="gausslegendre",
    )
    integral1 = sig1.integrate()
    sig2 = sig1 / 2.7
    integral2 = sig2.integrate()
    assert jnp.isclose(integral2, integral1 / 2.7)

