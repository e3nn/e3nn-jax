import jax
import jax.numpy as jnp
import pytest
from e3nn_jax import (Irrep, angles_to_matrix, rand_angles, wigner_3j,
                      wigner_D, wigner_generator_alpha, wigner_generator_beta, wigner_generator_delta)
from e3nn_jax._wigner import _W3j_indices
import math


def test_wigner_3j_symmetry():
    assert jnp.allclose(wigner_3j(1, 2, 3), jnp.swapaxes(wigner_3j(1, 3, 2), 1, 2))
    assert jnp.allclose(wigner_3j(1, 2, 3), jnp.swapaxes(wigner_3j(2, 1, 3), 0, 1))
    assert jnp.allclose(wigner_3j(1, 2, 3), jnp.swapaxes(wigner_3j(3, 2, 1), 0, 2))
    assert jnp.allclose(wigner_3j(1, 2, 3), jnp.swapaxes(jnp.swapaxes(wigner_3j(3, 1, 2), 0, 1), 1, 2))
    assert jnp.allclose(wigner_3j(1, 2, 3), jnp.swapaxes(jnp.swapaxes(wigner_3j(2, 3, 1), 0, 2), 1, 2))


@pytest.mark.parametrize('l1,l2,l3', _W3j_indices.keys())
def test_wigner_3j(keys, l1, l2, l3):
    if abs(l1 - l2) <= l3 <= l1 + l2:
        abc = rand_angles(keys[0], (10,))

        C = wigner_3j(l1, l2, l3)
        D1 = Irrep(l1, 1).D_from_angles(*abc)
        D2 = Irrep(l2, 1).D_from_angles(*abc)
        D3 = Irrep(l3, 1).D_from_angles(*abc)

        C2 = jnp.einsum("ijk,zil,zjm,zkn->zlmn", C, D1, D2, D3)
        assert jnp.max(jnp.abs(C - C2)) < 1e-6


def test_cartesian(keys):
    abc = rand_angles(keys[0], (10,))
    R = angles_to_matrix(*abc)
    D = wigner_D(1, *abc)
    assert jnp.max(jnp.abs(R - D)) < 1e-6


@pytest.mark.parametrize('l', range(1, 11 + 1))
def test_generator_alpha(l):
    G1 = wigner_generator_alpha(l)
    G2 = jax.jacobian(wigner_D, 1)(l, 0.0, 0.0, 0.0)
    assert jnp.abs(G2 - G1).max() < 1e-6


@pytest.mark.parametrize('l', range(1, 11 + 1))
def test_generator_beta(l):
    G1 = wigner_generator_beta(l)
    G2 = jax.jacobian(wigner_D, 2)(l, 0.0, 0.0, 0.0)
    assert jnp.abs(G2 - G1).max() < 1e-6


@pytest.mark.parametrize('l', range(1, 11 + 1))
def test_generator_delta(l):
    G1 = wigner_generator_delta(l)
    G2 = jax.jacobian(wigner_D, 2)(l, math.pi / 2, 0.0, -math.pi / 2)
    assert jnp.abs(G2 - G1).max() < 1e-5
