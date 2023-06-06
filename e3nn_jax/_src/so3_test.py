import math

import jax
import jax.numpy as jnp
import pytest
from e3nn_jax import Irrep, angles_to_matrix, clebsch_gordan, generators, rand_angles


def test_clebsch_gordan_symmetry():
    assert jnp.allclose(
        clebsch_gordan(1, 2, 3), jnp.swapaxes(clebsch_gordan(1, 3, 2), 1, 2)
    )
    assert jnp.allclose(
        clebsch_gordan(1, 2, 3), jnp.swapaxes(clebsch_gordan(2, 1, 3), 0, 1)
    )
    assert jnp.allclose(
        clebsch_gordan(1, 2, 3), jnp.swapaxes(clebsch_gordan(3, 2, 1), 0, 2)
    )
    assert jnp.allclose(
        clebsch_gordan(1, 2, 3),
        jnp.swapaxes(jnp.swapaxes(clebsch_gordan(3, 1, 2), 0, 1), 1, 2),
    )
    assert jnp.allclose(
        clebsch_gordan(1, 2, 3),
        jnp.swapaxes(jnp.swapaxes(clebsch_gordan(2, 3, 1), 0, 2), 1, 2),
    )


def unique_triplets(lmax):
    for l1 in range(lmax + 1):
        for l2 in range(l1 + 1):
            for l3 in range(l2 + 1):
                if abs(l1 - l2) <= l3 <= l1 + l2:
                    yield (l1, l2, l3)


@pytest.mark.parametrize("l1,l2,l3", unique_triplets(5))
def test_clebsch_gordan(keys, l1, l2, l3):
    abc = rand_angles(keys[0], (10,))

    C = clebsch_gordan(l1, l2, l3)
    D1 = Irrep(l1, 1).D_from_angles(*abc)
    D2 = Irrep(l2, 1).D_from_angles(*abc)
    D3 = Irrep(l3, 1).D_from_angles(*abc)

    C2 = jnp.einsum("ijk,zil,zjm,zkn->zlmn", C, D1, D2, D3)
    assert jnp.allclose(C, C2, atol=1e-3, rtol=1e-3)


def wigner_D(l, a, b, c):
    return Irrep(l, 1).D_from_angles(a, b, c)


def test_cartesian(keys):
    abc = rand_angles(keys[0], (10,))
    R = angles_to_matrix(*abc)
    D = wigner_D(1, *abc)
    assert jnp.max(jnp.abs(R - D)) < 1e-4


@pytest.mark.parametrize("l", range(1, 11 + 1))
def test_generator_x(l):
    G1 = generators(l)[0]
    G2 = jax.jacobian(wigner_D, 2)(l, 0.0, 0.0, 0.0)
    assert jnp.abs(G2 - G1).max() < 1e-6


@pytest.mark.parametrize("l", range(1, 11 + 1))
def test_generator_y(l):
    G1 = generators(l)[1]
    G2 = jax.jacobian(wigner_D, 1)(l, 0.0, 0.0, 0.0)
    assert jnp.abs(G2 - G1).max() < 1e-6


@pytest.mark.parametrize("l", range(1, 11 + 1))
def test_generator_z(l):
    G1 = generators(l)[2]
    G2 = jax.jacobian(wigner_D, 2)(l, -math.pi / 2, 0.0, math.pi / 2)
    assert jnp.abs(G2 - G1).max() < 0.005


def commutator(a, b):
    return a @ b - b @ a


@pytest.mark.parametrize("l", range(1, 11 + 1))
def test_commutation_generators(l):
    a, b, c = generators(l)

    assert jnp.allclose(commutator(a, b), c, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(commutator(b, c), a, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(commutator(c, a), b, atol=1e-5, rtol=1e-5)
