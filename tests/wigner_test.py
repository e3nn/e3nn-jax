import jax.numpy as jnp
import pytest
from e3nn_jax import Irrep, angles_to_matrix, rand_angles, wigner_3j, wigner_D
from e3nn_jax._wigner import _W3j_indices


def test_wigner_3j_symmetry():
    assert jnp.allclose(wigner_3j(1, 2, 3), jnp.swapaxes(wigner_3j(1, 3, 2), 1, 2))
    assert jnp.allclose(wigner_3j(1, 2, 3), jnp.swapaxes(wigner_3j(2, 1, 3), 0, 1))
    assert jnp.allclose(wigner_3j(1, 2, 3), jnp.swapaxes(wigner_3j(3, 2, 1), 0, 2))
    assert jnp.allclose(wigner_3j(1, 2, 3), jnp.swapaxes(jnp.swapaxes(wigner_3j(3, 1, 2), 0, 1), 1, 2))
    assert jnp.allclose(wigner_3j(1, 2, 3), jnp.swapaxes(jnp.swapaxes(wigner_3j(2, 3, 1), 0, 2), 1, 2))


@pytest.mark.parametrize('l1,l2,l3', _W3j_indices.keys())
def test_wigner_3j(key, l1, l2, l3):
    if abs(l1 - l2) <= l3 <= l1 + l2:
        abc = rand_angles(key, (10,))

        C = wigner_3j(l1, l2, l3)
        D1 = Irrep(l1, 1).D_from_angles(*abc)
        D2 = Irrep(l2, 1).D_from_angles(*abc)
        D3 = Irrep(l3, 1).D_from_angles(*abc)

        C2 = jnp.einsum("ijk,zil,zjm,zkn->zlmn", C, D1, D2, D3)
        assert jnp.max(jnp.abs(C - C2)) < 1e-6


def test_cartesian(key):
    abc = rand_angles(key, (10,))
    R = angles_to_matrix(*abc)
    D = wigner_D(1, *abc)
    assert jnp.max(jnp.abs(R - D)) < 1e-6
