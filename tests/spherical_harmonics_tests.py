import pytest
import jax
import jax.numpy as jnp
from e3nn_jax import spherical_harmonics, Irreps, rand_angles, angles_to_matrix


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4, 5, 6, 7, 8])
def test_equivariance(keys, l):
    irreps = Irreps([(1, (l, 1))])
    x = jax.random.normal(keys[0], (3,))
    abc = rand_angles(keys[1], ())
    y1 = spherical_harmonics(irreps, angles_to_matrix(*abc) @ x, False)
    y2 = irreps.transform_by_angles(spherical_harmonics(irreps, x, False), *abc)

    assert jnp.abs(y1 - y2).max() < 1e-4


def test_closure(keys):
    r"""
    integral of Ylm * Yjn = delta_lj delta_mn
    integral of 1 over the unit sphere = 4 pi
    """
    x = jax.random.normal(keys[0], (1_000_000, 3))
    Ys = [spherical_harmonics(Irreps([l]), x, True) for l in range(0, 3 + 1)]
    for l1, Y1 in enumerate(Ys):
        for l2, Y2 in enumerate(Ys):
            m = Y1[:, :, None] * Y2[:, None, :]
            m = m.mean(0) * 4 * jnp.pi
            if l1 == l2:
                i = jnp.eye(2 * l1 + 1)
                assert jnp.abs(m - i).max() < 0.01
            else:
                assert jnp.abs(m).max() < 0.01
