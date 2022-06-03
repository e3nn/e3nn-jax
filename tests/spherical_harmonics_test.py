import pytest
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4, 5, 6, 7])
def test_equivariance(keys, l):
    input = e3nn.IrrepsData.randn("1o", keys[0], (10,))

    abc = e3nn.rand_angles(keys[1], ())
    output1 = e3nn.spherical_harmonics(l, input.transform_by_angles(*abc), False)
    output2 = e3nn.spherical_harmonics(l, input, False).transform_by_angles(*abc)

    assert jnp.abs(output1.contiguous - output2.contiguous).max() < 0.01


def test_closure(keys):
    r"""
    integral of Ylm * Yjn = delta_lj delta_mn
    integral of 1 over the unit sphere = 4 pi
    """
    x = jax.random.normal(keys[0], (1_000_000, 3))
    Ys = [e3nn.spherical_harmonics(e3nn.Irreps([l]), x, True).contiguous for l in range(0, 3 + 1)]
    for l1, Y1 in enumerate(Ys):
        for l2, Y2 in enumerate(Ys):
            m = Y1[:, :, None] * Y2[:, None, :]
            m = m.mean(0) * 4 * jnp.pi
            if l1 == l2:
                i = jnp.eye(2 * l1 + 1)
                assert jnp.abs(m - i).max() < 0.01
            else:
                assert jnp.abs(m).max() < 0.01


@pytest.mark.parametrize("l", range(8 + 1))
def test_normalization(keys, l):
    irreps = e3nn.Irreps([l])

    n = jnp.mean(
        e3nn.spherical_harmonics(
            irreps, jax.random.normal(next(keys), (3,)), normalize=True, normalization="integral"
        ).contiguous
        ** 2
    )
    assert abs(n - 1 / (4 * jnp.pi)) < 1e-6

    n = jnp.sum(
        e3nn.spherical_harmonics(irreps, jax.random.normal(next(keys), (3,)), normalize=True, normalization="norm").contiguous
        ** 2
    )
    assert abs(n - 1) < 1e-6

    n = jnp.mean(
        e3nn.spherical_harmonics(
            irreps, jax.random.normal(next(keys), (3,)), normalize=True, normalization="component"
        ).contiguous
        ** 2
    )
    assert abs(n - 1) < 1e-6


@pytest.mark.parametrize("l", range(8 + 1))
def test_parity(keys, l):
    irreps = e3nn.Irreps([l])
    x = jax.random.normal(next(keys), (3,))

    y1 = (-1) ** l * e3nn.spherical_harmonics(irreps, x, normalize=True, normalization="integral")
    y2 = e3nn.spherical_harmonics(irreps, -x, normalize=True, normalization="integral")
    assert jnp.allclose(y1.contiguous, y2.contiguous)


@pytest.mark.parametrize("l", range(7 + 1))
def test_recurrence_relation(keys, l):
    x = jax.random.normal(next(keys), (3,))

    y1 = e3nn.spherical_harmonics(e3nn.Irreps([l + 1]), x, normalize=True, normalization="integral").contiguous
    y2 = jnp.einsum(
        "ijk,i,j->k",
        e3nn.clebsch_gordan(1, l, l + 1),
        x,
        e3nn.spherical_harmonics(e3nn.Irreps([l]), x, normalize=True, normalization="integral").contiguous,
    )

    y1 = y1 / jnp.linalg.norm(y1)
    y2 = y2 / jnp.linalg.norm(y2)
    assert jnp.allclose(y1, y2)
