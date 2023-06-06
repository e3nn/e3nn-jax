import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads


@pytest.fixture(
    params=[
        "recursive,dense",
        "recursive,sparse,custom_jvp",
        "legendre,dense,custom_jvp",
    ],
)
def algorithm(request):
    return tuple(request.param.split(","))


@pytest.mark.parametrize("l", [0, 1, 2, 3, 4, 5, 6, 7])
def test_equivariance(keys, algorithm, l):
    input = e3nn.normal("1o", keys[0], (10,))

    abc = e3nn.rand_angles(keys[1], ())
    output1 = e3nn.spherical_harmonics(
        l, input.transform_by_angles(*abc), False, algorithm=algorithm
    )
    output2 = e3nn.spherical_harmonics(
        l, input, False, algorithm=algorithm
    ).transform_by_angles(*abc)

    np.testing.assert_allclose(output1.array, output2.array, atol=1e-2, rtol=1e-2)


def test_closure(keys, algorithm):
    r"""
    integral of Ylm * Yjn = delta_lj delta_mn
    integral of 1 over the unit sphere = 4 pi
    """
    x = jax.random.normal(keys[0], (1_000_000, 3))
    Ys = [e3nn.sh(l, x, True, "integral", algorithm=algorithm) for l in range(0, 3 + 1)]
    for l1, Y1 in enumerate(Ys):
        for l2, Y2 in enumerate(Ys):
            m = Y1[:, :, None] * Y2[:, None, :]
            m = m.mean(0) * 4 * jnp.pi
            if l1 == l2:
                i = jnp.eye(2 * l1 + 1)
                assert jnp.abs(m - i).max() < 0.01
            else:
                assert jnp.abs(m).max() < 0.01


@pytest.mark.parametrize("l", range(13 + 1))
def test_normalization_integral(keys, algorithm, l):
    irreps = e3nn.Irreps([l])

    n = jnp.mean(
        e3nn.spherical_harmonics(
            irreps,
            jax.random.normal(keys[l + 0], (3,)),
            normalize=True,
            normalization="integral",
            algorithm=algorithm,
        ).array
        ** 2
    )
    assert abs((4 * jnp.pi) * n - 1) < 7e-7 * max((l / 4) ** 8, 1)


@pytest.mark.parametrize("l", range(13 + 1))
def test_normalization_norm(keys, algorithm, l):
    irreps = e3nn.Irreps([l])

    n = jnp.sum(
        e3nn.spherical_harmonics(
            irreps,
            jax.random.normal(keys[l + 1], (3,)),
            normalize=True,
            normalization="norm",
            algorithm=algorithm,
        ).array
        ** 2
    )
    assert abs(n - 1) < 6e-7 * max((l / 4) ** 8, 1)


@pytest.mark.parametrize("l", range(13 + 1))
def test_normalization_component(keys, algorithm, l):
    irreps = e3nn.Irreps([l])

    n = jnp.mean(
        e3nn.spherical_harmonics(
            irreps,
            jax.random.normal(keys[l + 2], (3,)),
            normalize=True,
            normalization="component",
            algorithm=algorithm,
        ).array
        ** 2
    )
    assert abs(n - 1) < 6e-7 * max((l / 4) ** 8, 1)


@pytest.mark.parametrize("l", range(8 + 1))
def test_parity(keys, algorithm, l):
    irreps = e3nn.Irreps([l])
    x = jax.random.normal(next(keys), (3,))

    y1 = (-1) ** l * e3nn.spherical_harmonics(
        irreps, x, normalize=True, normalization="integral", algorithm=algorithm
    )
    y2 = e3nn.spherical_harmonics(
        irreps, -x, normalize=True, normalization="integral", algorithm=algorithm
    )
    np.testing.assert_allclose(y1.array, y2.array, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("l", range(7 + 1))
def test_recurrence_relation(keys, algorithm, l):
    x = jax.random.normal(next(keys), (3,))

    y1 = e3nn.spherical_harmonics(
        e3nn.Irreps([l + 1]),
        x,
        normalize=True,
        normalization="integral",
        algorithm=algorithm,
    ).array
    y2 = jnp.einsum(
        "ijk,i,j->k",
        e3nn.clebsch_gordan(1, l, l + 1),
        x,
        e3nn.spherical_harmonics(
            e3nn.Irreps([l]),
            x,
            normalize=True,
            normalization="integral",
            algorithm=algorithm,
        ).array,
    )

    y1 = y1 / jnp.linalg.norm(y1)
    y2 = y2 / jnp.linalg.norm(y2)
    np.testing.assert_allclose(y1, y2, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("normalization", ["integral", "norm", "component"])
@pytest.mark.parametrize("irreps", ["3x1o+2e+2x4e", "2x0e", "10e"])
def test_check_grads(keys, algorithm, irreps, normalization):
    check_grads(
        lambda x: e3nn.spherical_harmonics(
            irreps, x, normalize=False, normalization=normalization, algorithm=algorithm
        ).array,
        (jax.random.normal(keys[0], (10, 3)),),
        1,
        modes=["fwd", "rev"],
        atol=3e-3,
        rtol=3e-3,
    )


@pytest.mark.parametrize("l", range(7 + 1))
def test_normalize(keys, algorithm, l):
    x = jax.random.normal(keys[0], (10, 3))
    y1 = (
        e3nn.spherical_harmonics(
            e3nn.Irreps([l]), x, normalize=True, algorithm=algorithm
        ).array
        * jnp.linalg.norm(x, axis=1, keepdims=True) ** l
    )
    y2 = e3nn.spherical_harmonics(
        e3nn.Irreps([l]), x, normalize=False, algorithm=algorithm
    ).array
    np.testing.assert_allclose(y1, y2, atol=1e-6, rtol=1e-5)


def test_edge_cases():
    x = jnp.array([1, 0, 0.0])
    y = e3nn.spherical_harmonics(e3nn.Irreps(""), x, True)
    assert y.shape == (0,)
    y = e3nn.spherical_harmonics("", x, True)
    assert y.shape == (0,)

    x = e3nn.IrrepsArray("1o", x)
    y = e3nn.spherical_harmonics([], x, True)
    assert y.shape == (0,)
