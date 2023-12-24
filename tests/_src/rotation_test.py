import jax
import jax.numpy as jnp
import numpy as np

import e3nn_jax as e3nn

float_tolerance = 2e-5


def test_xyz(keys):
    R = e3nn.rand_matrix(next(keys), (10,))
    assert jnp.max(jnp.abs(R @ jnp.swapaxes(R, -1, -2) - jnp.eye(3))) < float_tolerance

    a, b, c = e3nn.matrix_to_angles(R)
    pos1 = e3nn.angles_to_xyz(a, b)
    pos2 = R @ jnp.array([0, 1.0, 0])
    assert jnp.allclose(pos1, pos2, atol=float_tolerance)

    a2, b2 = e3nn.xyz_to_angles(pos2)
    assert jnp.max(jnp.abs(a - a2)) < float_tolerance
    assert jnp.max(jnp.abs(b - b2)) < float_tolerance

    rs = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0 / 2**0.5, 1.0 / 2**0.5],
        ]
    )
    for r in rs:
        a, b = e3nn.xyz_to_angles(r)
        R = e3nn.angles_to_matrix(a, -b, -a)
        np.testing.assert_allclose(
            R @ r, np.array([0.0, 1.0, 0.0]), atol=float_tolerance
        )

    Ja, Jb = jax.jacobian(e3nn.xyz_to_angles)(jnp.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(Ja, 0.0, atol=float_tolerance)
    np.testing.assert_allclose(Jb, 0.0, atol=float_tolerance)

    Ja, Jb = jax.jacobian(e3nn.xyz_to_angles)(jnp.array([0.0, -1.0, 0.0]))
    np.testing.assert_allclose(Ja, 0.0, atol=float_tolerance)
    np.testing.assert_allclose(Jb, 0.0, atol=float_tolerance)


def test_conversions(keys):
    jax.config.update("jax_enable_x64", True)

    def f(g):
        def wrap(f):
            def g(x):
                if isinstance(x, tuple):
                    return f(*x)
                else:
                    return f(x)

            return g

        def identity(x):
            return x

        conv = [
            [
                identity,
                wrap(e3nn.angles_to_matrix),
                wrap(e3nn.angles_to_axis_angle),
                wrap(e3nn.angles_to_quaternion),
                wrap(e3nn.angles_to_log_coordinates),
            ],
            [
                wrap(e3nn.matrix_to_angles),
                identity,
                wrap(e3nn.matrix_to_axis_angle),
                wrap(e3nn.matrix_to_quaternion),
                wrap(e3nn.matrix_to_log_coordinates),
            ],
            [
                wrap(e3nn.axis_angle_to_angles),
                wrap(e3nn.axis_angle_to_matrix),
                identity,
                wrap(e3nn.axis_angle_to_quaternion),
                wrap(e3nn.axis_angle_to_log_coordinates),
            ],
            [
                wrap(e3nn.quaternion_to_angles),
                wrap(e3nn.quaternion_to_matrix),
                wrap(e3nn.quaternion_to_axis_angle),
                identity,
                wrap(e3nn.quaternion_to_log_coordinates),
            ],
            [
                wrap(e3nn.log_coordinates_to_angles),
                wrap(e3nn.log_coordinates_to_matrix),
                wrap(e3nn.log_coordinates_to_axis_angle),
                wrap(e3nn.log_coordinates_to_quaternion),
                identity,
            ],
        ]
        path = [1, 2, 3, 0, 2, 0, 3, 1, 3, 2, 1, 0, 1, 4, 2, 4, 3, 4, 0, 4, 1]

        for i, j in zip(path, path[1:]):
            g = conv[i][j](g)
        return g

    R1 = e3nn.rand_matrix(next(keys), (100,), dtype=jnp.float64)
    R2 = f(R1)

    assert R2.dtype == jnp.float64
    np.testing.assert_allclose(R1, R2, rtol=0, atol=1e-10)

    R1 = e3nn.rand_matrix(next(keys), (2,), dtype=jnp.float32)
    assert f(R1).dtype == jnp.float32

    jax.config.update("jax_enable_x64", False)


def test_compose(keys):
    jax.config.update("jax_enable_x64", True)

    q1 = e3nn.rand_quaternion(keys[1], (10,), dtype=jnp.float64)
    q2 = e3nn.rand_quaternion(keys[2], (10,), dtype=jnp.float64)

    q = e3nn.compose_quaternion(q1, q2)

    R1 = e3nn.quaternion_to_matrix(q1)
    R2 = e3nn.quaternion_to_matrix(q2)

    R = R1 @ R2

    abc1 = e3nn.quaternion_to_angles(q1)
    abc2 = e3nn.quaternion_to_angles(q2)

    abc = e3nn.compose_angles(*abc1, *abc2)

    ax1, a1 = e3nn.quaternion_to_axis_angle(q1)
    ax2, a2 = e3nn.quaternion_to_axis_angle(q2)

    ax, a = e3nn.compose_axis_angle(ax1, a1, ax2, a2)

    R1 = e3nn.quaternion_to_matrix(q)
    R2 = R
    R3 = e3nn.angles_to_matrix(*abc)
    R4 = e3nn.axis_angle_to_matrix(ax, a)

    np.testing.assert_allclose(R1, R2, rtol=0, atol=1e-10)
    np.testing.assert_allclose(R1, R3, rtol=0, atol=1e-10)
    np.testing.assert_allclose(R1, R4, rtol=0, atol=1e-10)

    jax.config.update("jax_enable_x64", False)


def test_inverse_angles(keys):
    a = e3nn.rand_angles(next(keys), ())
    b = e3nn.inverse_angles(*a)
    c = e3nn.compose_angles(*a, *b)
    e = e3nn.identity_angles(())
    rc = e3nn.angles_to_matrix(*c)
    re = e3nn.angles_to_matrix(*e)
    assert jnp.max(jnp.abs(rc - re)) < float_tolerance


def test_inverse_log_coordinates(keys):
    a = e3nn.rand_log_coordinates(next(keys), ())
    b = e3nn.inverse_log_coordinates(a)
    c = e3nn.compose_log_coordinates(a, b)
    e = e3nn.identity_log_coordinates(())
    rc = e3nn.log_coordinates_to_matrix(c)
    re = e3nn.log_coordinates_to_matrix(e)
    assert jnp.max(jnp.abs(rc - re)) < float_tolerance


def test_rand_axis_angle(keys):
    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def f(key):
        axis, angle = e3nn.rand_axis_angle(key, (100_000,))
        return e3nn.axis_angle_to_matrix(axis, angle) @ jnp.array([0.2, 0.5, 0.3])

    x = f(next(keys))
    np.testing.assert_allclose(
        jnp.mean(x, axis=0), jnp.array([0.0, 0.0, 0.0]), atol=0.005
    )

    jax.config.update("jax_enable_x64", False)


def test_matrix_xyz(keys):
    x = jax.random.normal(keys[1], (100, 3))
    phi = jax.random.normal(keys[2], (100,))

    y = jnp.einsum("zij,zj->zi", e3nn.matrix_x(phi), x)
    assert jnp.max(jnp.abs(x[:, 0] - y[:, 0])) < float_tolerance

    y = jnp.einsum("zij,zj->zi", e3nn.matrix_y(phi), x)
    assert jnp.max(jnp.abs(x[:, 1] - y[:, 1])) < float_tolerance

    y = jnp.einsum("zij,zj->zi", e3nn.matrix_z(phi), x)
    assert jnp.max(jnp.abs(x[:, 2] - y[:, 2])) < float_tolerance


# TODO: fix this
# def test_matrix_to_axis_angle_stability():
#     R = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
#     axis, angle = jax.jacobian(e3nn.matrix_to_axis_angle)(R)
#     assert jnp.isnan(axis).sum() == 0
#     assert jnp.isnan(angle).sum() == 0


# TODO: fix this
# def test_matrix_to_angles_stability():
#     R = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
#     alpha, beta, gamma = jax.jacobian(e3nn.matrix_to_angles)(R)
#     assert jnp.isnan(alpha).sum() == 0
#     assert jnp.isnan(beta).sum() == 0
#     assert jnp.isnan(gamma).sum() == 0
