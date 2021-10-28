import jax
import jax.numpy as jnp

from e3nn_jax import *

float_tolerance = 2e-5


def test_xyz(keys):
    R = rand_matrix(next(keys), (10,))
    assert jnp.max(jnp.abs(R @ jnp.swapaxes(R, -1, -2) - jnp.eye(3))) < float_tolerance

    a, b, c = matrix_to_angles(R)
    pos1 = angles_to_xyz(a, b)
    pos2 = R @ jnp.array([0, 1.0, 0])
    assert jnp.allclose(pos1, pos2, atol=float_tolerance)

    a2, b2 = xyz_to_angles(pos2)
    assert jnp.max(jnp.abs(a - a2)) < float_tolerance
    assert jnp.max(jnp.abs(b - b2)) < float_tolerance


def test_conversions(keys):
    # @jax.jit
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
            [identity, wrap(angles_to_matrix), wrap(angles_to_axis_angle), wrap(angles_to_quaternion)],
            [wrap(matrix_to_angles), identity, wrap(matrix_to_axis_angle), wrap(matrix_to_quaternion)],
            [wrap(axis_angle_to_angles), wrap(axis_angle_to_matrix), identity, wrap(axis_angle_to_quaternion)],
            [wrap(quaternion_to_angles), wrap(quaternion_to_matrix), wrap(quaternion_to_axis_angle), identity],
        ]
        path = [1, 2, 3, 0, 2, 0, 3, 1, 3, 2, 1, 0, 1]

        for i, j in zip(path, path[1:]):
            g = conv[i][j](g)
        return g

    R1 = rand_matrix(next(keys), (100,))
    R2 = f(R1)

    assert jnp.max(jnp.abs(R1 - R2)) < float_tolerance


def test_compose(keys):
    q1 = rand_quaternion(keys[1], (10,))
    q2 = rand_quaternion(keys[2], (10,))

    q = compose_quaternion(q1, q2)

    R1 = quaternion_to_matrix(q1)
    R2 = quaternion_to_matrix(q2)

    R = R1 @ R2

    abc1 = quaternion_to_angles(q1)
    abc2 = quaternion_to_angles(q2)

    abc = compose_angles(*abc1, *abc2)

    ax1, a1 = quaternion_to_axis_angle(q1)
    ax2, a2 = quaternion_to_axis_angle(q2)

    ax, a = compose_axis_angle(ax1, a1, ax2, a2)

    R1 = quaternion_to_matrix(q)
    R2 = R
    R3 = angles_to_matrix(*abc)
    R4 = axis_angle_to_matrix(ax, a)

    assert jnp.max(jnp.abs(R1 - R2)) < float_tolerance
    assert jnp.max(jnp.abs(R1 - R3)) < float_tolerance
    assert jnp.max(jnp.abs(R1 - R4)) < float_tolerance


def test_inverse_angles(keys):
    a = rand_angles(next(keys), ())
    b = inverse_angles(*a)
    c = compose_angles(*a, *b)
    e = identity_angles(())
    rc = angles_to_matrix(*c)
    re = angles_to_matrix(*e)
    assert jnp.max(jnp.abs(rc - re)) < float_tolerance


def test_rand_axis_angle(keys):
    @jax.jit
    def f(key):
        axis, angle = rand_axis_angle(key, (10_000,))
        return axis_angle_to_matrix(axis, angle) @ jnp.array([0.2, 0.5, 0.3])

    x = f(next(keys))
    tol = 0.005
    assert jnp.max(jnp.abs(jnp.mean(x[:, 0]))) < tol
    assert jnp.max(jnp.abs(jnp.mean(x[:, 1]))) < tol
    assert jnp.max(jnp.abs(jnp.mean(x[:, 2]))) < tol


def test_matrix_xyz(keys):
    x = jax.random.normal(keys[1], (100, 3))
    phi = jax.random.normal(keys[2], (100,))

    y = jnp.einsum('zij,zj->zi', matrix_x(phi), x)
    assert jnp.max(jnp.abs(x[:, 0] - y[:, 0])) < float_tolerance

    y = jnp.einsum('zij,zj->zi', matrix_y(phi), x)
    assert jnp.max(jnp.abs(x[:, 1] - y[:, 1])) < float_tolerance

    y = jnp.einsum('zij,zj->zi', matrix_z(phi), x)
    assert jnp.max(jnp.abs(x[:, 2] - y[:, 2])) < float_tolerance
