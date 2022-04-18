from functools import partial

import jax
import jax.numpy as jnp

# matrix


def rand_matrix(key, shape):
    r"""random rotation matrix

    Args:
        key: a PRNGKey used as the random key.
        shape: a tuple of nonnegative integers representing the result shape.

    Returns:
        `jnp.ndarray`: array of shape :math:`(..., 3, 3)`
    """
    return angles_to_matrix(*rand_angles(key, shape))


# angles


def identity_angles(shape):
    r"""angles of the identity rotation

    Args:
        shape: a tuple of nonnegative integers representing the result shape.

    Returns:
        alpha (`jnp.ndarray`): array of shape :math:`(...)`
        beta (`jnp.ndarray`): array of shape :math:`(...)`
        gamma (`jnp.ndarray`): array of shape :math:`(...)`
    """
    return jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape)


@partial(jax.jit, static_argnums=(1,), inline=True)
def rand_angles(key, shape):
    r"""random rotation angles

    Args:
        key: a PRNGKey used as the random key.
        shape: a tuple of nonnegative integers representing the result shape.

    Returns:
        alpha (`jnp.ndarray`): array of shape :math:`(...)`
        beta (`jnp.ndarray`): array of shape :math:`(...)`
        gamma (`jnp.ndarray`): array of shape :math:`(...)`
    """
    x, y, z = jax.random.uniform(key, (3,) + shape)
    return 2 * jnp.pi * x, jnp.arccos(2 * z - 1), 2 * jnp.pi * y


def compose_angles(a1, b1, c1, a2, b2, c2):
    r"""compose angles

    Computes :math:`(a, b, c)` such that :math:`R(a, b, c) = R(a_1, b_1, c_1) \circ R(a_2, b_2, c_2)`

    Args:
        alpha1 (`jnp.ndarray`): array of shape :math:`(...)`
        beta1 (`jnp.ndarray`): array of shape :math:`(...)`
        gamma1 (`jnp.ndarray`): array of shape :math:`(...)`
        alpha2 (`jnp.ndarray`): array of shape :math:`(...)`
        beta2 (`jnp.ndarray`): array of shape :math:`(...)`
        gamma2 (`jnp.ndarray`): array of shape :math:`(...)`

    Returns:
        alpha (`jnp.ndarray`): array of shape :math:`(...)`
        beta (`jnp.ndarray`): array of shape :math:`(...)`
        gamma (`jnp.ndarray`): array of shape :math:`(...)`

    """
    a1, b1, c1, a2, b2, c2 = jnp.broadcast_arrays(a1, b1, c1, a2, b2, c2)
    return matrix_to_angles(angles_to_matrix(a1, b1, c1) @ angles_to_matrix(a2, b2, c2))


def inverse_angles(a, b, c):
    r"""angles of the inverse rotation

    Args:
        alpha (`jnp.ndarray`): array of shape :math:`(...)`
        beta (`jnp.ndarray`): array of shape :math:`(...)`
        gamma (`jnp.ndarray`): array of shape :math:`(...)`

    Returns:
        alpha (`jnp.ndarray`): array of shape :math:`(...)`
        beta (`jnp.ndarray`): array of shape :math:`(...)`
        gamma (`jnp.ndarray`): array of shape :math:`(...)`
    """
    return -c, -b, -a


# quaternions


def identity_quaternion(shape):
    r"""quaternion of identity rotation

    Args:
        shape: a tuple of nonnegative integers representing the result shape.

    Returns:
        `jnp.ndarray`: array of shape :math:`(..., 4)`
    """
    q = jnp.zeros(shape, 4)
    return q.at[..., 0].set(1)  # or -1...


@partial(jax.jit, static_argnums=(1,), inline=True)
def rand_quaternion(key, shape):
    r"""generate random quaternion

    Args:
        key: a PRNGKey used as the random key.
        shape: a tuple of nonnegative integers representing the result shape.

    Returns:
        `jnp.ndarray`: array of shape :math:`(..., 4)`
    """
    return angles_to_quaternion(*rand_angles(key, shape))


@partial(jax.jit, inline=True)
def compose_quaternion(q1, q2):
    r"""compose two quaternions: :math:`q_1 \circ q_2`

    Args:
        q1 (`jnp.ndarray`): array of shape :math:`(..., 4)`
        q2 (`jnp.ndarray`): array of shape :math:`(..., 4)`

    Returns:
        `jnp.ndarray`: array of shape :math:`(..., 4)`
    """
    q1, q2 = jnp.broadcast_arrays(q1, q2)
    return jnp.stack(
        [
            q1[..., 0] * q2[..., 0] - q1[..., 1] * q2[..., 1] - q1[..., 2] * q2[..., 2] - q1[..., 3] * q2[..., 3],
            q1[..., 1] * q2[..., 0] + q1[..., 0] * q2[..., 1] + q1[..., 2] * q2[..., 3] - q1[..., 3] * q2[..., 2],
            q1[..., 0] * q2[..., 2] - q1[..., 1] * q2[..., 3] + q1[..., 2] * q2[..., 0] + q1[..., 3] * q2[..., 1],
            q1[..., 0] * q2[..., 3] + q1[..., 1] * q2[..., 2] - q1[..., 2] * q2[..., 1] + q1[..., 3] * q2[..., 0],
        ],
        axis=-1,
    )


def inverse_quaternion(q):
    r"""inverse of a quaternion

    Works only for unit quaternions.

    Args:
        q (`jnp.ndarray`): array of shape :math:`(..., 4)`

    Returns:
        `jnp.ndarray`: array of shape :math:`(..., 4)`
    """
    return q.at[..., 1:].multiply(-1)


# axis-angle


def rand_axis_angle(key, shape):
    r"""generate random rotation as axis-angle

    Args:
        key: a PRNGKey used as the random key.
        shape: a tuple of nonnegative integers representing the result shape.

    Returns:
        axis (`jnp.ndarray`): array of shape :math:`(..., 3)`
        angle (`jnp.ndarray`): array of shape :math:`(...)`
    """
    return angles_to_axis_angle(*rand_angles(key, shape))


def compose_axis_angle(axis1, angle1, axis2, angle2):
    r"""compose :math:`(\vec x_1, \alpha_1)` with :math:`(\vec x_2, \alpha_2)`

    Args:
        axis1 (`jnp.ndarray`): array of shape :math:`(..., 3)`
        angle1 (`jnp.ndarray`): array of shape :math:`(...)`
        axis2 (`jnp.ndarray`): array of shape :math:`(..., 3)`
        angle2 (`jnp.ndarray`): array of shape :math:`(...)`

    Returns:
        axis (`jnp.ndarray`): array of shape :math:`(..., 3)`
        angle (`jnp.ndarray`): array of shape :math:`(...)`
    """
    return quaternion_to_axis_angle(
        compose_quaternion(axis_angle_to_quaternion(axis1, angle1), axis_angle_to_quaternion(axis2, angle2))
    )


# conversions


@partial(jax.jit, inline=True)
def matrix_x(angle):
    r"""matrix of rotation around X axis

    Args:
        angle (`jnp.ndarray`): array of shape :math:`(...)`

    Returns:
        `jnp.ndarray`: array of shape :math:`(..., 3, 3)`
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    o = jnp.ones_like(angle)
    z = jnp.zeros_like(angle)
    return jnp.stack(
        [
            jnp.stack([o, z, z], axis=-1),
            jnp.stack([z, c, -s], axis=-1),
            jnp.stack([z, s, c], axis=-1),
        ],
        axis=-2,
    )


@partial(jax.jit, inline=True)
def matrix_y(angle):
    r"""matrix of rotation around Y axis

    Args:
        angle (`jnp.ndarray`): array of shape :math:`(...)`

    Returns:
        `jnp.ndarray`: array of shape :math:`(..., 3, 3)`
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    o = jnp.ones_like(angle)
    z = jnp.zeros_like(angle)
    return jnp.stack(
        [
            jnp.stack([c, z, s], axis=-1),
            jnp.stack([z, o, z], axis=-1),
            jnp.stack([-s, z, c], axis=-1),
        ],
        axis=-2,
    )


@partial(jax.jit, inline=True)
def matrix_z(angle):
    r"""matrix of rotation around Z axis

    Args:
        angle (`jnp.ndarray`): array of shape :math:`(...)`

    Returns:
        `jnp.ndarray`: array of shape :math:`(..., 3, 3)`
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    o = jnp.ones_like(angle)
    z = jnp.zeros_like(angle)
    return jnp.stack([jnp.stack([c, -s, z], axis=-1), jnp.stack([s, c, z], axis=-1), jnp.stack([z, z, o], axis=-1)], axis=-2)


def angles_to_matrix(alpha, beta, gamma):
    r"""conversion from angles to matrix

    Args:
        alpha (`jnp.ndarray`): array of shape :math:`(...)`
        beta (`jnp.ndarray`): array of shape :math:`(...)`
        gamma (`jnp.ndarray`): array of shape :math:`(...)`

    Returns:
        `jnp.ndarray`: array of shape :math:`(..., 3, 3)`
    """
    alpha, beta, gamma = jnp.broadcast_arrays(alpha, beta, gamma)
    return matrix_y(alpha) @ matrix_x(beta) @ matrix_y(gamma)


@partial(jax.jit, inline=True)
def matrix_to_angles(R):
    r"""conversion from matrix to angles

    Args:
        R (`jnp.ndarray`): array of shape :math:`(..., 3, 3)`

    Returns:
        alpha (`jnp.ndarray`): array of shape :math:`(...)`
        beta (`jnp.ndarray`): array of shape :math:`(...)`
        gamma (`jnp.ndarray`): array of shape :math:`(...)`
    """
    # assert jnp.allclose(jnp.linalg.det(R), 1)
    x = R @ jnp.array([0.0, 1.0, 0.0])
    a, b = xyz_to_angles(x)
    R = jnp.swapaxes(angles_to_matrix(a, b, 0.0), -1, -2) @ R
    c = jnp.arctan2(R[..., 0, 2], R[..., 0, 0])
    return a, b, c


@partial(jax.jit, inline=True)
def angles_to_quaternion(alpha, beta, gamma):
    r"""conversion from angles to quaternion

    Args:
        alpha (`jnp.ndarray`): array of shape :math:`(...)`
        beta (`jnp.ndarray`): array of shape :math:`(...)`
        gamma (`jnp.ndarray`): array of shape :math:`(...)`

    Returns:
        q (`jnp.ndarray`): array of shape :math:`(..., 4)`
    """
    alpha, beta, gamma = jnp.broadcast_arrays(alpha, beta, gamma)
    qa = axis_angle_to_quaternion(jnp.array([0.0, 1.0, 0.0]), alpha)
    qb = axis_angle_to_quaternion(jnp.array([1.0, 0.0, 0.0]), beta)
    qc = axis_angle_to_quaternion(jnp.array([0.0, 1.0, 0.0]), gamma)
    return compose_quaternion(qa, compose_quaternion(qb, qc))


def matrix_to_quaternion(R):
    r"""conversion from matrix :math:`R` to quaternion :math:`q`

    Args:
        R (`jnp.ndarray`): array of shape :math:`(..., 3, 3)`

    Returns:
        q (`jnp.ndarray`): array of shape :math:`(..., 4)`
    """
    return axis_angle_to_quaternion(*matrix_to_axis_angle(R))


@partial(jax.jit, inline=True)
def axis_angle_to_quaternion(xyz, angle):
    r"""convertion from axis-angle to quaternion

    Args:
        axis (`jnp.ndarray`): array of shape :math:`(..., 3)`
        angle (`jnp.ndarray`): array of shape :math:`(...)`

    Returns:
        q (`jnp.ndarray`): array of shape :math:`(..., 4)`
    """
    xyz, angle = jnp.broadcast_arrays(xyz, angle[..., None])
    xyz = _normalize(xyz)
    c = jnp.cos(angle[..., :1] / 2)
    s = jnp.sin(angle / 2)
    return jnp.concatenate([c, xyz * s], axis=-1)


@partial(jax.jit, inline=True)
def quaternion_to_axis_angle(q):
    r"""convertion from quaternion to axis-angle

    Args:
        q (`jnp.ndarray`): array of shape :math:`(..., 4)`

    Returns:
        axis (`jnp.ndarray`): array of shape :math:`(..., 3)`
        angle (`jnp.ndarray`): array of shape :math:`(...)`
    """
    angle = 2 * jnp.arccos(jnp.clip(q[..., 0], -1, 1))
    axis = _normalize(q[..., 1:])
    return axis, angle


def _normalize(x):
    n = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.where(n > 0, n, 1.0)


@partial(jax.jit, inline=True)
def matrix_to_axis_angle(R):
    r"""conversion from matrix to axis-angle

    Args:
        R (`jnp.ndarray`): array of shape :math:`(..., 3, 3)`

    Returns:
        axis (`jnp.ndarray`): array of shape :math:`(..., 3)`
        angle (`jnp.ndarray`): array of shape :math:`(...)`
    """
    # assert jnp.allclose(jnp.linalg.det(R), 1)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    angle = jnp.arccos(jnp.clip((tr - 1) / 2, -1, 1))
    axis = jnp.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        axis=-1,
    )
    axis = _normalize(axis)
    return axis, angle


def angles_to_axis_angle(alpha, beta, gamma):
    r"""conversion from angles to axis-angle

    Args:
        alpha (`jnp.ndarray`): array of shape :math:`(...)`
        beta (`jnp.ndarray`): array of shape :math:`(...)`
        gamma (`jnp.ndarray`): array of shape :math:`(...)`

    Returns:
        axis (`jnp.ndarray`): array of shape :math:`(..., 3)`
        angle (`jnp.ndarray`): array of shape :math:`(...)`
    """
    return matrix_to_axis_angle(angles_to_matrix(alpha, beta, gamma))


@partial(jax.jit, inline=True)
def axis_angle_to_matrix(axis, angle):
    r"""conversion from axis-angle to matrix

    Args:
        axis (`jnp.ndarray`): array of shape :math:`(..., 3)`
        angle (`jnp.ndarray`): array of shape :math:`(...)`

    Returns:
        `jnp.ndarray`: array of shape :math:`(..., 3, 3)`
    """
    axis, angle = jnp.broadcast_arrays(axis, angle[..., None])
    alpha, beta = xyz_to_angles(axis)
    R = angles_to_matrix(alpha, beta, jnp.zeros_like(beta))
    Ry = matrix_y(angle[..., 0])
    return R @ Ry @ jnp.swapaxes(R, -2, -1)


def quaternion_to_matrix(q):
    r"""convertion from quaternion to matrix

    Args:
        q (`jnp.ndarray`): array of shape :math:`(..., 4)`

    Returns:
        `jnp.ndarray`: array of shape :math:`(..., 3, 3)`
    """
    return axis_angle_to_matrix(*quaternion_to_axis_angle(q))


def quaternion_to_angles(q):
    r"""convertion from quaternion to angles

    Args:
        q (`jnp.ndarray`): array of shape :math:`(..., 4)`

    Returns:
        alpha (`jnp.ndarray`): array of shape :math:`(...)`
        beta (`jnp.ndarray`): array of shape :math:`(...)`
        gamma (`jnp.ndarray`): array of shape :math:`(...)`
    """
    return matrix_to_angles(quaternion_to_matrix(q))


def axis_angle_to_angles(axis, angle):
    r"""convertion from axis-angle to angles

    Args:
        axis (`jnp.ndarray`): array of shape :math:`(..., 3)`
        angle (`jnp.ndarray`): array of shape :math:`(...)`

    Returns:
        alpha (`jnp.ndarray`): array of shape :math:`(...)`
        beta (`jnp.ndarray`): array of shape :math:`(...)`
        gamma (`jnp.ndarray`): array of shape :math:`(...)`
    """
    return matrix_to_angles(axis_angle_to_matrix(axis, angle))


# point on the sphere


@partial(jax.jit, inline=True)
def angles_to_xyz(alpha, beta):
    r"""convert :math:`(\alpha, \beta)` into a point :math:`(x, y, z)` on the sphere

    Args:
        alpha (`jnp.ndarray`): array of shape :math:`(...)`
        beta (`jnp.ndarray`): array of shape :math:`(...)`

    Returns:
        `jnp.ndarray`: array of shape :math:`(..., 3)`

    Examples:
        >>> angles_to_xyz(1.7, 0.0) + 0.0
        DeviceArray([0., 1., 0.], dtype=float32, weak_type=True)
    """
    alpha, beta = jnp.broadcast_arrays(alpha, beta)
    x = jnp.sin(beta) * jnp.sin(alpha)
    y = jnp.cos(beta)
    z = jnp.sin(beta) * jnp.cos(alpha)
    return jnp.stack([x, y, z], axis=-1)


@partial(jax.jit, inline=True)
def xyz_to_angles(xyz):
    r"""convert a point :math:`\vec r = (x, y, z)` on the sphere into angles :math:`(\alpha, \beta)`

    .. math::

        \vec r = R(\alpha, \beta, 0) \vec e_y

        \alpha = \arctan(x/z)

        \beta = \arccos(y)

    Args:
        xyz (`jnp.ndarray`): array of shape :math:`(..., 3)`

    Returns:
        alpha (`jnp.ndarray`): array of shape :math:`(...)`
        beta (`jnp.ndarray`): array of shape :math:`(...)`
    """
    xyz = _normalize(xyz)
    xyz = jnp.clip(xyz, -1, 1)

    beta = jnp.arccos(xyz[..., 1])
    alpha = jnp.arctan2(xyz[..., 0], xyz[..., 2])
    return alpha, beta
