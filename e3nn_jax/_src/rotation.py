import jax
import jax.numpy as jnp

# matrix


def rand_matrix(key, shape=(), dtype=jnp.float32):
    r"""Random rotation matrix.

    Args:
        key: a PRNGKey used as the random key.
        shape: a tuple of nonnegative integers representing the result shape.

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 3, 3)`
    """
    return angles_to_matrix(*rand_angles(key, shape, dtype=dtype))


def rotation_angle_from_matrix(R):
    r"""Angle of rotation from a rotation matrix.

    Args:
        m (`jax.numpy.ndarray`): array of shape :math:`(..., 3, 3)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(...)`
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return jnp.arccos(jnp.clip((trace - 1.0) / 2.0, -1.0, 1.0))


# angles


def identity_angles(shape=(), dtype=jnp.float32):
    r"""Angles of the identity rotation.

    Args:
        shape: a tuple of nonnegative integers representing the result shape.

    Returns:
        (tuple): tuple containing:

            alpha (`jax.numpy.ndarray`): array of shape :math:`(...)`
            beta (`jax.numpy.ndarray`): array of shape :math:`(...)`
            gamma (`jax.numpy.ndarray`): array of shape :math:`(...)`
    """
    return jnp.zeros(shape, dtype), jnp.zeros(shape, dtype), jnp.zeros(shape, dtype)


def rand_angles(key, shape=(), dtype=jnp.float32):
    r"""Random rotation angles.

    Args:
        key: a PRNGKey used as the random key.
        shape: a tuple of nonnegative integers representing the result shape.

    Returns:
        (tuple): tuple containing:

            alpha (`jax.numpy.ndarray`): array of shape :math:`(...)`
            beta (`jax.numpy.ndarray`): array of shape :math:`(...)`
            gamma (`jax.numpy.ndarray`): array of shape :math:`(...)`
    """
    x, y, z = jax.random.uniform(key, (3,) + shape, dtype=dtype)
    return 2 * jnp.pi * x, jnp.arccos(2 * z - 1), 2 * jnp.pi * y


def compose_angles(a1, b1, c1, a2, b2, c2):
    r"""Compose angles.

    Computes :math:`(a, b, c)` such that :math:`R(a, b, c) = R(a_1, b_1, c_1) \circ R(a_2, b_2, c_2)`

    Args:
        alpha1 (`jax.numpy.ndarray`): array of shape :math:`(...)`
        beta1 (`jax.numpy.ndarray`): array of shape :math:`(...)`
        gamma1 (`jax.numpy.ndarray`): array of shape :math:`(...)`
        alpha2 (`jax.numpy.ndarray`): array of shape :math:`(...)`
        beta2 (`jax.numpy.ndarray`): array of shape :math:`(...)`
        gamma2 (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        (tuple): tuple containing:

            alpha (`jax.numpy.ndarray`): array of shape :math:`(...)`
            beta (`jax.numpy.ndarray`): array of shape :math:`(...)`
            gamma (`jax.numpy.ndarray`): array of shape :math:`(...)`

    """
    a1, b1, c1, a2, b2, c2 = jnp.broadcast_arrays(a1, b1, c1, a2, b2, c2)
    return matrix_to_angles(angles_to_matrix(a1, b1, c1) @ angles_to_matrix(a2, b2, c2))


def inverse_angles(a, b, c):
    r"""Angles of the inverse rotation.

    Args:
        alpha (`jax.numpy.ndarray`): array of shape :math:`(...)`
        beta (`jax.numpy.ndarray`): array of shape :math:`(...)`
        gamma (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        (tuple): tuple containing:

            alpha (`jax.numpy.ndarray`): array of shape :math:`(...)`
            beta (`jax.numpy.ndarray`): array of shape :math:`(...)`
            gamma (`jax.numpy.ndarray`): array of shape :math:`(...)`
    """
    return -c, -b, -a


def rotation_angle_from_angles(a1, b1, c1, a2, b2, c2):
    r"""Angle of rotation from two triplets of angles.

    Args:
        alpha1 (`jax.numpy.ndarray`): array of shape :math:`(...)`
        beta1 (`jax.numpy.ndarray`): array of shape :math:`(...)`
        gamma1 (`jax.numpy.ndarray`): array of shape :math:`(...)`
        alpha2 (`jax.numpy.ndarray`): array of shape :math:`(...)`
        beta2 (`jax.numpy.ndarray`): array of shape :math:`(...)`
        gamma2 (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(...)`
    """
    R1 = angles_to_matrix(a1, b1, c1)
    R2 = angles_to_matrix(a2, b2, c2)
    R1, R2 = jnp.broadcast_arrays(R1, R2)
    return rotation_angle_from_matrix(R1 @ R2.T)


# quaternions


def identity_quaternion(shape=(), dtype=jnp.float32):
    r"""Quaternion of identity rotation.

    Args:
        shape: a tuple of nonnegative integers representing the result shape.

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 4)`
    """
    q = jnp.zeros(shape + (4,), dtype=dtype)
    return q.at[..., 0].set(1)  # or -1...


def rand_quaternion(key, shape=(), dtype=jnp.float32):
    r"""Generate random quaternion.

    Args:
        key: a PRNGKey used as the random key.
        shape: a tuple of nonnegative integers representing the result shape.

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 4)`
    """
    return angles_to_quaternion(*rand_angles(key, shape, dtype))


def compose_quaternion(q1, q2):
    r"""Compose two quaternions: :math:`q_1 \circ q_2`.

    Args:
        q1 (`jax.numpy.ndarray`): array of shape :math:`(..., 4)`
        q2 (`jax.numpy.ndarray`): array of shape :math:`(..., 4)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 4)`
    """
    q1, q2 = jnp.broadcast_arrays(q1, q2)
    return jnp.stack(
        [
            q1[..., 0] * q2[..., 0]
            - q1[..., 1] * q2[..., 1]
            - q1[..., 2] * q2[..., 2]
            - q1[..., 3] * q2[..., 3],
            q1[..., 1] * q2[..., 0]
            + q1[..., 0] * q2[..., 1]
            + q1[..., 2] * q2[..., 3]
            - q1[..., 3] * q2[..., 2],
            q1[..., 0] * q2[..., 2]
            - q1[..., 1] * q2[..., 3]
            + q1[..., 2] * q2[..., 0]
            + q1[..., 3] * q2[..., 1],
            q1[..., 0] * q2[..., 3]
            + q1[..., 1] * q2[..., 2]
            - q1[..., 2] * q2[..., 1]
            + q1[..., 3] * q2[..., 0],
        ],
        axis=-1,
    )


def inverse_quaternion(q):
    r"""Inverse of a quaternion.

    Works only for unit quaternions.

    Args:
        q (`jax.numpy.ndarray`): array of shape :math:`(..., 4)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 4)`
    """
    return q.at[..., 1:].multiply(-1)


def rotation_angle_from_quaternion(q1, q2):
    r"""Rotation angle between two quaternions.

    Args:
        q1 (`jax.numpy.ndarray`): array of shape :math:`(..., 4)`
        q2 (`jax.numpy.ndarray`): array of shape :math:`(..., 4)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(...)`
    """
    q1, q2 = jnp.broadcast_arrays(q1, q2)
    dot = jnp.sum(q1 * q2, axis=-1)
    cap = jnp.minimum(jnp.abs(dot), 1.0)
    return 2.0 * jnp.arccos(cap)


# axis-angle


def rand_axis_angle(key, shape=(), dtype=jnp.float32):
    r"""Generate random rotation as axis-angle.

    Args:
        key: a PRNGKey used as the random key.
        shape: a tuple of nonnegative integers representing the result shape.

    Returns:
        (tuple): tuple containing:

            axis (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
            angle (`jax.numpy.ndarray`): array of shape :math:`(...)`
    """
    return angles_to_axis_angle(*rand_angles(key, shape, dtype))


def compose_axis_angle(axis1, angle1, axis2, angle2):
    r"""Compose :math:`(\vec x_1, \alpha_1)` with :math:`(\vec x_2, \alpha_2)`.

    Args:
        axis1 (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
        angle1 (`jax.numpy.ndarray`): array of shape :math:`(...)`
        axis2 (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
        angle2 (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        (tuple): tuple containing:

            axis (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
            angle (`jax.numpy.ndarray`): array of shape :math:`(...)`
    """
    return quaternion_to_axis_angle(
        compose_quaternion(
            axis_angle_to_quaternion(axis1, angle1),
            axis_angle_to_quaternion(axis2, angle2),
        )
    )


def rotation_angle_from_axis_angle(axis1, angle1, axis2, angle2):
    r"""Rotation angle between :math:`(\vec x_1, \alpha_1)` and :math:`(\vec x_2, \alpha_2)`.

    Args:
        axis1 (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
        angle1 (`jax.numpy.ndarray`): array of shape :math:`(...)`
        axis2 (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
        angle2 (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(...)`
    """
    return rotation_angle_from_quaternion(
        axis_angle_to_quaternion(axis1, angle1),
        axis_angle_to_quaternion(axis2, angle2),
    )


# log coordinates


def identity_log_coordinates(shape=(), dtype=jnp.float32):
    r"""Log coordinates of identity rotation.

    Args:
        shape: a tuple of nonnegative integers representing the result shape.

    Returns:
        log coordinates (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
    """
    return jnp.zeros(shape + (3,), dtype=dtype)


def rand_log_coordinates(key, shape=(), dtype=jnp.float32):
    r"""Generate random rotation as log coordinates.

    Args:
        key: a PRNGKey used as the random key.
        shape: a tuple of nonnegative integers representing the result shape.

    Returns:
        log coordinates (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
    """
    return axis_angle_to_log_coordinates(*rand_axis_angle(key, shape, dtype))


def compose_log_coordinates(log1, log2):
    r"""Compose :math:`\vec \alpha_1` with :math:`\vec \alpha_2`.

    Args:
        log1 (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
        log2 (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`

    Returns:
        log coordinates (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
    """
    return quaternion_to_log_coordinates(
        compose_quaternion(
            log_coordinates_to_quaternion(log1), log_coordinates_to_quaternion(log2)
        )
    )


def inverse_log_coordinates(log):
    r"""Inverse of log coordinates.

    Args:
        log (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`

    Returns:
        log coordinates (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
    """
    return -log


def rotation_angle_from_log_coordinates(log1, log2):
    r"""Rotation angle between a pair of log coordinates.

    Args:
        log1 (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
        log2 (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(...)`
    """
    return rotation_angle_from_quaternion(
        log_coordinates_to_quaternion(log1),
        log_coordinates_to_quaternion(log2),
    )


# conversions


def matrix_x(angle):
    r"""Matrix of rotation around X axis.

    Args:
        angle (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 3, 3)`
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


def matrix_y(angle):
    r"""Matrix of rotation around Y axis.

    Args:
        angle (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 3, 3)`
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


def matrix_z(angle):
    r"""Matrix of rotation around Z axis.

    Args:
        angle (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 3, 3)`
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    o = jnp.ones_like(angle)
    z = jnp.zeros_like(angle)
    return jnp.stack(
        [
            jnp.stack([c, -s, z], axis=-1),
            jnp.stack([s, c, z], axis=-1),
            jnp.stack([z, z, o], axis=-1),
        ],
        axis=-2,
    )


def angles_to_matrix(alpha, beta, gamma):
    r"""Conversion from angles to matrix.

    Args:
        alpha (`jax.numpy.ndarray`): array of shape :math:`(...)`
        beta (`jax.numpy.ndarray`): array of shape :math:`(...)`
        gamma (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 3, 3)`
    """
    alpha, beta, gamma = jnp.broadcast_arrays(alpha, beta, gamma)
    return matrix_y(alpha) @ matrix_x(beta) @ matrix_y(gamma)


def matrix_to_angles(R):
    r"""Conversion from matrix to angles.
    Warning: this function is not differentiable at rotation angles :math:`\pi`.

    Args:
        R (`jax.numpy.ndarray`): array of shape :math:`(..., 3, 3)`

    Returns:
        (tuple): tuple containing:

            alpha (`jax.numpy.ndarray`): array of shape :math:`(...)`
            beta (`jax.numpy.ndarray`): array of shape :math:`(...)`
            gamma (`jax.numpy.ndarray`): array of shape :math:`(...)`
    """
    # assert jnp.allclose(jnp.linalg.det(R), 1)
    x = R @ jnp.array([0.0, 1.0, 0.0], dtype=R.dtype)
    a, b = xyz_to_angles(x)
    R = jnp.swapaxes(angles_to_matrix(a, b, 0.0), -1, -2) @ R
    c = jnp.arctan2(R[..., 0, 2], R[..., 0, 0])
    return a, b, c


def angles_to_quaternion(alpha, beta, gamma):
    r"""Conversion from angles to quaternion.

    Args:
        alpha (`jax.numpy.ndarray`): array of shape :math:`(...)`
        beta (`jax.numpy.ndarray`): array of shape :math:`(...)`
        gamma (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        q (`jax.numpy.ndarray`): array of shape :math:`(..., 4)`
    """
    alpha, beta, gamma = jnp.broadcast_arrays(alpha, beta, gamma)
    qa = axis_angle_to_quaternion(jnp.array([0.0, 1.0, 0.0], alpha.dtype), alpha)
    qb = axis_angle_to_quaternion(jnp.array([1.0, 0.0, 0.0], beta.dtype), beta)
    qc = axis_angle_to_quaternion(jnp.array([0.0, 1.0, 0.0], gamma.dtype), gamma)
    return compose_quaternion(qa, compose_quaternion(qb, qc))


def matrix_to_quaternion(R):
    r"""Conversion from matrix :math:`R` to quaternion :math:`q`.

    Args:
        R (`jax.numpy.ndarray`): array of shape :math:`(..., 3, 3)`

    Returns:
        q (`jax.numpy.ndarray`): array of shape :math:`(..., 4)`
    """
    return axis_angle_to_quaternion(*matrix_to_axis_angle(R))


def axis_angle_to_quaternion(xyz, angle):
    r"""Conversion from axis-angle to quaternion.

    Args:
        axis (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
        angle (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        q (`jax.numpy.ndarray`): array of shape :math:`(..., 4)`
    """
    angle = jnp.asarray(angle)
    xyz, angle = jnp.broadcast_arrays(xyz, angle[..., None])
    xyz = _normalize(xyz)
    c = jnp.cos(angle[..., :1] / 2)
    s = jnp.sin(angle / 2)
    return jnp.concatenate([c, xyz * s], axis=-1)


def quaternion_to_axis_angle(q):
    r"""Conversion from quaternion to axis-angle.

    Args:
        q (`jax.numpy.ndarray`): array of shape :math:`(..., 4)`

    Returns:
        (tuple): tuple containing:

            axis (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
            angle (`jax.numpy.ndarray`): array of shape :math:`(...)`
    """
    angle = 2 * jnp.arccos(jnp.clip(q[..., 0], -1, 1))
    axis = _normalize(q[..., 1:])
    return axis, angle


def _normalize(x):
    n2 = jnp.sum(x**2, axis=-1, keepdims=True)
    n2 = jnp.where(n2 > 0.0, n2, 1.0)
    return x / jnp.sqrt(n2)


def matrix_to_axis_angle(R):
    r"""Conversion from matrix to axis-angle.
    Warning: this function is not differentiable at rotation angles :math:`\pi`.

    Args:
        R (`jax.numpy.ndarray`): array of shape :math:`(..., 3, 3)`

    Returns:
        (tuple): tuple containing:

            axis (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
            angle (`jax.numpy.ndarray`): array of shape :math:`(...)`
    """
    # assert jnp.allclose(jnp.linalg.det(R), 1)
    angle = rotation_angle_from_matrix(R)
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
    r"""Conversion from angles to axis-angle.

    Args:
        alpha (`jax.numpy.ndarray`): array of shape :math:`(...)`
        beta (`jax.numpy.ndarray`): array of shape :math:`(...)`
        gamma (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        (tuple): tuple containing:

            axis (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
            angle (`jax.numpy.ndarray`): array of shape :math:`(...)`
    """
    return matrix_to_axis_angle(angles_to_matrix(alpha, beta, gamma))


def axis_angle_to_matrix(axis, angle):
    r"""Conversion from axis-angle to matrix.

    Args:
        axis (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
        angle (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 3, 3)`
    """
    axis, angle = jnp.broadcast_arrays(axis, angle[..., None])
    alpha, beta = xyz_to_angles(axis)
    R = angles_to_matrix(alpha, beta, jnp.zeros_like(beta))
    Ry = matrix_y(angle[..., 0])
    return R @ Ry @ jnp.swapaxes(R, -2, -1)


def quaternion_to_matrix(q):
    r"""Conversion from quaternion to matrix.

    Args:
        q (`jax.numpy.ndarray`): array of shape :math:`(..., 4)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 3, 3)`
    """
    return axis_angle_to_matrix(*quaternion_to_axis_angle(q))


def quaternion_to_angles(q):
    r"""Conversion from quaternion to angles.

    Args:
        q (`jax.numpy.ndarray`): array of shape :math:`(..., 4)`

    Returns:
        (tuple): tuple containing:

            alpha (`jax.numpy.ndarray`): array of shape :math:`(...)`
            beta (`jax.numpy.ndarray`): array of shape :math:`(...)`
            gamma (`jax.numpy.ndarray`): array of shape :math:`(...)`
    """
    return matrix_to_angles(quaternion_to_matrix(q))


def axis_angle_to_angles(axis, angle):
    r"""Conversion from axis-angle to angles.

    Args:
        axis (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
        angle (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        (tuple): tuple containing:

            alpha (`jax.numpy.ndarray`): array of shape :math:`(...)`
            beta (`jax.numpy.ndarray`): array of shape :math:`(...)`
            gamma (`jax.numpy.ndarray`): array of shape :math:`(...)`
    """
    return matrix_to_angles(axis_angle_to_matrix(axis, angle))


def log_coordinates_to_matrix(log_coordinates: jnp.ndarray) -> jnp.ndarray:
    r"""Conversion from log coordinates to matrix.

    Args:
        log_coordinates (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 3, 3)`
    """
    shape = log_coordinates.shape[:-1]
    log_coordinates = log_coordinates.reshape(-1, 3)

    X = jnp.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],  # zy
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],  # xz
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # yx
        ],
        dtype=log_coordinates.dtype,
    )

    R = jax.vmap(jax.scipy.linalg.expm)(jnp.einsum("aij,za->zij", X, log_coordinates))
    return R.reshape(shape + (3, 3))


def log_coordinates_to_axis_angle(log_coordinates):
    r"""Conversion from log coordinates to axis-angle.

    Args:
        log_coordinates (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`

    Returns:
        (tuple): tuple containing:

            axis (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
            angle (`jax.numpy.ndarray`): array of shape :math:`(...)`
    """
    n2 = jnp.sum(log_coordinates**2, axis=-1)
    n2_ = jnp.where(n2 > 0.0, n2, 1.0)
    axis = log_coordinates / jnp.sqrt(n2_)[..., None]
    angle = jnp.where(n2 > 0.0, jnp.sqrt(n2_), 0.0)
    return axis, angle


def log_coordinates_to_quaternion(log_coordinates):
    r"""Conversion from log coordinates to quaternion.

    Args:
        log_coordinates (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 4)`
    """
    return axis_angle_to_quaternion(*log_coordinates_to_axis_angle(log_coordinates))


def log_coordinates_to_angles(log_coordinates):
    r"""Conversion from log coordinates to angles.

    Args:
        log_coordinates (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`

    Returns:
        (tuple): tuple containing:

            alpha (`jax.numpy.ndarray`): array of shape :math:`(...)`
            beta (`jax.numpy.ndarray`): array of shape :math:`(...)`
            gamma (`jax.numpy.ndarray`): array of shape :math:`(...)`
    """
    return matrix_to_angles(log_coordinates_to_matrix(log_coordinates))


def axis_angle_to_log_coordinates(axis, angle):
    r"""Conversion from axis-angle to log coordinates.

    Args:
        axis (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
        angle (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 3)`
    """
    angle = jnp.asarray(angle)
    axis, angle = jnp.broadcast_arrays(axis, angle[..., None])
    return axis * angle


def matrix_to_log_coordinates(R):
    r"""Conversion from matrix to log coordinates.

    Args:
        R (`jax.numpy.ndarray`): array of shape :math:`(..., 3, 3)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 3)`
    """
    return axis_angle_to_log_coordinates(*matrix_to_axis_angle(R))


def angles_to_log_coordinates(alpha, beta, gamma):
    r"""Conversion from angles to log coordinates.

    Args:
        alpha (`jax.numpy.ndarray`): array of shape :math:`(...)`
        beta (`jax.numpy.ndarray`): array of shape :math:`(...)`
        gamma (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 3)`
    """
    return axis_angle_to_log_coordinates(*angles_to_axis_angle(alpha, beta, gamma))


def quaternion_to_log_coordinates(q):
    r"""Conversion from quaternion to log coordinates.

    Args:
        q (`jax.numpy.ndarray`): array of shape :math:`(..., 4)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 3)`
    """
    return axis_angle_to_log_coordinates(*quaternion_to_axis_angle(q))


# point on the sphere


def angles_to_xyz(alpha, beta):
    r"""Convert :math:`(\alpha, \beta)` into a point :math:`(x, y, z)` on the sphere.

    Args:
        alpha (`jax.numpy.ndarray`): array of shape :math:`(...)`
        beta (`jax.numpy.ndarray`): array of shape :math:`(...)`

    Returns:
        `jax.numpy.ndarray`: array of shape :math:`(..., 3)`

    Examples:
        >>> angles_to_xyz(1.7, 0.0) + 0.0
        Array([0., 1., 0.], dtype=float32, weak_type=True)
    """
    alpha, beta = jnp.broadcast_arrays(alpha, beta)
    x = jnp.sin(beta) * jnp.sin(alpha)
    y = jnp.cos(beta)
    z = jnp.sin(beta) * jnp.cos(alpha)
    return jnp.stack([x, y, z], axis=-1)


def xyz_to_angles(xyz):
    r"""The rotation :math:`R(\alpha, \beta, 0)` such that :math:`\vec r = R \vec e_y`.

    .. math::

        \vec r = R(\alpha, \beta, 0) \vec e_y

        \alpha = \arctan(x/z)

        \beta = \arccos(y)

    Args:
        xyz (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`

    Returns:
        (tuple): tuple containing:

            alpha (`jax.numpy.ndarray`): array of shape :math:`(...)`
            beta (`jax.numpy.ndarray`): array of shape :math:`(...)`
    """
    xyz = _normalize(xyz)
    xyz = jnp.clip(xyz, -1, 1)

    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    x_ = jnp.where((x == 0.0) & (z == 0.0), 0.0, x)
    y_ = jnp.where((x == 0.0) & (z == 0.0), 0.0, y)
    z_ = jnp.where((x == 0.0) & (z == 0.0), 1.0, z)

    beta = jnp.where(y == 1.0, 0.0, jnp.where(y == -1, jnp.pi, jnp.arccos(y_)))
    alpha = jnp.arctan2(x_, z_)
    return alpha, beta
