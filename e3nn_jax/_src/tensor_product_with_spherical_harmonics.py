import e3nn_jax as e3nn
import jax.numpy as jnp
import jax


def tensor_product_with_spherical_harmonics(
    input: e3nn.IrrepsArray, vector: e3nn.IrrepsArray, degree: int
) -> e3nn.IrrepsArray:
    """Tensor product of something with the spherical harmonics of a vector.

    The idea of this optimization comes from the paper::

        Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs

    Args:
        input (IrrepsArray): input
        vector (IrrepsArray): vector, irreps must be "1o" or "1e"
        degree (int): the maximum degree of the spherical harmonics

    Returns:
        IrrepsArray: tensor product

    Notes:
        This function is equivalent to::

            tensor_product(input, spherical_harmonics(range(degree + 1), vector, True))

    Examples:
        >>> input = e3nn.normal("3x0e + 2x1o", jax.random.PRNGKey(0))
        >>> vector = e3nn.normal("1e", jax.random.PRNGKey(1))
        >>> degree = 2
        >>> output1 = tensor_product_with_spherical_harmonics(input, vector, degree)
        >>> output2 = e3nn.tensor_product(input, e3nn.spherical_harmonics(range(degree + 1), vector, True))
        >>> assert output1.irreps == output2.irreps
        >>> assert jnp.allclose(output1.array, output2.array, atol=1e-6)
    """
    input = e3nn.as_irreps_array(input)

    if not (vector.irreps == "1o" or vector.irreps == "1e"):
        raise ValueError(
            "tensor_product_with_spherical_harmonics: vector must be a vector."
        )

    leading_shape = jnp.broadcast_shapes(input.shape[:-1], vector.shape[:-1])
    input = input.broadcast_to(leading_shape + (-1,))
    vector = vector.broadcast_to(leading_shape + (-1,))

    f = impl
    for _ in range(len(leading_shape)):
        f = e3nn.utils.vmap(f, in_axes=(0, 0, None), out_axes=0)

    return f(input, vector, degree)


def impl(
    input: e3nn.IrrepsArray, vector: e3nn.IrrepsArray, degree: int
) -> e3nn.IrrepsArray:
    """
    This implementation looks like a lot of operations, but actually only few lines are
    traced by JAX. They are indicated by the comment `# <-- ops`.
    """
    assert input.shape == (input.irreps.dim,)
    assert vector.shape == (3,)
    vector = e3nn.IrrepsArray(vector.irreps, normalize(vector.array))

    # Avoid gimbal lock
    gimbal_lock = jnp.abs(vector.array[1]) > 0.99

    def fix_gimbal_lock(array, inverse):
        array_rot = array.transform_by_angles(0.0, jnp.pi / 2.0, 0.0, inverse=inverse)
        return jax.tree_util.tree_map(
            lambda x_rot, x: jnp.where(gimbal_lock, x_rot, x), array_rot, array
        )

    input = fix_gimbal_lock(input, inverse=True)  # <-- ops
    vector = fix_gimbal_lock(vector, inverse=True)  # <-- ops

    # Calculate the rotation and align the input with the vector axis
    alpha, beta = e3nn.xyz_to_angles(vector.array)  # <-- ops
    input = input.transform_by_angles(alpha, beta, 0.0, inverse=True)  # <-- ops

    # Compute the spherical harmonics but only at compilation time
    with jax.ensure_compile_time_eval():
        vector = e3nn.IrrepsArray(vector.irreps, jnp.array([0.0, 1.0, 0.0]))
        shs = e3nn.spherical_harmonics(range(degree + 1), vector, True)

    irreps_out = []
    outputs = []

    for (mul, irx), x in zip(input.irreps, input.chunks):
        assert x.shape == (mul, irx.dim)
        for (_, iry), y in zip(shs.irreps, shs.chunks):
            assert y.shape == (1, iry.dim)

            # Verify that the spherical harmonics have a specific form
            with jax.ensure_compile_time_eval():
                y = y.squeeze(0)
                assert jnp.allclose(y.at[iry.l].set(0.0), 0.0)
                y = y[iry.l]  # keep only the central non-zero value

            for irz in irx * iry:
                irreps_out.append((mul, irz))

                if x is None:
                    outputs.append(None)
                    continue

                # Compute the Clebsch-Gordan coefficients and normalize them
                l = min(irx.l, irz.l)
                with jax.ensure_compile_time_eval():
                    cg = jnp.sqrt(irz.dim) * e3nn.clebsch_gordan(irx.l, iry.l, irz.l)
                    cg = cg[sl(irx.l, l), iry.l, sl(irz.l, l)]

                    # Verify that the Clebsch-Gordan coefficients have a specific form
                    diag = is_diag(cg)
                    if not diag:
                        cg = cg[::-1, :]
                        assert is_diag(cg)

                    cg = y * jnp.diag(cg)
                    cg = cg.astype(x.dtype)
                    assert cg.shape == (2 * l + 1,)

                xx = x[:, sl(irx.l, l)]  # <-- ops
                assert xx.shape == (mul, 2 * l + 1)

                if not diag:
                    xx = xx[:, ::-1]  # <-- ops

                out = xx * cg  # <-- ops

                if l < irz.l:
                    zeros = jnp.zeros_like(out, shape=(mul, irz.dim))  # <-- ops
                    out = zeros.at[:, sl(irz.l, l)].set(out)  # <-- ops

                outputs.append(out)

    out = e3nn.from_chunks(irreps_out, outputs, (), x.dtype)
    out = out.regroup()  # <-- ops
    out = out.transform_by_angles(alpha, beta, 0.0)  # <-- ops

    # Avoid gimbal lock
    out = fix_gimbal_lock(out, inverse=False)  # <-- ops

    return out


def sl(lout: int, lin: int) -> slice:
    return slice(lout - lin, lout + lin + 1)


def is_diag(x: jnp.ndarray) -> bool:
    return jnp.allclose(jnp.diag(jnp.diag(x)), x)


def normalize(x):
    n2 = jnp.sum(x**2, axis=-1, keepdims=True)
    n2 = jnp.where(n2 > 0.0, n2, 1.0)
    return x / jnp.sqrt(n2)
