import jax
import jax.numpy as jnp
import opt_einsum


class Zero:
    r"""An array of zeros

    Examples:
        0 + x = x

        >>> Zero((2,)) + jnp.ones((2, 1))
        DeviceArray([[1., 1.],
                     [1., 1.]], dtype=float32)

        0 * x = 0
        >>> Zero((2,)) * jnp.ones((2, 1))
        Zero((2, 2))

        zeinsum
        >>> zeinsum("ij,jk", jnp.ones((6, 7)), Zero((7, 3)))
        Zero((6, 3))

        >>> Zero.to_jax(Zero((2,)))
        DeviceArray([0., 0.], dtype=float32)
    """
    def __init__(self, shape):
        self.shape = shape
        self.dtype = jnp.float32

    def __repr__(self):
        return f"Zero({self.shape})"

    def __mul__(self, other):
        if hasattr(other, 'shape'):
            shape = jnp.broadcast_shapes(self.shape, other.shape)
            return Zero(shape)
        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if isinstance(other, Zero):
            shape = jnp.broadcast_shapes(self.shape, other.shape)
            return Zero(shape)
        if isinstance(other, jnp.ndarray):
            shape = jnp.broadcast_shapes(self.shape, other.shape)
            return jnp.broadcast_to(other, shape)
        return NotImplemented

    def __radd__(self, other):
        return self + other

    @classmethod
    def to_jax(cls, x):
        def fun(y):
            if isinstance(y, cls):
                return jnp.zeros(y.shape)
            return y
        return jax.tree_map(fun, x)


def _einsum_shape(*operands):
    input_subscripts, output_subscript, operands = opt_einsum.parser.parse_einsum_input(operands)
    dims = {
        i: dim
        for ii, op in zip(input_subscripts.split(','), operands)
        for i, dim in zip(ii, op.shape)
    }
    return tuple(dims[i] for i in output_subscript)


def zeinsum(*operands):
    if any(isinstance(x, Zero) for x in operands):
        output_shape = _einsum_shape(*operands)
        return Zero(output_shape)
    return jnp.einsum(*operands)
