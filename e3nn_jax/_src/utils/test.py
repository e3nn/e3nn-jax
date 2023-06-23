from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

import e3nn_jax as e3nn
from e3nn_jax._src.utils.dtype import get_pytree_dtype


def equivariance_test(
    fun: Callable[[e3nn.IrrepsArray], e3nn.IrrepsArray],
    rng_key: jnp.ndarray,
    *args,
):
    r"""Test equivariance of a function.

    Args:
        fun: function to test
        rng_key: random number generator key
        *args: arguments to pass to fun, can be IrrepsArray or Irreps
            if an argument is Irreps, it will be replaced by a random IrrepsArray

    Returns:
        out1, out2: outputs of fun(R args) and R fun(args) for a random rotation R and inversion

    Example:
        >>> fun = e3nn.norm
        >>> rng = jax.random.PRNGKey(0)
        >>> x = e3nn.IrrepsArray("1e", jnp.array([0.0, 4.0, 3.0]))
        >>> equivariance_test(fun, rng, x)
        (1x0e [5.], 1x0e [5.])
    """
    args = [e3nn.Irreps(arg) if isinstance(arg, str) else arg for arg in args]
    args = [
        e3nn.as_irreps_array(arg) if isinstance(arg, jnp.ndarray) else arg
        for arg in args
    ]

    assert all(isinstance(arg, (e3nn.Irreps, e3nn.IrrepsArray)) for arg in args)
    dtype = get_pytree_dtype(args, real_part=True)
    if dtype.kind == "i":
        dtype = jnp.float32

    new_args = []
    for arg in args:
        if isinstance(arg, e3nn.Irreps):
            k, rng_key = jax.random.split(rng_key)
            arg = e3nn.normal(arg, k, dtype=dtype)
        new_args.append(arg)
    args = tuple(new_args)

    R = -e3nn.rand_matrix(rng_key, (), dtype=dtype)  # random rotation and inversion

    out1 = fun(*[arg.transform_by_matrix(R) for arg in args])
    out2 = fun(*args).transform_by_matrix(R)

    return out1, out2


def assert_equivariant(
    fun: Callable[[e3nn.IrrepsArray], e3nn.IrrepsArray],
    rng_key: jnp.ndarray,
    *args,
    atol: float = 1e-6,
    rtol: float = 1e-6,
):
    r"""Assert that a function is equivariant.

    Args:
        fun: function to test
        rng_key: random number generator key
        *args: arguments to pass to fun, can be IrrepsArray or Irreps
            if an argument is Irreps, it will be replaced by a random IrrepsArray
        atol: absolute tolerance
        rtol: relative tolerance

    Examples:
        >>> fun = e3nn.norm
        >>> rng = jax.random.PRNGKey(0)
        >>> x = e3nn.IrrepsArray("1e", jnp.array([0.0, 4.0, 3.0]))
        >>> assert_equivariant(fun, rng, x)

        We can also pass the irreps of the inputs instead of the inputs themselves:
        >>> assert_equivariant(fun, rng, "1e")
    """
    out1, out2 = equivariance_test(fun, rng_key, *args)

    def assert_(x, y):
        np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)

    jax.tree_util.tree_map(assert_, out1, out2)


def assert_output_dtype_matches_input_dtype(fun: Callable, *args, **kwargs):
    """Checks that the dtype of ``fun(*args, **kwargs)`` matches that of the input ``(*args, **kwargs)``.

    Args:
        fun: function to test
        *args: arguments to pass to fun
        **kwargs: keyword arguments to pass to fun

    Raises:
        AssertionError: if the dtype of fun(*args, **kwargs) does not match that of the input (*args, **kwargs).
    """
    if not jax.config.read("jax_enable_x64"):
        raise ValueError("This test requires jax_enable_x64=True")

    dtype = get_pytree_dtype(args, kwargs, real_part=True)
    assert (
        get_pytree_dtype(
            jax.eval_shape(fun, *args, **kwargs), default_dtype=dtype, real_part=True
        )
        == dtype
    )

    def astype(x, dtype):
        if x.dtype.kind == "f":
            return x.astype(dtype)
        if x.dtype.kind == "c":
            return x.real.astype(dtype) + 1j * x.imag.astype(dtype)
        return x

    for dtype in [jnp.float32, jnp.float64]:
        args = jax.tree_util.tree_map(lambda x: astype(x, dtype), args)
        kwargs = jax.tree_util.tree_map(lambda x: astype(x, dtype), kwargs)

        out = jax.eval_shape(fun, *args, **kwargs)
        if get_pytree_dtype(out, default_dtype=dtype, real_part=True) != dtype:
            in_dtype = jax.tree_util.tree_map(lambda x: x.dtype, args)
            out_dtype = jax.tree_util.tree_map(lambda x: x.dtype, out)

            raise AssertionError(
                f"Expected {dtype} -> {dtype}. Got {in_dtype} -> {out_dtype}"
            )
