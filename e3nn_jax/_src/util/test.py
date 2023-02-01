from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import e3nn_jax as e3nn
from e3nn_jax._src.util.dtype import get_pytree_dtype


def equivariance_test(
    fun: Callable[[e3nn.IrrepsArray], e3nn.IrrepsArray],
    rng_key: jnp.ndarray,
    *args,
):
    assert all(isinstance(arg, e3nn.IrrepsArray) for arg in args)
    dtype = get_pytree_dtype(args)
    if dtype.kind == "i":
        dtype = jnp.float32

    R = -e3nn.rand_matrix(rng_key, (), dtype=dtype)  # random rotation and inversion

    out1 = fun(*[arg.transform_by_matrix(R) for arg in args])
    out2 = fun(*args).transform_by_matrix(R)

    return out1, out2


def assert_equivariant(
    fun: Callable[[e3nn.IrrepsArray], e3nn.IrrepsArray],
    rng_key: jnp.ndarray,
    *,
    args_in: Optional[Tuple[e3nn.IrrepsArray, ...]] = None,
    irreps_in: Optional[Tuple[e3nn.Irreps, ...]] = None,
    atol: float = 1e-6,
    rtol: float = 1e-6,
):
    if args_in is None and irreps_in is None:
        raise ValueError("Either args_in or irreps_in must be provided")

    if args_in is None:
        args_in = [e3nn.normal(irreps, rng_key, ()) for irreps in irreps_in]

    out1, out2 = equivariance_test(fun, rng_key, *args_in)

    def assert_(x, y):
        np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)

    jax.tree_util.tree_map(assert_, out1, out2)


def assert_output_dtype_matches_input_dtype(fun: Callable, *args, **kwargs):
    """Checks that the dtype of fun(*args, **kwargs) matches that of the input (*args, **kwargs)."""
    if not jax.config.read("jax_enable_x64"):
        raise ValueError("This test requires jax_enable_x64=True")

    dtype = get_pytree_dtype(args, kwargs, real_part=True)
    assert get_pytree_dtype(jax.eval_shape(fun, *args, **kwargs), default_dtype=dtype, real_part=True) == dtype

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

            raise AssertionError(f"Expected {dtype} -> {dtype}. Got {in_dtype} -> {out_dtype}")
