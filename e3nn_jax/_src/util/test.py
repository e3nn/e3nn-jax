from typing import Callable, Optional, Tuple

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np


def equivariance_test(
    fun: Callable[[e3nn.IrrepsArray], e3nn.IrrepsArray],
    rng_key: jnp.ndarray,
    *args,
):
    jax_enable_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)

    assert all(isinstance(arg, e3nn.IrrepsArray) for arg in args)

    R = -e3nn.rand_matrix(rng_key, ())  # random rotation and inversion

    out1 = fun(*[arg.transform_by_matrix(R) for arg in args])
    out2 = fun(*args).transform_by_matrix(R)

    jax.config.update("jax_enable_x64", jax_enable_x64)
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
