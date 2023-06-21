from math import prod
from typing import Callable, List

import jax
import jax.numpy as jnp

import e3nn_jax as e3nn


def grad(
    fun: Callable[[e3nn.IrrepsArray], e3nn.IrrepsArray],
    argnums: int = 0,
    has_aux: bool = False,
    regroup_output: bool = True,
) -> e3nn.IrrepsArray:
    r"""Take the gradient of an equivariant function and reduce it into irreps.

    Args:
        fun: An equivariant function.
        argnums: The argument number to differentiate with respect to.
        has_aux: If True, the function returns a tuple of the output and an auxiliary value.
        regroup_output (bool, optional): Regroup the outputs into irreps. Defaults to True.

    Returns:
        The gradient of the function. Also an equivariant function.

    Examples:
        >>> jnp.set_printoptions(precision=3, suppress=True)
        >>> f = grad(lambda x: 0.5 * e3nn.norm(x, squared=True))
        >>> x = e3nn.IrrepsArray("1o", jnp.array([1.0, 2, 3]))
        >>> f(x)
        1x1o [1. 2. 3.]
    """
    if not isinstance(argnums, int):
        raise ValueError("argnums must be an int.")

    def _grad(*args, **kwargs) -> e3nn.IrrepsArray:
        args = list(args)
        x: e3nn.IrrepsArray = args[argnums]
        if not isinstance(x, e3nn.IrrepsArray):
            raise TypeError(f"arg{argnums} must be an e3nn.IrrepsArray.")
        irreps_in = x.irreps
        leading_shape_in = x.shape[:-1]
        x = e3nn.IrrepsArray(x.irreps, x.array)  # drop zero_flags
        args[argnums] = x.chunks

        def naked_fun(*args, **kwargs) -> List[jnp.ndarray]:
            args = list(args)
            args[argnums] = e3nn.from_chunks(
                irreps_in, args[argnums], leading_shape_in, x.dtype
            )
            if has_aux:
                y, aux = fun(*args, **kwargs)
                if not isinstance(y, e3nn.IrrepsArray):
                    raise TypeError(
                        f"Expected equivariant function to return an e3nn.IrrepsArray, got {type(y)}."
                    )
                return y.chunks, (y.irreps, y.shape[:-1], aux)
            else:
                y = fun(*args, **kwargs)
                if not isinstance(y, e3nn.IrrepsArray):
                    raise TypeError(
                        f"Expected equivariant function to return an e3nn.IrrepsArray, got {type(y)}."
                    )
                return y.chunks, (y.irreps, y.shape[:-1])

        output = jax.jacobian(
            naked_fun,
            argnums=argnums,
            has_aux=True,
        )(*args, **kwargs)

        if has_aux:
            jac, (irreps_out, leading_shape_out, aux) = output
        else:
            jac, (irreps_out, leading_shape_out) = output

        irreps = []
        lst = []
        for mir_out, y_list in zip(irreps_out, jac):
            for mir_in, z in zip(irreps_in, y_list):
                assert z.shape == (
                    leading_shape_out
                    + (mir_out.mul, mir_out.ir.dim)
                    + leading_shape_in
                    + (mir_in.mul, mir_in.ir.dim)
                )
                z = jnp.reshape(
                    z,
                    (
                        prod(leading_shape_out),
                        mir_out.mul,
                        mir_out.ir.dim,
                        prod(leading_shape_in),
                        mir_in.mul,
                        mir_in.ir.dim,
                    ),
                )
                for ir in mir_out.ir * mir_in.ir:
                    irreps.append((mir_out.mul * mir_in.mul, ir))
                    lst.append(
                        jnp.einsum(
                            "auibvj,ijk->abuvk",
                            z,
                            jnp.sqrt(ir.dim)
                            * e3nn.clebsch_gordan(mir_out.ir.l, mir_in.ir.l, ir.l),
                        ).reshape(
                            leading_shape_out
                            + leading_shape_in
                            + (mir_out.mul * mir_in.mul, ir.dim)
                        )
                    )
        output = e3nn.from_chunks(
            irreps, lst, leading_shape_out + leading_shape_in, x.dtype
        )
        if regroup_output:
            output = output.regroup()
        if has_aux:
            return output, aux
        else:
            return output

    return _grad
