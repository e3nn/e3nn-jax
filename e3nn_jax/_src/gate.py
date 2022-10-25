from functools import partial
from typing import Callable, Optional

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from e3nn_jax import IrrepsArray, scalar_activation
from e3nn_jax._src.util.decorators import overload_for_irreps_without_array


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _gate(input: IrrepsArray, even_act, odd_act, even_gate_act, odd_gate_act) -> IrrepsArray:
    act = {1: even_act, -1: odd_act}
    gate_act = {1: even_gate_act, -1: odd_gate_act}

    scalars = input.filtered(["0e", "0o"])
    vectors = input.filtered(lambda mul_ir: mul_ir.ir.l > 0)
    del input

    if vectors.shape[-1] == 0:
        return scalar_activation(scalars, [act[ir.p] for _, ir in scalars.irreps])

    if scalars.irreps.dim < vectors.irreps.num_irreps:
        raise ValueError("The input must have at least as many scalars as the number of non-scalar irreps")

    scalars_extra = scalars.slice_by_mul[: scalars.irreps.dim - vectors.irreps.num_irreps]
    scalars_gates = scalars.slice_by_mul[scalars.irreps.dim - vectors.irreps.num_irreps :]
    del scalars

    scalars_extra = scalar_activation(scalars_extra, [act[ir.p] for _, ir in scalars_extra.irreps])
    scalars_gates = scalar_activation(scalars_gates, [gate_act[ir.p] for _, ir in scalars_gates.irreps])

    return e3nn.concatenate([scalars_extra, scalars_gates * vectors], axis=-1)


@overload_for_irreps_without_array((0,))
def gate(
    input: IrrepsArray,
    even_act: Optional[Callable[[float], float]] = None,
    odd_act: Optional[Callable[[float], float]] = None,
    even_gate_act: Optional[Callable[[float], float]] = None,
    odd_gate_act: Optional[Callable[[float], float]] = None,
) -> IrrepsArray:
    r"""Gate activation function.

    The input is split into scalars that are activated separately, scalars that are used as gates, and non-scalars that are
    multiplied by the gates.

    List of assumptions:

    - The gate scalars are on the right side of the scalars.

    Args:
        input (IrrepsArray): Input data.
        even_act (Callable[[float], float], optional): Activation function for even scalars.
        odd_act (Callable[[float], float], optional): Activation function for odd scalars.
        even_gate_act (Callable[[float], float], optional): Activation function for even gate scalars.
        odd_gate_act (Callable[[float], float], optional): Activation function for odd gate scalars.

    Returns:
        IrrepsArray: Output data.

    Examples:
        The 3 last scalars are used as gates.

        >>> gate("15x0e + 2x1e + 1x2e")
        12x0e+2x1e+1x2e

        Odd scalars used as gates change the parity of the gated quantities:

        >>> gate("12x0e + 3x0o + 2x1e + 1x2e")
        12x0e+2x1o+1x2o

        Without anything to gate, all the scalars are activated:

        >>> gate("12x0e + 3x0o")
        12x0e+3x0o
    """
    assert isinstance(input, IrrepsArray)

    if even_act is None:
        even_act = jax.nn.gelu
    if odd_act is None:
        odd_act = lambda x: (1 - jnp.exp(-(x**2))) * x
    if even_gate_act is None:
        even_gate_act = jax.nn.sigmoid
    if odd_gate_act is None:
        odd_gate_act = jax.nn.tanh

    return _gate(input, even_act, odd_act, even_gate_act, odd_gate_act)
