from functools import partial
from typing import Callable, Optional

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from e3nn_jax import IrrepsArray, elementwise_tensor_product, scalar_activation
from e3nn_jax._src.util.decorators import overload_for_irreps_without_array


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _gate(input: IrrepsArray, even_act, odd_act, even_gate_act, odd_gate_act) -> IrrepsArray:
    scalars, gated = input, None
    for j, (_, ir) in enumerate(input.irreps):
        if ir.l > 0:
            scalars, gated = input.split([j])
            break
    assert scalars.irreps.lmax == 0

    # No gates:
    if gated is None:
        return scalar_activation(scalars, [even_act if ir.p == 1 else odd_act for _, ir in scalars.irreps])

    # Get the scalar gates:
    gates = None
    for i in range(len(scalars.irreps)):
        if scalars.irreps[i:].num_irreps == gated.irreps.num_irreps:
            scalars, gates = scalars.split([i])
            break

    if gates is None:
        raise ValueError(
            f"Gate: did not manage to split the input into scalars, gates and gated. "
            f"({input.irreps}) = ({scalars.irreps}) + ({gated.irreps})."
        )

    scalars = scalar_activation(scalars, [even_act if ir.p == 1 else odd_act for _, ir in scalars.irreps])
    gates = scalar_activation(gates, [even_gate_act if ir.p == 1 else odd_gate_act for _, ir in gates.irreps])

    return e3nn.concatenate([scalars, elementwise_tensor_product(gates, gated)], axis=-1)


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

    - The scalars are on the left side of the input.
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
        The 3 even scalars are used as gates.
        >>> gate("12x0e + 3x0e + 2x1e + 1x2e")
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
