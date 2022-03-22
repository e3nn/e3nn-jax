from typing import Callable, List, Optional

import jax

from e3nn_jax import IrrepsData, elementwise_tensor_product, scalar_activation
from e3nn_jax.util.decorators import overload_for_irreps_without_data


@overload_for_irreps_without_data((0,))
def gate(input: IrrepsData, acts: List[Optional[Callable]] = None, even_act=jax.nn.gelu, odd_act=jax.nn.tanh, even_gate_act=jax.nn.sigmoid, odd_gate_act=jax.nn.tanh) -> IrrepsData:
    r"""Gate activation function.

    The input is split into scalars that are activated separately, scalars that are used as gates, and non-scalars that are multiplied by the gates.

    List of assumptions:

    - The scalars are on the left side of the input.
    - The gate scalars are on the right side of the scalars.

    Args:
        input (IrrepsData): Input data.
        acts: The list of activation functions. Its length must be equal to the number of scalar blocks in the input.

    Returns:
        IrrepsData: Output data.
    """
    assert isinstance(input, IrrepsData)

    if acts is not None:
        j = len(acts)
    else:
        j = 0
        for j, (_, ir) in enumerate(input.irreps):
            if not ir.is_scalar:
                break
    scalars, gated = input.split([j])

    if acts is not None:
        scalars = scalar_activation(scalars, acts)

    if gated.irreps.dim == 0:
        if acts is None:
            scalars = scalar_activation(scalars, [even_act if ir.p == 1 else odd_act for _, ir in scalars.irreps])
        return scalars

    gates = None
    for i in range(j + 1):
        if scalars.irreps[i:].num_irreps == gated.irreps.num_irreps:
            scalars, gates = scalars.split([i])
            break

    if gates is None:
        raise ValueError(f"Gate: did not manage to split the input {input.irreps} into scalars, gates and gated.")

    if acts is None:
        scalars = scalar_activation(scalars, [even_act if ir.p == 1 else odd_act for _, ir in scalars.irreps])
        gates = scalar_activation(gates, [even_gate_act if ir.p == 1 else odd_gate_act for _, ir in gates.irreps])

    return IrrepsData.cat([scalars, elementwise_tensor_product(gates, gated)])
