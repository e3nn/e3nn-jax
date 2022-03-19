from typing import Callable, List, Optional

from e3nn_jax import IrrepsData, elementwise_tensor_product, scalar_activation
from e3nn_jax.util.decorators import overload_for_irreps_without_data


@overload_for_irreps_without_data((0,))
def gate(input: IrrepsData, acts: List[Optional[Callable]]) -> IrrepsData:
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

    input = scalar_activation(input, acts + [None] * (len(input.irreps) - len(acts)))

    scalars = None
    j = len(acts)
    irreps_gated = input.irreps[j:]
    for i in range(j + 1):
        if input.irreps[i:j].num_irreps == irreps_gated.num_irreps:
            scalars, gates, gated = input.split([i, j])
            break

    if scalars is None:
        raise ValueError(f"Gate: did not manage to split the input {input.irreps} into scalars, gates and gated.")

    return IrrepsData.cat([scalars, elementwise_tensor_product(gates, gated)])
