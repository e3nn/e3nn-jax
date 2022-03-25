from functools import partial

import jax
import jax.numpy as jnp

from e3nn_jax import IrrepsData, elementwise_tensor_product, scalar_activation
from e3nn_jax.util.decorators import overload_for_irreps_without_data


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _gate(input: IrrepsData, even_act, odd_act, even_gate_act, odd_gate_act) -> IrrepsData:
    # split l=0 vs l>0
    j = 0
    for j, (_, ir) in enumerate(input.irreps):
        if ir.l > 0:
            break
    scalars, gated = input.split([j])
    assert scalars.irreps.lmax == 0

    # apply scalar activation if there is no gate
    if gated.irreps.dim == 0:
        scalars = scalar_activation(scalars, [even_act if ir.p == 1 else odd_act for _, ir in scalars.irreps])
        return scalars

    # extract gates from scalars
    gates = None
    for i in range(j + 1):
        if scalars.irreps[i:].num_irreps == gated.irreps.num_irreps:
            scalars, gates = scalars.split([i])
            break

    if gates is None:
        raise ValueError(f"Gate: did not manage to split the input {input.irreps} into scalars, gates and gated.")

    scalars = scalar_activation(scalars, [even_act if ir.p == 1 else odd_act for _, ir in scalars.irreps])
    gates = scalar_activation(gates, [even_gate_act if ir.p == 1 else odd_gate_act for _, ir in gates.irreps])

    return IrrepsData.cat([scalars, elementwise_tensor_product(gates, gated)])


@overload_for_irreps_without_data((0,))
def gate(input: IrrepsData, even_act=None, odd_act=None, even_gate_act=None, odd_gate_act=None) -> IrrepsData:
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

    Examples:
        >>> gate("12x0e + 3x0e + 2x1e + 1x2e")
        12x0e+2x1e+1x2e
    """
    assert isinstance(input, IrrepsData)

    if even_act is None:
        even_act = jax.nn.gelu
    if odd_act is None:
        odd_act = lambda x: (1 - jnp.exp(-x**2)) * x
    if even_gate_act is None:
        even_gate_act = jax.nn.sigmoid
    if odd_gate_act is None:
        odd_gate_act = jax.nn.tanh

    return _gate(input, even_act, odd_act, even_gate_act, odd_gate_act)
