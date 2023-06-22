from typing import Callable

import jax

import e3nn_jax as e3nn
from e3nn_jax import IrrepsArray, scalar_activation
from e3nn_jax._src.utils.decorators import overload_for_irreps_without_array


def _gate(
    input: IrrepsArray, even_act, odd_act, even_gate_act, odd_gate_act, normalize_act
) -> IrrepsArray:
    scalars = input.filter(keep=["0e", "0o"])
    vectors = input.filter(drop=["0e", "0o"])
    del input

    if vectors.shape[-1] == 0:
        return scalar_activation(
            scalars, even_act=even_act, odd_act=odd_act, normalize_act=normalize_act
        )

    if scalars.irreps.dim < vectors.irreps.num_irreps:
        raise ValueError(
            "The input must have at least as many scalars as the number of non-scalar irreps"
        )

    scalars_extra: e3nn.IrrepsArray = scalars.slice_by_mul[
        : scalars.irreps.dim - vectors.irreps.num_irreps
    ]
    scalars_gates: e3nn.IrrepsArray = scalars.slice_by_mul[
        scalars.irreps.dim - vectors.irreps.num_irreps :
    ]
    del scalars

    scalars_extra = scalar_activation(
        scalars_extra, even_act=even_act, odd_act=odd_act, normalize_act=normalize_act
    )
    scalars_gates = scalar_activation(
        scalars_gates,
        even_act=even_gate_act,
        odd_act=odd_gate_act,
        normalize_act=normalize_act,
    )

    return e3nn.concatenate([scalars_extra, scalars_gates * vectors], axis=-1)


@overload_for_irreps_without_array((0,))
def gate(
    input: IrrepsArray,
    even_act: Callable[[float], float] = jax.nn.gelu,
    odd_act: Callable[[float], float] = e3nn.soft_odd,
    even_gate_act: Callable[[float], float] = jax.nn.sigmoid,
    odd_gate_act: Callable[[float], float] = jax.nn.tanh,
    normalize_act: bool = True,
) -> IrrepsArray:
    r"""Gate activation function.

    The input is split into scalars that are activated separately, scalars that are used as gates, and non-scalars that are
    multiplied by the gates.

    List of assumptions:

    - The gate scalars are on the right side of the scalars.

    Args:
        input (IrrepsArray): Input data.
        even_act (Callable[[float], float]): Activation function for even scalars. Default: :func:`jax.nn.gelu`.
        odd_act (Callable[[float], float]): Activation function for odd scalars. Default: :math:`(1 - \exp(-x^2)) x`.
        even_gate_act (Callable[[float], float]): Activation function for even gate scalars. Default: :func:`jax.nn.sigmoid`.
        odd_gate_act (Callable[[float], float]): Activation function for odd gate scalars. Default: :func:`jax.nn.tanh`.
        normalize_act (bool): If True, the activation functions are normalized using `e3nn_jax.normalize_function`.

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

    return _gate(input, even_act, odd_act, even_gate_act, odd_gate_act, normalize_act)
