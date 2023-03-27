from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
from e3nn_jax._src.util.decorators import overload_for_irreps_without_array


def normalize_function(phi: Callable[[float], float]) -> Callable[[float], float]:
    r"""Normalize a function, :math:`\psi(x)=\phi(x)/c` where :math:`c` is the normalization constant such that

    .. math::

        \int_{-\infty}^{\infty} \psi(x)^2 dx = 1
    """
    with jax.ensure_compile_time_eval():
        k = jax.random.PRNGKey(0)
        x = jax.random.normal(k, (1_000_000,), dtype=jnp.float64)
        c = jnp.mean(phi(x) ** 2) ** 0.5
        c = c.item()

        if jnp.allclose(c, 1.0):
            return phi
        else:

            def rho(x):
                return phi(x) / c

            return rho


def parity_function(phi: Callable[[float], float]) -> int:
    with jax.ensure_compile_time_eval():
        x = jnp.linspace(0.0, 10.0, 256)

        a1, a2 = phi(x), phi(-x)
        if jnp.max(jnp.abs(a1 - a2)) < 1e-5:
            return 1
        elif jnp.max(jnp.abs(a1 + a2)) < 1e-5:
            return -1
        else:
            return 0


def is_zero_in_zero(phi: Callable[[float], float]) -> bool:
    with jax.ensure_compile_time_eval():
        return jnp.allclose(phi(jnp.array(0.0)), 0.0)


@overload_for_irreps_without_array(irrepsarray_argnums=[0])
def scalar_activation(
    input: e3nn.IrrepsArray, acts: List[Optional[Callable[[float], float]]], *, normalize_act: bool = True
) -> e3nn.IrrepsArray:
    r"""Apply activation functions to the scalars of an `IrrepsArray`.
    The activation functions are by default normalized.

    Args:
        input (IrrepsArray): input array
        acts (list of functions): list of activation functions, one for each chunk of the input
        normalize_act (bool): if True, normalize the activation functions using `normalize_function`

    Returns:
        IrrepsArray: output array

    Examples:
        >>> x = e3nn.IrrepsArray("0e + 0o + 1o", jnp.array([1.0, 0.0, 1.0, 1.0, 2.0]))
        >>> scalar_activation(x, [jnp.exp, jnp.sin, None])
        1x0e+1x0o+1x1o [1.0021242 0.        1.        1.        2.       ]

        >>> scalar_activation(x, [jnp.exp, jnp.cos, None])
        1x0e+1x0e+1x1o [1.0021242 1.3270178 1.        1.        2.       ]

    Note:
        The parity of the output depends on the parity of the activation function.
    """
    assert isinstance(input, e3nn.IrrepsArray)

    assert len(input.irreps) == len(acts), (input.irreps, acts)

    list = []

    irreps_out = []
    for (mul, (l_in, p_in)), x, act in zip(input.irreps, input.list, acts):
        if act is not None:
            if l_in != 0:
                raise ValueError(
                    f"Activation: cannot apply an activation function to a non-scalar input. {input.irreps} {acts}"
                )

            if normalize_act:
                act = normalize_function(act)

            p_out = parity_function(act) if p_in == -1 else p_in
            if p_out == 0:
                raise ValueError(
                    "Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd."
                )

            irreps_out.append((mul, (0, p_out)))
            if x is None:
                if is_zero_in_zero(act):
                    list.append(None)
                else:
                    list.append(act(jnp.ones(input.shape[:-1] + (mul, 1), input.dtype)))
            else:
                list.append(act(x))
        else:
            irreps_out.append((mul, (l_in, p_in)))
            list.append(x)

    irreps_out = e3nn.Irreps(irreps_out)

    # for performance, if all the activation functions are the same, we can apply it to the contiguous array as well:
    if acts and acts.count(acts[0]) == len(acts):
        if acts[0] is None:
            array = input.array
        else:
            act = acts[0]
            if normalize_act:
                act = normalize_function(act)
            array = act(input.array)
        return e3nn.IrrepsArray(irreps=irreps_out, array=array, list=list)

    return e3nn.IrrepsArray.from_list(irreps_out, list, input.shape[:-1], input.dtype)


def key_value_activation(phi, key, value):
    assert key.ndim == 1
    assert value.ndim == 1

    d = value.shape[0]
    key = key / jnp.sqrt(1 / 16 + jnp.sum(key**2))  # 1/16 is arbitrary small... but not too small...
    scalar = jnp.sum(key * value)
    scalar = normalize_function(phi)(scalar)
    return d**0.5 * scalar * key  # component normalized
