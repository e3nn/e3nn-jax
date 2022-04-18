from typing import Callable, List, Optional

import jax
import jax.numpy as jnp

from e3nn_jax import Irreps, IrrepsData
from e3nn_jax.util.decorators import overload_for_irreps_without_data


def normalize_function(phi):
    with jax.ensure_compile_time_eval():
        k = jax.random.PRNGKey(0)
        x = jax.random.normal(k, (1_000_000,))
        c = jnp.mean(phi(x) ** 2) ** 0.5

        if jnp.allclose(c, 1.0):
            return phi
        else:

            def rho(x):
                return phi(x) / c

            return rho


def parity_function(phi):
    with jax.ensure_compile_time_eval():
        x = jnp.linspace(0.0, 10.0, 256)

        a1, a2 = phi(x), phi(-x)
        if jnp.max(jnp.abs(a1 - a2)) < 1e-5:
            return 1
        elif jnp.max(jnp.abs(a1 + a2)) < 1e-5:
            return -1
        else:
            return 0


def is_zero_in_zero(phi):
    with jax.ensure_compile_time_eval():
        return jnp.allclose(phi(jnp.array(0.0)), 0.0)


@overload_for_irreps_without_data(irrepsdata_argnums=[0])
def scalar_activation(input: IrrepsData, acts: List[Optional[Callable[[float], float]]]) -> IrrepsData:
    assert isinstance(input, IrrepsData)

    assert len(input.irreps) == len(acts), (input.irreps, acts)

    list = []

    irreps_out = []
    for (mul, (l_in, p_in)), x, act in zip(input.irreps, input.list, acts):
        if act is not None:
            if l_in != 0:
                raise ValueError(
                    f"Activation: cannot apply an activation function to a non-scalar input. {input.irreps} {acts}"
                )

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
                    list.append(act(jnp.ones(input.shape + (mul, 1))))
            else:
                list.append(act(x))
        else:
            irreps_out.append((mul, (l_in, p_in)))
            list.append(x)

    irreps_out = Irreps(irreps_out)

    if acts and acts.count(acts[0]) == len(acts):
        # for performance, if all the activation functions are the same, we can apply it to the contiguous array as well
        contiguous = input.contiguous if acts[0] is None else normalize_function(acts[0])(input.contiguous)
        return IrrepsData(irreps=irreps_out, contiguous=contiguous, list=list)

    return IrrepsData.from_list(irreps_out, list, input.shape)


# TODO remove this class and follow the same pattern as scalar_activation
class KeyValueActivation:
    irreps_key: Irreps
    irreps_value: Irreps
    irreps_out: Irreps

    def __init__(self, irreps_key, irreps_value, phi):
        self.irreps_key = Irreps(irreps_key)
        self.irreps_value = Irreps(irreps_value)

        # TODO compute irreps_out
        # irreps_out =

    def __call__(self, keys, values):
        return [jax.vmap(key_value_activation)(self.phi, k, v) for k, v in zip(keys, values)]


def key_value_activation(phi, key, value):
    assert key.ndim == 1
    assert value.ndim == 1

    d = value.shape[0]
    key = key / jnp.sqrt(1 / 16 + jnp.sum(key ** 2))  # 1/16 is arbitrary small... but not too small...
    scalar = jnp.sum(key * value)
    scalar = normalize_function(phi)(scalar)
    return d ** 0.5 * scalar * key  # component normalized
