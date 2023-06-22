from typing import Callable, List, Optional

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

import e3nn_jax as e3nn
from e3nn_jax._src.utils.decorators import overload_for_irreps_without_array


def soft_odd(x):
    """Smooth odd function that can be used as activation function for odd scalars.

    .. math::

        x (1 - e^{-x^2})

    Note:
        Odd scalars (l=0 and p=-1) has to be activated by functions with well defined parity:

        * even (:math:`f(-x)=f(x)`)
        * odd (:math:`f(-x)=-f(x)`).
    """
    return (1 - jnp.exp(-(x**2))) * x


def normalspace(n: int) -> jnp.ndarray:
    r"""Sequence of normally distributed numbers :math:`x_i` for :math:`i=1, \ldots, n` such that

    .. math::

        \int_{-\infty}^{x_i} \phi(x) dx = \frac{i}{n+1}

    where :math:`\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}` is the normal distribution.

    Args:
        n (int): Number of points

    Returns:
        jnp.ndarray: Sequence of normally distributed numbers

    Examples:
        >>> normalspace(5)
        Array([-0.96742135, -0.4307273 ,  0.        ,  0.43072742,  0.96742165],      dtype=float32)
    """
    return jnp.sqrt(2) * jsp.erfinv(jnp.linspace(-1.0, 1.0, n + 2)[1:-1])


def normalize_function(phi: Callable[[float], float]) -> Callable[[float], float]:
    r"""Normalize a function, :math:`\psi(x)=\phi(x)/c` where :math:`c` is the normalization constant such that

    .. math::

        \int_{-\infty}^{\infty} \psi(x)^2 \frac{e^{-x^2/2}}{\sqrt{2\pi}} dx = 1
    """
    with jax.ensure_compile_time_eval():
        # k = jax.random.PRNGKey(0)
        # x = jax.random.normal(k, (1_000_000,))
        x = normalspace(1_000_001)
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
    input: e3nn.IrrepsArray,
    acts: List[Optional[Callable[[float], float]]] = None,
    *,
    even_act: Callable[[float], float] = jax.nn.gelu,
    odd_act: Callable[[float], float] = soft_odd,
    normalize_act: bool = True,
) -> e3nn.IrrepsArray:
    r"""Apply activation functions to the scalars of an `IrrepsArray`.
    The activation functions are by default normalized.

    Args:
        input (IrrepsArray): input array
        acts (optional, list of functions): list of activation functions, one for each chunk of the input
        even_act (Callable[[float], float]): Activation function for even scalars. Default: :func:`jax.nn.gelu`.
        odd_act (Callable[[float], float]): Activation function for odd scalars. Default: :math:`(1 - \exp(-x^2)) x`.
        normalize_act (bool): if True, normalize the activation functions using `normalize_function`

    Returns:
        IrrepsArray: output array

    Examples:
        >>> x = e3nn.IrrepsArray("0e + 0o + 1o", jnp.array([1.0, 0.0, 1.0, 1.0, 2.0]))
        >>> scalar_activation(x, [jnp.exp, jnp.sin, None])
        1x0e+1x0o+1x1o [1.0010498 0.        1.        1.        2.       ]

        >>> scalar_activation(x, [jnp.exp, jnp.cos, None])
        1x0e+1x0e+1x1o [1.0010498 1.3272501 1.        1.        2.       ]

    Note:
        The parity of the output depends on the parity of the activation function.
    """
    input = e3nn.as_irreps_array(input)
    assert isinstance(input, e3nn.IrrepsArray)

    if acts is None:
        acts = [
            {1: even_act, -1: odd_act}[ir.p] if ir.l == 0 else None
            for _, ir in input.irreps
        ]

    assert len(input.irreps) == len(acts), (input.irreps, acts)

    chunks = []

    irreps_out = []
    for (mul, (l_in, p_in)), x, act in zip(input.irreps, input.chunks, acts):
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
                    chunks.append(None)
                else:
                    chunks.append(
                        act(jnp.ones(input.shape[:-1] + (mul, 1), input.dtype))
                    )
            else:
                chunks.append(act(x))
        else:
            irreps_out.append((mul, (l_in, p_in)))
            chunks.append(x)

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
        return e3nn.IrrepsArray(
            irreps_out, array, zero_flags=[x is None for x in chunks]
        )

    return e3nn.from_chunks(irreps_out, chunks, input.shape[:-1], input.dtype)


def norm_activation(
    input: e3nn.IrrepsArray,
    acts: List[Optional[Callable[[float], float]]],
    *,
    normalization: str = "component",
) -> e3nn.IrrepsArray:
    r"""Apply activation functions to the norms of the vectors of an `IrrepsArray`.

    Args:
        input (IrrepsArray): input array
        acts (list of functions): list of activation functions, one for each chunk of the input
        normalization (str): "component" or "norm"
            if "component" the norm is divided by the square root of the number of components.

    Returns:
        IrrepsArray: output array

    Examples:
        >>> x = e3nn.IrrepsArray("0e + 1o", jnp.array([1.0, 1.0, 1.0, 2.0]))
        >>> norm_activation(x, [None, jnp.tanh])
        1x0e+1x1o [1.        0.8883856 0.8883856 1.7767712]
    """
    assert isinstance(input, e3nn.IrrepsArray)
    assert normalization in ["component", "norm"]

    assert len(input.irreps) == len(acts), (input.irreps, acts)

    list = []

    for x, act in zip(input.chunks, acts):
        if act is None:
            list.append(x)
            continue

        if x is not None:
            n2 = jnp.sum(x**2, axis=-1, keepdims=True)
            if normalization == "component":
                n2 = n2 / x.shape[-1]
            n = jnp.where(n2 > 0.0, jnp.sqrt(jnp.where(n2 > 0.0, n2, 1.0)), 1.0)
            x = x * act(n)

        list.append(x)

    return e3nn.from_chunks(input.irreps, list, input.shape[:-1], input.dtype)


def key_value_activation(phi, key, value):
    assert key.ndim == 1
    assert value.ndim == 1

    d = value.shape[0]
    key = key / jnp.sqrt(
        1 / 16 + jnp.sum(key**2)
    )  # 1/16 is arbitrary small... but not too small...
    scalar = jnp.sum(key * value)
    scalar = normalize_function(phi)(scalar)
    return d**0.5 * scalar * key  # component normalized
