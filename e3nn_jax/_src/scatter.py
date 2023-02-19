import warnings
from typing import Optional, Union

import jax
import jax.numpy as jnp

import e3nn_jax as e3nn


def _distinct_but_small(x: jnp.ndarray) -> jnp.ndarray:
    """Maps the input to the integers 0, 1, 2, ..., n-1, where n is the number of distinct elements in x.

    Args:
        x (`jax.numpy.ndarray`): array of integers

    Returns:
        `jax.numpy.ndarray`: array of integers of same size
    """
    assert x.ndim == 1
    unique = jnp.unique(x, size=x.shape[0])  # Pigeonhole principle
    return jax.lax.scan(lambda _, i: (None, jnp.where(i == unique, size=1)[0][0]), None, x)[1]


def scatter_sum(
    data: Union[jnp.ndarray, e3nn.IrrepsArray],
    *,
    dst: Optional[jnp.ndarray] = None,
    nel: Optional[jnp.ndarray] = None,
    output_size: Optional[int] = None,
    map_back: bool = False,
) -> Union[jnp.ndarray, e3nn.IrrepsArray]:
    r"""Scatter sum of data.

    Performs either of the following two operations:
    ``output[dst[i]] += data[i]`` or ``output[i] = sum(data[sum(nel[:i]):sum(nel[:i+1])])``

    Args:
        data (`jax.numpy.ndarray` or `IrrepsArray`): array of shape ``(n, ...)``
        dst (optional, `jax.numpy.ndarray`): array of shape ``(n,)``. If not specified, ``nel`` must be specified.
        nel (optional, `jax.numpy.ndarray`): array of shape ``(output_size,)``. If not specified, ``dst`` must be specified.
        output_size (optional, int): size of output array. If not specified, ``nel`` must be specified
            or ``map_back`` must be ``True``.
        map_back (bool): whether to map back to the input position

    Returns:
        `jax.numpy.ndarray` or `IrrepsArray`: output array of shape ``(output_size, ...)``
    """
    return _scatter_op("sum", 0.0, data, dst=dst, nel=nel, output_size=output_size, map_back=map_back)


def scatter_max(
    data: Union[jnp.ndarray, e3nn.IrrepsArray],
    *,
    dst: Optional[jnp.ndarray] = None,
    nel: Optional[jnp.ndarray] = None,
    initial: float = -jnp.inf,
    output_size: Optional[int] = None,
    map_back: bool = False,
) -> Union[jnp.ndarray, e3nn.IrrepsArray]:
    r"""Scatter max of data.

    Performs either of the following two operations::

        output[i] = max(initial, *(x for j, x in zip(dst, data) if j == i))

    or::

        output[i] = max(initial, *data[sum(nel[:i]):sum(nel[:i+1])])

    Args:
        data (`jax.numpy.ndarray` or `IrrepsArray`): array of shape ``(n, ...)``
        dst (optional, `jax.numpy.ndarray`): array of shape ``(n,)``. If not specified, ``nel`` must be specified.
        nel (optional, `jax.numpy.ndarray`): array of shape ``(output_size,)``. If not specified, ``dst`` must be specified.
        initial (float): initial value to compare to
        output_size (optional, int): size of output array. If not specified, ``nel`` must be specified
            or ``map_back`` must be ``True``.
        map_back (bool): whether to map back to the input position

    Returns:
        `jax.numpy.ndarray` or `IrrepsArray`: output array of shape ``(output_size, ...)``
    """
    if isinstance(data, e3nn.IrrepsArray):
        if not data.irreps.is_scalar():
            raise ValueError("scatter_max only works with scalar IrrepsArray")

    return _scatter_op("max", initial, data, dst=dst, nel=nel, output_size=output_size, map_back=map_back)


def _scatter_op(
    op: str,
    initial: float,
    data: Union[jnp.ndarray, e3nn.IrrepsArray],
    *,
    dst: Optional[jnp.ndarray] = None,
    nel: Optional[jnp.ndarray] = None,
    output_size: Optional[int] = None,
    map_back: bool = False,
) -> Union[jnp.ndarray, e3nn.IrrepsArray]:
    if dst is None and nel is None:
        raise ValueError("Either dst or nel must be specified")
    if dst is not None and nel is not None:
        raise ValueError("Only one of dst or nel must be specified")

    if nel is not None:
        if output_size is not None:
            raise ValueError("output_size must not be specified if nel is specified")
        output_size = nel.shape[0]
        num_elements = data.shape[0]
        dst = jnp.repeat(jnp.arange(output_size), nel, total_repeat_length=num_elements)
        indices_are_sorted = True
        if map_back:
            output_size = None
    else:
        indices_are_sorted = False

    assert dst.shape[0] == data.shape[0]

    if output_size is None and map_back is False:
        raise ValueError("output_size must be specified if map_back is False")
    if output_size is not None and map_back is True:
        raise ValueError("output_size must not be specified if map_back is True")

    if output_size is None and map_back is True:
        output_size = dst.shape[0]
        dst = _distinct_but_small(dst)

    def _op(x):
        z = initial * jnp.ones((output_size,) + x.shape[1:], x.dtype)
        if op == "sum":
            return z.at[(dst,)].add(x, indices_are_sorted=indices_are_sorted)
        elif op == "max":
            return z.at[(dst,)].max(x, indices_are_sorted=indices_are_sorted)

    output = jax.tree_util.tree_map(_op, data)

    if map_back:
        output = output[(dst,)]

    return output


def index_add(
    indices: jnp.ndarray = None,
    input: Union[jnp.ndarray, e3nn.IrrepsArray] = None,
    *,
    n_elements: jnp.ndarray = None,
    out_dim: int = None,
    map_back: bool = False,
) -> Union[jnp.ndarray, e3nn.IrrepsArray]:
    r"""Perform the operation.

    ```
    out = zeros(out_dim, ...)
    out[indices] += input
    ```

    if ``map_back`` is ``True``, then the output is mapped back to the input position.

    ```
    return out[indices]
    ```

    Args:
        indices (`jax.numpy.ndarray`): array of indices
        input (`jax.numpy.ndarray` or `IrrepsArray`): array of data
        out_dim (int): size of the output
        map_back (bool): whether to map back to the input position

    Returns:
        `jax.numpy.ndarray` or ``IrrepsArray``: output

    Examples:
       >>> i = jnp.array([0, 2, 2, 0])
       >>> x = jnp.array([1.0, 2.0, 3.0, -10.0])
       >>> index_add(i, x, out_dim=4)
       Array([-9.,  0.,  5.,  0.], dtype=float32)
    """
    warnings.warn("e3nn.index_add is deprecated, use e3nn.scatter_sum instead", DeprecationWarning)
    return scatter_sum(input, dst=indices, nel=n_elements, output_size=out_dim, map_back=map_back)
