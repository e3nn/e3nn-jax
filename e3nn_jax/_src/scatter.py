import warnings
from typing import Optional, Union

import jax
import jax.numpy as jnp

import e3nn_jax as e3nn


def _distinct_but_small(x: jax.Array) -> jax.Array:
    """Maps the input to the integers 0, 1, 2, ..., n-1, where n is the number of distinct elements in x.

    Args:
        x (`jax.Array`): array of integers

    Returns:
        `jax.Array`: array of integers of same size
    """
    shape = x.shape
    x = jnp.ravel(x)
    unique = jnp.unique(x, size=x.shape[0])  # Pigeonhole principle
    x = jax.lax.scan(
        lambda _, i: (None, jnp.where(i == unique, size=1)[0][0]), None, x
    )[1]
    return jnp.reshape(x, shape)


def scatter_sum(
    data: Union[jax.Array, e3nn.IrrepsArray],
    *,
    dst: Optional[jax.Array] = None,
    nel: Optional[jax.Array] = None,
    output_size: Optional[int] = None,
    map_back: bool = False,
    mode: str = "promise_in_bounds",
) -> Union[jax.Array, e3nn.IrrepsArray]:
    r"""Scatter sum of data.

    Performs either of the following two operations::
        output[dst[i]] += data[i]

    or::

        output[i] = sum(data[sum(nel[:i]):sum(nel[:i+1])])

    Args:
        data (`jax.Array` or `IrrepsArray`): array of shape ``(n1,..nd, ...)``
        dst (optional, `jax.Array`): array of shape ``(n1,..nd)``. If not specified, ``nel`` must be specified.
        nel (optional, `jax.Array`): array of shape ``(output_size,)``. If not specified, ``dst`` must be specified.
        output_size (optional, int): size of output array.
            If not specified, ``nel`` must be specified or ``map_back`` must be ``True``.
        map_back (bool): whether to map back to the input position

    Returns:
        `jax.Array` or `IrrepsArray`: output array of shape ``(output_size, ...)``
    """
    return _scatter_op(
        "sum",
        0.0,
        data,
        dst=dst,
        nel=nel,
        output_size=output_size,
        map_back=map_back,
        mode=mode,
    )


def scatter_mean(
    data: Union[jax.Array, e3nn.IrrepsArray],
    *,
    dst: Optional[jax.Array] = None,
    nel: Optional[jax.Array] = None,
    output_size: Optional[int] = None,
    map_back: bool = False,
    mode: str = "promise_in_bounds",
) -> Union[jax.Array, e3nn.IrrepsArray]:
    r"""Scatter mean of data.

    Performs either of the following two operations::

        n[dst[i]] += 1
        output[dst[i]] += data[i] / n[i]

    or::

        output[i] = sum(data[sum(nel[:i]):sum(nel[:i+1])]) / nel[i]

    Args:
        data (`jax.Array` or `IrrepsArray`): array of shape ``(n1,..nd, ...)``
        dst (optional, `jax.Array`): array of shape ``(n1,..nd)``. If not specified, ``nel`` must be specified.
        nel (optional, `jax.Array`): array of shape ``(output_size,)``. If not specified, ``dst`` must be specified.
        output_size (optional, int): size of output array.
            If not specified, ``nel`` must be specified or ``map_back`` must be ``True``.
        map_back (bool): whether to map back to the input position

    Returns:
        `jax.Array` or `IrrepsArray`: output array of shape ``(output_size, ...)``
    """
    if map_back and nel is not None:
        assert dst is None
        assert output_size is None

        total = _scatter_op(
            "sum",
            0.0,
            data,
            nel=nel,
            map_back=False,
            mode=mode,
        )
        den = jnp.maximum(1, nel)

        for _ in range(total.ndim - nel.ndim):
            den = den[..., None]

        output = total / den.astype(total.dtype)
        output = jax.tree_map(
            lambda x: jnp.repeat(x, nel, axis=0, total_repeat_length=data.shape[0]),
            output,
        )
        return output

    total = _scatter_op(
        "sum",
        0.0,
        data,
        dst=dst,
        nel=nel,
        output_size=output_size,
        map_back=map_back,
        mode=mode,
    )

    if dst is not None or map_back:
        if dst is not None:
            ones = jnp.ones(data.shape[: dst.ndim], jnp.int32)
        if nel is not None:
            ones = jnp.ones(data.shape[:1], jnp.int32)

        nel = _scatter_op(
            "sum",
            0.0,
            ones,
            dst=dst,
            nel=nel,
            output_size=output_size,
            map_back=map_back,
            mode=mode,
        )

    nel = jnp.maximum(1, nel)

    for _ in range(total.ndim - nel.ndim):
        nel = nel[..., None]

    return total / nel.astype(total.dtype)


def scatter_max(
    data: Union[jax.Array, e3nn.IrrepsArray],
    *,
    dst: Optional[jax.Array] = None,
    nel: Optional[jax.Array] = None,
    initial: float = -jnp.inf,
    output_size: Optional[int] = None,
    map_back: bool = False,
    mode: str = "promise_in_bounds",
) -> Union[jax.Array, e3nn.IrrepsArray]:
    r"""Scatter max of data.

    Performs either of the following two operations::

        output[i] = max(initial, *(x for j, x in zip(dst, data) if j == i))

    or::

        output[i] = max(initial, *data[sum(nel[:i]):sum(nel[:i+1])])

    Args:
        data (`jax.Array` or `IrrepsArray`): array of shape ``(n, ...)``
        dst (optional, `jax.Array`): array of shape ``(n,)``. If not specified, ``nel`` must be specified.
        nel (optional, `jax.Array`): array of shape ``(output_size,)``. If not specified, ``dst`` must be specified.
        initial (float): initial value to compare to
        output_size (optional, int): size of output array. If not specified, ``nel`` must be specified
            or ``map_back`` must be ``True``.
        map_back (bool): whether to map back to the input position

    Returns:
        `jax.Array` or `IrrepsArray`: output array of shape ``(output_size, ...)``
    """
    if isinstance(data, e3nn.IrrepsArray):
        if not data.irreps.is_scalar():
            raise ValueError("scatter_max only works with scalar IrrepsArray")

    return _scatter_op(
        "max",
        initial,
        data,
        dst=dst,
        nel=nel,
        output_size=output_size,
        map_back=map_back,
        mode=mode,
    )


def _scatter_op(
    op: str,
    initial: float,
    data: Union[jax.Array, e3nn.IrrepsArray],
    *,
    dst: Optional[jax.Array] = None,
    nel: Optional[jax.Array] = None,
    output_size: Optional[int] = None,
    map_back: bool = False,
    mode: str = "promise_in_bounds",
) -> Union[jax.Array, e3nn.IrrepsArray]:
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

    if not (dst.shape == data.shape[: dst.ndim]):
        raise ValueError(
            (
                f"trying to do e3nn.scatter_{op} with dst.shape={dst.shape} and data.shape={data.shape}"
                f" but dst.shape must be equal to data.shape[: dst.ndim]"
            )
        )

    if output_size is None and map_back is False:
        raise ValueError("output_size must be specified if map_back is False")
    if output_size is not None and map_back is True:
        raise ValueError("output_size must not be specified if map_back is True")

    if output_size is None and map_back is True:
        output_size = dst.size
        dst = _distinct_but_small(dst)

    def _op(x):
        z = initial * jnp.ones((output_size,) + x.shape[dst.ndim :], x.dtype)
        if op == "sum":
            return z.at[(dst,)].add(x, indices_are_sorted=indices_are_sorted, mode=mode)
        elif op == "max":
            return z.at[(dst,)].max(x, indices_are_sorted=indices_are_sorted, mode=mode)

    output = jax.tree_util.tree_map(_op, data)

    if map_back:
        output = output[(dst,)]

    return output


def index_add(
    indices: jax.Array = None,
    input: Union[jax.Array, e3nn.IrrepsArray] = None,
    *,
    n_elements: jax.Array = None,
    out_dim: int = None,
    map_back: bool = False,
) -> Union[jax.Array, e3nn.IrrepsArray]:
    warnings.warn(
        "e3nn.index_add is deprecated, use e3nn.scatter_sum instead", DeprecationWarning
    )
    return scatter_sum(
        input, dst=indices, nel=n_elements, output_size=out_dim, map_back=map_back
    )
