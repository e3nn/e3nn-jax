import warnings
from typing import Optional, Union

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp


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
        data (`jax.numpy.ndarray` or `e3nn.IrrepsArray`): array of shape ``(n, ...)``
        dst (optional, `jax.numpy.ndarray`): array of shape ``(n,)``. If not specified, ``nel`` must be specified.
        nel (optional, `jax.numpy.ndarray`): array of shape ``(output_size,)``. If not specified, ``dst`` must be specified.
        output_size (optional, int): size of output array. If not specified, ``nel`` must be specified
            or ``map_back`` must be ``True``.
        map_back (bool): whether to map back to the input position

    Returns:
        `jax.numpy.ndarray` or `e3nn.IrrepsArray`: output array of shape ``(output_size, ...)``
    """
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

    output = jax.tree_util.tree_map(
        lambda x: jnp.zeros((output_size,) + x.shape[1:], x.dtype).at[(dst,)].add(x, indices_are_sorted=indices_are_sorted),
        data,
    )

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

    Example:
       >>> i = jnp.array([0, 2, 2, 0])
       >>> x = jnp.array([1.0, 2.0, 3.0, -10.0])
       >>> index_add(i, x, out_dim=4)
       DeviceArray([-9.,  0.,  5.,  0.], dtype=float32)
    """
    warnings.warn("index_add is deprecated, use scatter_sum instead", DeprecationWarning)
    return scatter_sum(input, dst=indices, nel=n_elements, output_size=out_dim, map_back=map_back)


def radius_graph(
    pos: Union[e3nn.IrrepsArray, jnp.ndarray],
    r_max: float,
    *,
    batch: jnp.ndarray = None,
    size: int = None,
    loop: bool = False,
    fill_src: int = -1,
    fill_dst: int = -1,
):
    r"""Naive and inefficient version of ``torch_cluster.radius_graph``.

    Args:
        pos (`jax.numpy.ndarray`): array of shape ``(n, 3)``
        r_max (float):
        batch (`jax.numpy.ndarray`): indices
        size (int): size of the output
        loop (bool): whether to include self-loops

    Returns:
        - jax.numpy.ndarray: source indices
        - jax.numpy.ndarray: destination indices

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> pos = jax.random.normal(key, (20, 3))
        >>> batch = jnp.arange(20) < 10
        >>> radius_graph(pos, 0.8, batch=batch)
        (DeviceArray([ 3,  7, 10, 11, 12, 18], dtype=int32), DeviceArray([ 7,  3, 11, 10, 18, 12], dtype=int32))
    """
    if isinstance(pos, e3nn.IrrepsArray):
        pos = pos.array

    r = jax.vmap(jax.vmap(lambda x, y: jnp.linalg.norm(x - y), (None, 0), 0), (0, None), 0)(pos, pos)
    if loop:
        mask = r < r_max
    else:
        mask = (r < r_max) & (r > 0)

    src, dst = jnp.where(mask, size=size, fill_value=-1)

    if fill_src != -1:
        src = jnp.where(src == -1, fill_src, src)
    if fill_dst != -1:
        dst = jnp.where(dst == -1, fill_dst, dst)

    if batch is None:
        return src, dst
    return src[batch[src] == batch[dst]], dst[batch[src] == batch[dst]]
