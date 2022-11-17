from typing import Union

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
    if indices is None and n_elements is None:
        raise ValueError("Either indices or n_elements must be specified")
    if indices is not None and n_elements is not None:
        raise ValueError("Only one of indices or n_elements must be specified")

    if indices is None:
        out_dim = n_elements.shape[0]
        num_elements = input.shape[0]
        indices = jnp.repeat(jnp.arange(out_dim), n_elements, total_repeat_length=num_elements)

    assert indices.shape[0] == input.shape[0]

    if out_dim is None and map_back is False:
        # out_dim = jnp.max(indices) + 1
        raise ValueError("out_dim must be specified if map_back is False")
    if out_dim is not None and map_back is True:
        raise ValueError("out_dim must not be specified if map_back is True")

    if out_dim is None and map_back is True:
        out_dim = indices.shape[0]
        indices = _distinct_but_small(indices)

    output = jax.tree_util.tree_map(lambda x: jnp.zeros((out_dim,) + x.shape[1:], x.dtype).at[(indices,)].add(x), input)

    if map_back:
        output = output[(indices,)]

    return output


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
