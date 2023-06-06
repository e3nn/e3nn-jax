from typing import Union

import jax
import jax.numpy as jnp

import e3nn_jax as e3nn


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
    r"""Try to use ``matscipy.neighbours.neighbour_list`` instead.

    Args:
        pos (`jax.numpy.ndarray`): array of shape ``(n, 3)``
        r_max (float):
        batch (`jax.numpy.ndarray`): indices
        size (int): size of the output
        loop (bool): whether to include self-loops

    Returns:
        (tuple): tuple containing:

            jax.numpy.ndarray: source indices
            jax.numpy.ndarray: destination indices

    Examples:
        >>> key = jax.random.PRNGKey(0)
        >>> pos = jax.random.normal(key, (20, 3))
        >>> batch = jnp.arange(20) < 10
        >>> radius_graph(pos, 0.8, batch=batch)
        (Array([ 3,  7, 10, 11, 12, 18], dtype=int32), Array([ 7,  3, 11, 10, 18, 12], dtype=int32))
    """
    # TODO(mario): replace with the function made for Allan once the project is finished
    if isinstance(pos, e3nn.IrrepsArray):
        pos = pos.array

    r = jax.vmap(
        jax.vmap(lambda x, y: jnp.linalg.norm(x - y), (None, 0), 0), (0, None), 0
    )(pos, pos)
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
