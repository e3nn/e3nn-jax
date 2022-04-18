import jax
import jax.numpy as jnp


def index_add(i, x, out_dim):
    r"""perform the operation

    ```
    out = zeros(out_dim, ...)
    out[i] += x
    ```

    Args:
        i (`jnp.ndarray`): array of indices
        x (`jnp.ndarray`): array of data
        out_dim (int): size of the output

    Returns:
        `jnp.ndarray`: ``out``

    Example:
       >>> i = jnp.array([0, 2, 2, 0])
       >>> x = jnp.array([1.0, 2.0, 3.0, -10.0])
       >>> index_add(i, x, out_dim=4)
       DeviceArray([-9.,  0.,  5.,  0.], dtype=float32)
    """
    # out_dim = jnp.max(i) + 1
    return jnp.zeros((out_dim,) + x.shape[1:]).at[i].add(x)


def radius_graph(pos, r_max, *, batch=None, size=None, loop=False):
    r"""naive and inefficient version of ``torch_cluster.radius_graph``

    Args:
        pos (`jnp.ndarray`): array of shape ``(n, 3)``
        r_max (float):
        batch (`jnp.ndarray`): indices
        size (int): size of the output
        loop (bool): whether to include self-loops

    Returns:
        `jnp.ndarray`: src
        `jnp.ndarray`: dst

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> pos = jax.random.normal(key, (20, 3))
        >>> batch = jnp.arange(20) < 10
        >>> radius_graph(pos, 0.8, batch=batch)
        (DeviceArray([ 3,  7, 10, 11, 12, 18], dtype=int32), DeviceArray([ 7,  3, 11, 10, 18, 12], dtype=int32))
    """
    r = jax.vmap(jax.vmap(lambda x, y: jnp.linalg.norm(x - y), (None, 0), 0), (0, None), 0)(pos, pos)
    if loop:
        mask = r < r_max
    else:
        mask = (r < r_max) & (r > 0)

    src, dst = jnp.where(mask, size=size, fill_value=-1)

    if batch is None:
        return src, dst
    return src[batch[src] == batch[dst]], dst[batch[src] == batch[dst]]
