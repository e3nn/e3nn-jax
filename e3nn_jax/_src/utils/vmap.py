from typing import Any, Callable, Sequence, Union

import jax
from attr import attrib, attrs

import e3nn_jax as e3nn


def vmap(
    fun: Callable[..., Any],
    in_axes: Union[int, None, Sequence[Any]] = 0,
    out_axes: Any = 0,
):
    r"""Wrapper around :func:`jax.vmap` that handles :class:`e3nn_jax.IrrepsArray` objects.

    Args:
        fun: Function to be mapped.
        in_axes: Specifies which axes to map over for the input arguments. See :func:`jax.vmap` for details.
        out_axes: Specifies which axes to map over for the output arguments. See :func:`jax.vmap` for details.

    Returns:
        Batched/vectorized version of ``fun``.

    Example:
        >>> import jax.numpy as jnp
        >>> x = e3nn.from_chunks("0e + 0e", [jnp.ones((100, 1, 1)), None], (100,))
        >>> x.zero_flags
        (False, True)
        >>> y = vmap(e3nn.scalar_activation)(x)
        >>> y.zero_flags
        (False, True)
    """

    def to_via(x):
        return _VIA(x) if isinstance(x, e3nn.IrrepsArray) else x

    def from_via(x):
        return x.a if isinstance(x, _VIA) else x

    def inside_fun(*args, **kwargs):
        args, kwargs = jax.tree_util.tree_map(
            from_via, (args, kwargs), is_leaf=lambda x: isinstance(x, _VIA)
        )
        out = fun(*args, **kwargs)
        return jax.tree_util.tree_map(
            to_via, out, is_leaf=lambda x: isinstance(x, e3nn.IrrepsArray)
        )

    def outside_fun(*args, **kwargs):
        args, kwargs = jax.tree_util.tree_map(
            to_via, (args, kwargs), is_leaf=lambda x: isinstance(x, e3nn.IrrepsArray)
        )
        out = jax.vmap(inside_fun, in_axes, out_axes)(*args, **kwargs)
        return jax.tree_util.tree_map(
            from_via, out, is_leaf=lambda x: isinstance(x, _VIA)
        )

    return outside_fun


@attrs(frozen=True)
class _VIA:
    a: e3nn.IrrepsArray = attrib()


jax.tree_util.register_pytree_node(
    _VIA,
    lambda x: ((x.a.array,), (x.a.irreps, x.a.zero_flags)),
    lambda attrs, data: _VIA(e3nn.IrrepsArray(attrs[0], data[0], zero_flags=attrs[1])),
)
