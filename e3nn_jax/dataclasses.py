"""Utilities for defining dataclasses that can be used with jax transformations.

This code was copied and adapted from https://github.com/google/jax-md/blob/main/jax_md/dataclasses.py.

Accessed on 02/28/2022.
"""

import dataclasses
import jax


def dataclass(clz):
    """Create a class which can be passed to functional transformations.

    Jax transformations such as `jax.jit` and `jax.grad` require objects that are
    immutable and can be mapped over using the `jax.tree_util` methods.

    The `dataclass` decorator makes it easy to define custom classes that can be
    passed safely to Jax.

    Args:
        clz: the class that will be transformed by the decorator.
    Returns:
        The new class.
    """
    data_clz = dataclasses.dataclass(frozen=True)(clz)
    meta_fields = []
    data_fields = []
    for name, field_info in data_clz.__dataclass_fields__.items():
        is_static = field_info.metadata.get('static', False)
        if is_static:
            meta_fields.append(name)
        else:
            data_fields.append(name)

    def iterate_clz(x):
        meta = tuple(getattr(x, name) for name in meta_fields)
        data = tuple(getattr(x, name) for name in data_fields)
        return data, meta

    def clz_from_iterable(meta, data):
        meta_args = tuple(zip(meta_fields, meta))
        data_args = tuple(zip(data_fields, data))
        kwargs = dict(meta_args + data_args)
        return data_clz(**kwargs)

    jax.tree_util.register_pytree_node(data_clz, iterate_clz, clz_from_iterable)

    return data_clz


def static_field():
    return dataclasses.field(metadata={'static': True})


replace = dataclasses.replace
asdict = dataclasses.asdict
astuple = dataclasses.astuple


def unpack(dc) -> tuple:
    return tuple(getattr(dc, field.name) for field in dataclasses.fields(dc))
