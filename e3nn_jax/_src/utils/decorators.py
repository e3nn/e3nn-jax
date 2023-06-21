import inspect
from functools import wraps

import jax
import e3nn_jax as e3nn


def overload_for_irreps_without_array(
    irrepsarray_argnums=None, irrepsarray_argnames=None, shape=()
):
    def decorator(func):
        # TODO: this is very bad to use a function from the internal API
        try:
            from jax._src.api import (
                _infer_argnums_and_argnames as infer_argnums_and_argnames,
            )
        except ImportError:
            from jax._src.api_util import infer_argnums_and_argnames

        argnums, argnames = infer_argnums_and_argnames(
            inspect.signature(func), irrepsarray_argnums, irrepsarray_argnames
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            concerned_args = [args[i] for i in argnums if i < len(args)] + [
                kwargs[k] for k in argnames if k in kwargs
            ]

            if any(isinstance(arg, (e3nn.Irreps, str)) for arg in concerned_args):
                # assume arguments are Irreps (not IrrepsArray)

                converted_args = {
                    i: e3nn.zeros(a, shape) for i, a in enumerate(args) if i in argnums
                }
                converted_args.update(
                    {
                        k: e3nn.zeros(v, shape)
                        for k, v in kwargs.items()
                        if k in argnames
                    }
                )

                def fn(converted_args):
                    args_ = [converted_args.get(i, a) for i, a in enumerate(args)]
                    kwargs_ = {k: converted_args.get(k, v) for k, v in kwargs.items()}
                    return func(*args_, **kwargs_)

                output = jax.eval_shape(fn, converted_args)

                if isinstance(output, e3nn.IrrepsArray):
                    return output.irreps
                if isinstance(output, tuple):
                    return tuple(
                        o.irreps if isinstance(o, e3nn.IrrepsArray) else o
                        for o in output
                    )
                raise TypeError(
                    f"{func.__name__} returned {type(output)} which is not supported by `overload_irrep_no_data`."
                )

            # otherwise, assume arguments are IrrepsArray
            return func(*args, **kwargs)

        return wrapper

    return decorator
