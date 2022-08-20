import inspect
from functools import wraps

import jax
from e3nn_jax import Irreps, IrrepsArray


def overload_for_irreps_without_array(irrepsarray_argnums=None, irrepsarray_argnames=None, shape=()):
    def decorator(func):
        # TODO: this is very bad to use a function from the internal API
        from jax._src.api import _infer_argnums_and_argnames

        argnums, argnames = _infer_argnums_and_argnames(inspect.signature(func), irrepsarray_argnums, irrepsarray_argnames)

        @wraps(func)
        def wrapper(*args, **kwargs):
            concerned_args = [args[i] for i in argnums if i < len(args)] + [kwargs[k] for k in argnames if k in kwargs]

            if any(isinstance(arg, (Irreps, str)) for arg in concerned_args):
                # assume arguments are Irreps (not IrrepsArray)

                converted_args = {i: IrrepsArray.ones(a, shape) for i, a in enumerate(args) if i in argnums}
                converted_args.update({k: IrrepsArray.ones(v, shape) for k, v in kwargs.items() if k in argnames})

                def fn(converted_args):
                    args_ = [converted_args.get(i, a) for i, a in enumerate(args)]
                    kwargs_ = {k: converted_args.get(k, v) for k, v in kwargs.items()}
                    return func(*args_, **kwargs_)

                output = jax.eval_shape(fn, converted_args)

                if isinstance(output, IrrepsArray):
                    return output.irreps
                if isinstance(output, tuple):
                    return tuple(o.irreps if isinstance(o, IrrepsArray) else o for o in output)
                raise TypeError(f"{func.__name__} returned {type(output)} which is not supported by `overload_irrep_no_data`.")

            # otherwise, assume arguments are IrrepsArray
            return func(*args, **kwargs)

        return wrapper

    return decorator
