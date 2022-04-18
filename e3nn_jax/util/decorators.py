from functools import wraps

from e3nn_jax import Irreps, IrrepsData


def overload_for_irreps_without_data(irrepsdata_argnums=None, irrepsdata_argnames=None, shape=()):
    def decorator(func):
        from jax._src.api import _infer_argnums_and_argnames

        argnums, argnames = _infer_argnums_and_argnames(func, irrepsdata_argnums, irrepsdata_argnames)

        @wraps(func)
        def wrapper(*args, **kwargs):
            concerned_args = [args[i] for i in argnums if i < len(args)] + [kwargs[k] for k in argnames if k in kwargs]

            if any(isinstance(arg, (Irreps, str)) for arg in concerned_args):
                # assume arguments are Irreps (not IrrepsData)

                args = [IrrepsData.ones(a, shape) if i in argnums else a for i, a in enumerate(args)]
                kwargs = {k: IrrepsData.ones(v, shape) if k in argnames else v for k, v in kwargs.items()}
                output = func(*args, **kwargs)
                if isinstance(output, IrrepsData):
                    return output.irreps
                if isinstance(output, tuple):
                    return tuple(o.irreps if isinstance(o, IrrepsData) else o for o in output)
                raise TypeError(f"{func.__name__} returned {type(output)} which is not supported by `overload_irrep_no_data`.")

            # otherwise, assume arguments are IrrepsData
            return func(*args, **kwargs)

        return wrapper

    return decorator
