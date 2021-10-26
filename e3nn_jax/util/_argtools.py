import jax.numpy as jnp
import jax
import warnings
from e3nn_jax import Irreps


def _transform(dat, irreps_dat, rot_mat, translation=0.0):
    """Transform ``dat`` by ``rot_mat`` and ``translation`` according to ``irreps_dat``."""
    out = []
    for irreps, a in zip(irreps_dat, dat):
        if irreps is None:
            out.append(a)
        elif irreps == "cartesian_points":
            translation = jnp.array(translation)
            out.append((a @ rot_mat.T.to(a.device)) + translation)
        else:
            out.append(a @ irreps.D_from_matrix(rot_mat).T)
    return out


def _get_io_irreps(func, irreps_in=None, irreps_out=None):
    """Preprocess or, if not given, try to infer the I/O irreps for ``func``."""
    SPECIAL_VALS = ["cartesian_points", None]

    if irreps_in is None:
        if hasattr(func, "irreps_in"):
            irreps_in = func.irreps_in  # gets checked for type later
        elif hasattr(func, "irreps_in1"):
            irreps_in = [func.irreps_in1, func.irreps_in2]
        else:
            raise ValueError(
                "Cannot infer irreps_in for %r; provide them explicitly" % func
            )
    if irreps_out is None:
        if hasattr(func, "irreps_out"):
            irreps_out = func.irreps_out  # gets checked for type later
        else:
            raise ValueError(
                "Cannot infer irreps_out for %r; provide them explicitly" % func
            )

    if isinstance(irreps_in, Irreps) or irreps_in in SPECIAL_VALS:
        irreps_in = [irreps_in]
    elif isinstance(irreps_in, list):
        irreps_in = [i if i in SPECIAL_VALS else Irreps(i) for i in irreps_in]
    else:
        if isinstance(irreps_in, tuple) and not isinstance(irreps_in, Irreps):
            warnings.warn(
                f"Module {func} had irreps_in of type tuple but not Irreps; ambiguous whether the tuple should be interpreted as a tuple representing a single Irreps or a tuple of objects each to be converted to Irreps. Assuming the former. If the latter, use a list."
            )
        irreps_in = [Irreps(irreps_in)]

    if isinstance(irreps_out, Irreps) or irreps_out in SPECIAL_VALS:
        irreps_out = [irreps_out]
    elif isinstance(irreps_out, list):
        irreps_out = [i if i in SPECIAL_VALS else Irreps(i) for i in irreps_out]
    else:
        if isinstance(irreps_in, tuple) and not isinstance(irreps_in, Irreps):
            warnings.warn(
                f"Module {func} had irreps_out of type tuple but not Irreps; ambiguous whether the tuple should be interpreted as a tuple representing a single Irreps or a tuple of objects each to be converted to Irreps. Assuming the former. If the latter, use a list."
            )
        irreps_out = [Irreps(irreps_out)]

    return irreps_in, irreps_out


def _get_args_in(func, rng_key, args_in=None, irreps_in=None, irreps_out=None):
    irreps_in, irreps_out = _get_io_irreps(
        func, irreps_in=irreps_in, irreps_out=irreps_out
    )
    if args_in is None:
        rng_key, sub_key = jax.random.split(rng_key)
        args_in = _rand_args(irreps_in, sub_key)
    assert len(args_in) == len(irreps_in), "irreps_in and args_in don't match in length"
    return args_in, irreps_in, irreps_out


def _rand_args(irreps_in, rng_key, batch_size=None):
    rng_key, *sub_keys = jax.random.split(rng_key, num=1 + len(irreps_in))
    if not all((isinstance(i, Irreps) or i == "cartesian_points") for i in irreps_in):
        raise ValueError(
            "Random arguments cannot be generated when argument types besides Irreps and `'cartesian_points'` are specified; provide explicit ``args_in``"
        )
    if batch_size is None:
        # Generate random args with random size batch dim between 1 and 4:
        batch_size = 4
    args_in = [
        jax.random.normal(sub_keys[i], (batch_size, 3))
        if (irreps == "cartesian_points")
        else irreps.randn(sub_keys[i], (batch_size, -1))
        for i, irreps in enumerate(irreps_in)
    ]
    return args_in
