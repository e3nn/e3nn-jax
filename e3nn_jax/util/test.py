import inspect
import itertools
import logging
import warnings

import jax
import jax.numpy as jnp
from e3nn_jax import Irreps, rand_matrix


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


def equivariance_error(
    func,
    rng_key,
    args_in,
    irreps_in=None,
    irreps_out=None,
    ntrials=1,
    do_parity=True,
    do_translation=True
):
    r"""Get the maximum equivariance error for ``func`` over ``ntrials``
    Each trial randomizes the equivariant transformation tested.
    Parameters
    ----------
    func : callable
        the function to test
    args_in : list
        the original inputs to pass to ``func``.
    rng_key: jax.random.PRNGKey
        a random key for generation of random values
    irreps_in : list of `e3nn.o3.Irreps` or `e3nn.o3.Irreps`
        the input irreps for each of the arguments in ``args_in``. If left as the default of ``None``, ``get_io_irreps`` will be used to try to infer them. If a sequence is provided, valid elements are also the string ``'cartesian'``, which denotes that the corresponding input should be dealt with as cartesian points in 3D, and ``None``, which indicates that the argument should not be transformed.
    irreps_out : list of `e3nn.o3.Irreps` or `e3nn.o3.Irreps`
        the out irreps for each of the return values of ``func``. Accepts similar values to ``irreps_in``.
    ntrials : int
        run this many trials with random transforms
    do_parity : bool
        whether to test parity
    do_translation : bool
        whether to test translation for ``'cartesian'`` inputs
    Returns
    -------
    dictionary mapping tuples ``(parity_k, did_translate)`` to errors
    """
    irreps_in, irreps_out = _get_io_irreps(func, irreps_in=irreps_in, irreps_out=irreps_out)

    if do_parity:
        parity_ks = [0, 1]
    else:
        parity_ks = [0]

    if 'cartesian_points' not in irreps_in:
        # There's nothing to translate
        do_translation = False
    if do_translation:
        do_translation = [False, True]
    else:
        do_translation = [False]

    tests = itertools.product(parity_ks, do_translation)

    neg_inf = -float("Inf")
    biggest_errs = {}

    for trial in range(ntrials):
        for this_test in tests:
            parity_k, this_do_translate = this_test
            # Build a rotation matrix for point data
            rng_key, *sub_keys = jax.random.split(rng_key, num=3)
            rot_mat = rand_matrix(shape=(1,), key=sub_keys[0])[0]
            # add parity
            rot_mat *= (-1)**parity_k
            # build translation
            translation = 10 * jax.random.normal(sub_keys[1], shape=(1, 3), dtype=rot_mat.dtype) if this_do_translate else 0.

            # Evaluate the function on rotated arguments:
            rot_args = _transform(args_in, irreps_in, rot_mat, translation)
            x1 = func(*rot_args)

            # Evaluate the function on the arguments, then apply group action:
            x2 = func(*args_in)

            # Deal with output shapes
            assert type(x1) == type(x2), f"Inconsistant return types {type(x1)} and {type(x2)}"  # pylint: disable=unidiomatic-typecheck
            if isinstance(x1, jnp.DeviceArray):
                # Make sequences
                x1 = [x1]
                x2 = [x2]
            elif isinstance(x1, (list, tuple)):
                # They're already tuples
                x1 = list(x1)
                x2 = list(x2)
            else:
                raise TypeError(f"equivariance_error cannot handle output type {type(x1)}")
            assert len(x1) == len(x2), f"Length of x1: {len(x1)}, length of x2: {len(x2)}"
            assert len(x1) == len(irreps_out), f"Length of x1: {len(x1)}, length of irreps_out: {len(irreps_out)}"

            # apply the group action to x2
            x2 = _transform(x2, irreps_out, rot_mat, translation)

            error = max(
                jnp.max(jnp.abs(a - b))
                for a, b in zip(x1, x2)
            )

            if error > biggest_errs.get(this_test, neg_inf):
                biggest_errs[this_test] = error

    return biggest_errs


def _logging_name(func) -> str:
    """Get a decent string representation of ``func`` for logging"""
    if inspect.isfunction(func):
        return func.__name__
    else:
        return repr(func)


def format_equivariance_error(errors: dict) -> str:
    """Format the dictionary returned by ``equivariance_error`` into a readable string.
    Parameters
    ----------
        errors : dict
            A dictionary of errors returned by ``equivariance_error``.
    Returns
    -------
        A string.
    """
    return "; ".join(
        "(parity_k={:d}, did_translate={}) -> error={:.3e}".format(
            int(k[0]),
            bool(k[1]),
            float(v)
        )
        for k, v in errors.items()
    )


def assert_equivariant(
    func,
    rng_key,
    args_in=None,
    irreps_in=None,
    irreps_out=None,
    tolerance=None,
    **kwargs
) -> dict:
    r"""Assert that ``func`` is equivariant.
    Parameters
    ----------
        args_in : list or None
            the original input arguments for the function. If ``None`` and the function has ``irreps_in`` consisting only of ``o3.Irreps`` and ``'cartesian'``, random test inputs will be generated.
        irreps_in : object
            see ``equivariance_error``
        irreps_out : object
            see ``equivariance_error``
        tolerance : float or None
            the threshold below which the equivariance error must fall.
        **kwargs : kwargs
            passed through to ``equivariance_error``.
    Returns
    -------
    The same as ``equivariance_error``: a dictionary mapping tuples ``(parity_k, did_translate)`` to errors
    """
    # Prevent pytest from showing this function in the traceback
    __tracebackhide__ = True

    rng_key, sub_key = jax.random.split(rng_key)

    args_in, irreps_in, irreps_out = _get_args_in(
        func,
        rng_key=rng_key,
        args_in=args_in,
        irreps_in=irreps_in,
        irreps_out=irreps_out
    )

    # Get error
    errors = equivariance_error(
        func,
        sub_key,
        args_in=args_in,
        irreps_in=irreps_in,
        irreps_out=irreps_out,
        **kwargs
    )

    logging.info(
        "Tested equivariance of `%s` -- max componentwise errors: %s",
        _logging_name(func),
        format_equivariance_error(errors),
    )

    # Check it
    if tolerance is None:
        tolerance = 1e-3

    problems = {case: err for case, err in errors.items() if err > tolerance}

    if len(problems) != 0:
        errstr = "Largest componentwise equivariance error was too large for: "
        errstr += format_equivariance_error(problems)
        assert len(problems) == 0, errstr

    return errors
