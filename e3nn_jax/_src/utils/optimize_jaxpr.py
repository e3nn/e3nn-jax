from functools import partial
from typing import Any, List

import jax
import numpy as np
from jax import linear_util as lu
from jax.core import Atom, ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Var, jaxpr_as_fun


def curry(f):
    return partial(partial, f)


@curry
@curry
def closed_jaxpr_transform_to_fn_transform(
    closed_jaxpr_transform, fn, *args
):  # pragma: no cover
    f = lu.wrap_init(fn)

    in_flat, in_tree = jax.tree_util.tree_flatten(args)
    f, out_tree = jax.flatten_fun_nokwargs(f, in_tree)
    closed_jaxpr = jax.make_jaxpr(f.call_wrapped)(*in_flat)
    closed_jaxpr, input_indices = closed_jaxpr_transform(closed_jaxpr)
    out_flat = jaxpr_as_fun(closed_jaxpr)(*[in_flat[i] for i in input_indices])

    return jax.tree_util.tree_unflatten(out_tree(), out_flat)


def replace_var(jaxpr: Jaxpr, old: Var, new: Var) -> List[JaxprEqn]:  # pragma: no cover
    def re(vars):
        return [new if var == old else var for var in vars]

    return jaxpr.replace(
        constvars=re(jaxpr.constvars),
        invars=re(jaxpr.invars),
        outvars=re(jaxpr.outvars),
        eqns=[
            eq.replace(
                invars=re(eq.invars),
                outvars=re(eq.outvars),
            )
            for eq in jaxpr.eqns
        ],
    )


def remove_deadcode(
    jaxpr: Jaxpr, output_indices=None
) -> ClosedJaxpr:  # pragma: no cover
    if output_indices is None:
        output_indices = range(len(jaxpr.outvars))

    needed = set([jaxpr.outvars[i] for i in output_indices])
    eqns = []

    for eqn in reversed(jaxpr.eqns):
        if len(needed.intersection(eqn.outvars)) == 0:
            continue

        from jax.interpreters.xla import xla_call_p  # TODO: fix import

        if eqn.primitive in [xla_call_p]:
            xla_call_jaxpr = eqn.params["call_jaxpr"]
            xla_call_jaxpr, input_indices = remove_deadcode(
                xla_call_jaxpr,
                [i for i, out in enumerate(eqn.outvars) if out in needed],
            )
            eqn = eqn.replace(
                params={
                    **eqn.params,
                    "call_jaxpr": xla_call_jaxpr,
                    "donated_invars": tuple(
                        eqn.params["donated_invars"][i] for i in input_indices
                    ),
                },
                outvars=[out for out in eqn.outvars if out in needed],
                invars=[inv for i, inv in enumerate(eqn.invars) if i in input_indices],
            )

        eqns.insert(0, eqn)

        for invar in eqn.invars:
            if type(invar) is Var:
                needed.add(invar)

    jaxpr = jaxpr.replace(eqns=eqns)
    jaxpr = jaxpr.replace(outvars=[jaxpr.outvars[i] for i in output_indices])
    input_indices = [i for i, invar in enumerate(jaxpr.invars) if invar in needed]
    jaxpr = jaxpr.replace(invars=[invar for invar in jaxpr.invars if invar in needed])

    return (jaxpr, input_indices)


def remove_duplicate_constants(
    closed_jaxpr: ClosedJaxpr,
) -> ClosedJaxpr:  # pragma: no cover
    for i, cst1 in enumerate(closed_jaxpr.consts):
        for j, cst2 in enumerate(closed_jaxpr.consts[:i]):
            if type(cst1) is np.ndarray and type(cst2) is np.ndarray:
                if np.array_equal(cst1, cst2) and cst1.dtype == cst2.dtype:
                    closed_jaxpr.consts.pop(j)
                    new = closed_jaxpr.jaxpr.constvars[i]
                    old = closed_jaxpr.jaxpr.constvars.pop(j)
                    closed_jaxpr.jaxpr = replace_var(closed_jaxpr.jaxpr, old, new)
                    return remove_duplicate_constants(closed_jaxpr)

    # apply reccursively
    closed_jaxpr.jaxpr = closed_jaxpr.jaxpr.replace(
        eqns=[
            eqn.replace(
                params={
                    k: remove_duplicate_constants(v) if type(v) is ClosedJaxpr else v
                    for k, v in eqn.params.items()
                }
            )
            for eqn in closed_jaxpr.jaxpr.eqns
        ]
    )

    return closed_jaxpr


def remove_duplicate_equations(
    jaxpr: Jaxpr, skip_first=0
) -> ClosedJaxpr:  # pragma: no cover
    def atom_key(a: Atom):
        if type(a) is Literal:
            return a.val
        return a

    def param_key(p: Any):
        if type(p) in [Jaxpr, ClosedJaxpr]:
            return str(p)
        if type(p) is np.ndarray:
            return (p.shape, p.dtype, p.tobytes())
        if type(p) is list:
            return [param_key(x) for x in p]
        if type(p) is dict:
            return {k: param_key(v) for k, v in p.items()}
        if type(p) is tuple:
            return tuple(param_key(x) for x in p)
        if callable(p):
            return p.__name__
        return str(p)

    for i, eq1 in enumerate(jaxpr.eqns):
        if i < skip_first:
            continue

        for j, eq2 in enumerate(jaxpr.eqns[:i]):
            if eq1.primitive == eq2.primitive:
                if list(map(atom_key, eq1.invars)) == list(map(atom_key, eq2.invars)):
                    if param_key(eq1.params) == param_key(eq2.params):
                        assert i > j
                        for old, new in zip(eq1.outvars, eq2.outvars):
                            jaxpr = replace_var(jaxpr, old, new)
                        jaxpr.eqns.pop(i)
                        return remove_duplicate_equations(jaxpr, skip_first=i - 1)

    # apply reccursively
    jaxpr = jaxpr.replace(
        eqns=[
            eqn.replace(
                params={
                    k: remove_duplicate_equations(v) if type(v) is Jaxpr else v
                    for k, v in eqn.params.items()
                }
            )
            for eqn in jaxpr.eqns
        ]
    )
    jaxpr = jaxpr.replace(
        eqns=[
            eqn.replace(
                params={
                    k: v.replace(jaxpr=remove_duplicate_equations(v.jaxpr))
                    if type(v) is ClosedJaxpr
                    else v
                    for k, v in eqn.params.items()
                }
            )
            for eqn in jaxpr.eqns
        ]
    )

    return jaxpr


def optimize_jaxpr(closed_jaxpr: ClosedJaxpr) -> ClosedJaxpr:  # pragma: no cover
    closed_jaxpr = remove_duplicate_constants(closed_jaxpr)
    jaxpr = closed_jaxpr.jaxpr
    jaxpr, input_indices = remove_deadcode(jaxpr)
    jaxpr = remove_duplicate_equations(jaxpr)
    return closed_jaxpr.replace(jaxpr=jaxpr), input_indices


reduce_compile_time = closed_jaxpr_transform_to_fn_transform(optimize_jaxpr)


# TEST
# Compile time of allegro with small amount of deadcode and duplicate equations
# without calling reduce_compile_time: 51 seconds
# with calling reduce_compile_time: 53 seconds
#
# Compile time of allegro with large amount of deadcode and duplicate equations
# without calling reduce_compile_time: 1 minute 14 seconds
# with calling reduce_compile_time: 1 minute 25 seconds
