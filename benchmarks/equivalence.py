"""Numerical equivalence check against golden outputs from a baseline revision.

Workflow::

    # 1. Dump golden data with the *baseline* e3nn_jax (e.g. a worktree of develop):
    JAX_PLATFORMS=cpu PYTHONPATH=/path/to/baseline-worktree:. \
        python benchmarks/equivalence.py dump benchmarks/results/golden.npz

    # 2. Check the working tree against it:
    JAX_PLATFORMS=cpu PYTHONPATH=. python benchmarks/equivalence.py check benchmarks/results/golden.npz

Each case computes outputs and VJPs (random fixed cotangent) for every float
input, in float64 and float32. The check requires:

* the output irreps to be identical;
* float64 results to match the baseline float64 results (max rel. error <= 1e-12);
* float32 error vs the baseline float64 results to be no worse than the
  baseline's own float32 error (ratio <= 2, floor 4 float32 ulps).
"""

import sys

import jax

jax.config.update("jax_enable_x64", True)

import flax.linen as nn  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import e3nn_jax as e3nn  # noqa: E402

F64_TOL = 1e-12
F32_RATIO = 2.0
# Errors below a few float32 ulps (relative to max|ref|) are reassociation noise,
# not accuracy changes: eps32 = 2**-23 ~ 1.19e-7.
F32_FLOOR = 4 * 2.0**-23

CASES = {}


def case(name):
    def deco(f):
        CASES[name] = f
        return f

    return deco


def _n(irreps, key, shape, dtype):
    return e3nn.normal(irreps, jax.random.PRNGKey(key), shape, dtype=dtype)


def _a(key, shape, dtype):
    return jax.random.normal(jax.random.PRNGKey(key), shape, dtype)


# Irreps pairs: unsorted, repeated, both parities, mul > 1, high l.
TP_PAIRS = [
    ("2x0e + 1o", "0o + 2o"),
    ("32x0e + 32x1o + 32x2e", "0e + 1o + 2e"),
    ("4x0e + 4x0o + 4x1e + 4x1o + 4x2e + 4x2o + 4x3e + 4x3o", "0e + 1o + 2e + 3o"),
    ("2x1o + 0e + 1o + 3x2e", "1e + 2x0e + 1o"),
    ("0e + 1o + 2e + 3o + 4e", "0e + 1o + 2e + 3o + 4e"),
    ("3x1e", "3x1e"),
]

for i, (ir1, ir2) in enumerate(TP_PAIRS):
    for norm in ("component", "norm", "none"):
        for regroup in (True, False):

            @case(f"tensor_product/{i}/{norm}/regroup={regroup}")
            def _(dtype, ir1=ir1, ir2=ir2, norm=norm, regroup=regroup):
                x = _n(ir1, 1, (7,), dtype)
                y = _n(ir2, 2, (7,), dtype)
                f = lambda x, y: e3nn.tensor_product(  # noqa: E731
                    x, y, irrep_normalization=norm, regroup_output=regroup
                )
                return f, (x, y)

    @case(f"tensor_product/{i}/filter")
    def _(dtype, ir1=ir1, ir2=ir2):
        x = _n(ir1, 3, (5,), dtype)
        y = _n(ir2, 4, (5,), dtype)
        filt = ["0e", "1o", "2e"]
        return (lambda x, y: e3nn.tensor_product(x, y, filter_ir_out=filt)), (x, y)

    @case(f"tensor_product/{i}/broadcast")
    def _(dtype, ir1=ir1, ir2=ir2):
        x = _n(ir1, 5, (3, 1), dtype)
        y = _n(ir2, 6, (1, 4), dtype)
        return e3nn.tensor_product, (x, y)


@case("tensor_product/zero_chunks")
def _(dtype):
    chunk = _a(7, (5, 3, 3), dtype)
    y = _n("0e + 1o", 8, (5,), dtype)

    def f(chunk, y):
        x = e3nn.from_chunks("2x0e + 3x1o", [None, chunk], (5,), chunk.dtype)
        return e3nn.tensor_product(x, y)

    return f, (chunk, y)


@case("tensor_product/channels_axis")
def _(dtype):
    x = _n("0e + 1o + 2e + 3o", 9, (6, 4), dtype)
    y = _n("0e + 1o + 2e + 3o", 10, (6, 4), dtype)
    filt = e3nn.Irreps("0e + 1o + 2e + 3o")
    return (lambda x, y: e3nn.tensor_product(x, y, filter_ir_out=filt)), (x, y)


for i, (ir1, ir2) in enumerate(
    [("2x0e + 3x1o + 2e", "2x1o + 3x0e + 2e"), ("4x1o + 4x2e", "4x0e + 4x0e")]
):
    for norm in ("component", "norm", "none"):

        @case(f"elementwise_tensor_product/{i}/{norm}")
        def _(dtype, ir1=ir1, ir2=ir2, norm=norm):
            x = _n(ir1, 11, (5,), dtype)
            y = _n(ir2, 12, (5,), dtype)
            return (
                lambda x, y: e3nn.elementwise_tensor_product(
                    x, y, irrep_normalization=norm
                )
            ), (x, y)


for irreps in (
    "0e + 1o",
    "3x0e + 4x1o + 2x2e",
    "8x0e + 8x1o + 8x2e",
    "2e + 0e + 1o + 2x0e",
):
    for norm in ("component", "norm", "none"):
        for normalized_input in (False, True):
            for regroup in (True, False):

                @case(
                    f"tensor_square/{irreps}/{norm}/ni={normalized_input}/rg={regroup}"
                )
                def _(dtype, irreps=irreps, norm=norm, ni=normalized_input, rg=regroup):
                    x = _n(irreps, 13, (6,), dtype)
                    return (
                        lambda x: e3nn.tensor_square(
                            x,
                            irrep_normalization=norm,
                            normalized_input=ni,
                            regroup_output=rg,
                        )
                    ), (x,)


for irreps in ("32x0e + 32x1o + 32x2e", "2x1o + 0e + 3x2e + 1o", "4x0e"):

    @case(f"mul/per_irrep_array/{irreps}")
    def _(dtype, irreps=irreps):
        x = _n(irreps, 14, (5,), dtype)
        w = _a(15, (5, e3nn.Irreps(irreps).num_irreps), dtype)
        return (lambda x, w: x * w), (x, w)

    @case(f"rmul/per_irrep_array/{irreps}")
    def _(dtype, irreps=irreps):
        x = _n(irreps, 16, (5,), dtype)
        w = _a(17, (5, e3nn.Irreps(irreps).num_irreps), dtype)
        return (lambda x, w: w * x), (x, w)

    @case(f"mul/scalar_irreps_array/{irreps}")
    def _(dtype, irreps=irreps):
        x = _n(irreps, 18, (5,), dtype)
        n = e3nn.Irreps(irreps).num_irreps
        w = _n(f"{n}x0e", 19, (5,), dtype)
        return (lambda x, w: x * w), (x, w)

    @case(f"mul/broadcast_scalar/{irreps}")
    def _(dtype, irreps=irreps):
        x = _n(irreps, 20, (5,), dtype)
        w = _a(21, (5, 1), dtype)
        return (lambda x, w: x * w), (x, w)

    @case(f"div/per_irrep_array/{irreps}")
    def _(dtype, irreps=irreps):
        x = _n(irreps, 22, (5,), dtype)
        w = 2.0 + jnp.abs(_a(23, (5, e3nn.Irreps(irreps).num_irreps), dtype))
        return (lambda x, w: x / w), (x, w)


@case("concatenate_regroup")
def _(dtype):
    x = _n("32x0e + 32x1o + 32x2e", 24, (5,), dtype)
    y = _n("0e + 1o + 2e", 25, (5,), dtype)
    filt = e3nn.Irreps("32x0e + 32x1o + 32x2e") + "0e"

    def f(x, y):
        tp = e3nn.tensor_product(x, y, filter_ir_out=filt)
        return e3nn.concatenate([x.filter(filt), tp]).regroup()

    return f, (x, y)


@case("sort_simplify_regroup")
def _(dtype):
    x = _n("2x1o + 0e + 3x2e + 1o + 0e", 26, (5,), dtype)
    return (lambda x: e3nn.concatenate([x.sort(), x.simplify(), x.regroup()])), (x,)


for lmax in (2, 3, 6):
    for norm in ("component", "norm", "integral"):

        @case(f"spherical_harmonics/l{lmax}/{norm}")
        def _(dtype, lmax=lmax, norm=norm):
            v = e3nn.IrrepsArray("1o", _a(27, (9, 3), dtype))
            return (
                lambda v: e3nn.spherical_harmonics(
                    range(lmax + 1), v, normalize=True, normalization=norm
                )
            ), (v,)


@case("tensor_product_with_spherical_harmonics")
def _(dtype):
    x = _n("3x0e + 2x1o + 2e", 28, (6,), dtype)
    v = e3nn.IrrepsArray("1o", _a(29, (6, 3), dtype))
    return (lambda x, v: e3nn.tensor_product_with_spherical_harmonics(x, v, 3)), (x, v)


@case("gate")
def _(dtype):
    x = _n("16x0e + 8x0o + 4x1o + 4x2e + 8x0e", 30, (5,), dtype)
    return e3nn.gate, (x,)


@case("norm_activation")
def _(dtype):
    x = _n("4x0e + 4x1o + 4x2e", 31, (5,), dtype)
    return (
        lambda x: e3nn.norm_activation(x, [jax.nn.tanh, jax.nn.sigmoid, jax.nn.silu])
    ), (x,)


@case("scatter_add_gather")
def _(dtype):
    x = _n("4x0e + 4x1o + 4x2e", 32, (11,), dtype)
    idx = jax.random.randint(jax.random.PRNGKey(33), (30,), 0, 11)

    def f(x):
        m = x[idx]
        return e3nn.zeros(m.irreps, (11,), m.dtype).at[idx].add(m)

    return f, (x,)


class _Linear(nn.Module):
    @nn.compact
    def __call__(self, x):
        return e3nn.flax.Linear("8x0e + 6x1o + 4x2e + 2x3o")(x)


@case("linear_flax")
def _(dtype):
    x = _n("4x0e + 3x1o + 2x1o + 2x2e + 0e", 34, (5,), dtype)
    mod = _Linear()
    params = jax.tree.map(
        lambda p: p.astype(dtype), mod.init(jax.random.PRNGKey(35), x)
    )
    return mod.apply, (params, x)


# --------------------------------------------------------------------------


def _leaves(tree):
    leaves = []
    for leaf in jax.tree_util.tree_leaves(tree):
        leaves.append(
            np.asarray(leaf.array if isinstance(leaf, e3nn.IrrepsArray) else leaf)
        )
    return leaves


def _cast(tree, dtype):
    return jax.tree.map(
        lambda a: a.astype(dtype) if jnp.issubdtype(a.dtype, jnp.floating) else a, tree
    )


def run_case(name, dtype):
    # Inputs are always generated in float64 and then cast, so that the float32
    # and float64 runs see the same values (up to rounding).
    fn, args = CASES[name](jnp.float64)
    args = _cast(args, dtype)

    @jax.jit
    def go(*args):
        out, vjp = jax.vjp(fn, *args)
        ct = jax.tree.map(
            lambda a: jax.random.normal(
                jax.random.PRNGKey(99), a.shape, jnp.float64
            ).astype(a.dtype),
            out,
        )
        return out, vjp(ct)

    out, grads = go(*args)
    irreps = str(out.irreps) if isinstance(out, e3nn.IrrepsArray) else ""
    return irreps, _leaves(out) + _leaves(grads)


def dump(path):
    data = {"__e3nn_file__": np.array(e3nn.__file__)}
    for name in CASES:
        for dt in ("float64", "float32"):
            irreps, arrays = run_case(name, jnp.dtype(dt))
            data[f"{name}|{dt}|irreps"] = np.array(irreps)
            for i, a in enumerate(arrays):
                data[f"{name}|{dt}|{i}"] = a
    np.savez_compressed(path, **data)
    print(f"dumped {len(CASES)} cases using {e3nn.__file__} -> {path}")


def _err(a, ref):
    a = np.asarray(a, np.float64)
    ref = np.asarray(ref, np.float64)
    scale = max(np.max(np.abs(ref)), 1e-30) if ref.size else 1.0
    return float(np.max(np.abs(a - ref)) / scale) if ref.size else 0.0


def check(path, pattern=None):
    gold = np.load(path)
    print(f"golden from {gold['__e3nn_file__']}; checking {e3nn.__file__}")
    failures = 0
    worst64 = 0.0
    worst_ratio = 0.0
    for name in CASES:
        if pattern and pattern not in name:
            continue
        irreps64, new64 = run_case(name, jnp.dtype("float64"))
        irreps32, new32 = run_case(name, jnp.dtype("float32"))
        ref_irreps = str(gold[f"{name}|float64|irreps"])
        problems = []
        if irreps64 != ref_irreps or irreps32 != str(gold[f"{name}|float32|irreps"]):
            problems.append(f"irreps {irreps64} != {ref_irreps}")
        n = len(
            [
                k
                for k in gold.files
                if k.startswith(f"{name}|float64|") and not k.endswith("irreps")
            ]
        )
        if n != len(new64):
            problems.append(f"#arrays {len(new64)} != {n}")
        else:
            for i in range(n):
                ref64 = gold[f"{name}|float64|{i}"]
                old32 = gold[f"{name}|float32|{i}"]
                if ref64.shape != new64[i].shape:
                    problems.append(f"[{i}] shape {new64[i].shape} != {ref64.shape}")
                    continue
                e64 = _err(new64[i], ref64)
                worst64 = max(worst64, e64)
                if e64 > F64_TOL:
                    problems.append(f"[{i}] f64 err {e64:.2e}")
                e_new32 = _err(new32[i], ref64)
                e_old32 = _err(old32, ref64)
                ratio = e_new32 / max(e_old32, F32_FLOOR)
                worst_ratio = max(worst_ratio, ratio)
                if e_new32 > max(F32_RATIO * e_old32, F32_FLOOR):
                    problems.append(
                        f"[{i}] f32 err {e_new32:.2e} vs baseline {e_old32:.2e}"
                    )
        if problems:
            failures += 1
            print(f"FAIL {name}: {'; '.join(problems)}")
    total = len([n for n in CASES if not pattern or pattern in n])
    print(
        f"{total - failures}/{total} cases OK; worst f64 rel err {worst64:.2e};"
        f" worst f32 err ratio vs baseline {worst_ratio:.2f}"
    )
    return failures == 0


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == "dump":
        dump(path)
    elif cmd == "check":
        ok = check(path, sys.argv[3] if len(sys.argv) > 3 else None)
        sys.exit(0 if ok else 1)
    else:
        raise SystemExit(__doc__)
