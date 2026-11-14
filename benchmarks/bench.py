"""Run benchmark workloads and write results as JSON.

Examples::

    python -m benchmarks.bench --list
    python -m benchmarks.bench -k tp_features_sh -k macro --modes fwd bwd
    python -m benchmarks.bench --out benchmarks/results/develop.json --dump-hlo
"""

import argparse
import fnmatch
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import e3nn_jax as e3nn
from benchmarks.harness import measure
from benchmarks.workloads import REGISTRY

ROOT = Path(__file__).resolve().parent.parent


def _git(*args) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _as_array(out):
    return out.array if isinstance(out, e3nn.IrrepsArray) else out


def build(workload, size, dtype, mode, seed=0):
    fn, diff_args, nondiff_args = workload.make(size, dtype, jax.random.PRNGKey(seed))

    if mode == "fwd":

        def f(diff_args, nondiff_args):
            return _as_array(fn(*diff_args, *nondiff_args))

    elif mode == "bwd":

        def loss(diff_args, nondiff_args):
            out = _as_array(fn(*diff_args, *nondiff_args))
            return jnp.sum(jnp.tanh(out))

        f = jax.value_and_grad(loss)
    else:
        raise ValueError(mode)

    return f, (diff_args, nondiff_args)


def select(patterns):
    if not patterns:
        return list(REGISTRY.values())
    return [
        w
        for w in REGISTRY.values()
        if any(p in w.name or fnmatch.fnmatch(w.name, p) for p in patterns)
    ]


def main(argv=None):
    p = argparse.ArgumentParser(prog="benchmarks.bench")
    p.add_argument(
        "-k",
        action="append",
        default=[],
        help="substring/glob of workload names (repeatable)",
    )
    p.add_argument("--modes", nargs="+", default=["fwd", "bwd"], choices=["fwd", "bwd"])
    p.add_argument(
        "--sizes", nargs="+", type=int, default=None, help="override workload sizes"
    )
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--out", type=Path, default=None, help="JSON output path")
    p.add_argument(
        "--dump-hlo", action="store_true", help="write optimised HLO next to the JSON"
    )
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        for w in REGISTRY.values():
            print(f"{w.name:40s} sizes={w.sizes}  {w.description}")
        return

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.dtype(args.dtype)

    dev = jax.devices()[0]
    meta = dict(
        git_ref=_git("rev-parse", "--abbrev-ref", "HEAD"),
        git_sha=_git("rev-parse", "--short", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
        jax=jax.__version__,
        device=f"{dev.platform}:{dev.device_kind}",
        dtype=args.dtype,
        python=platform.python_version(),
        time=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    print(json.dumps(meta), file=sys.stderr)

    hlo_dir = None
    if args.dump_hlo and args.out:
        hlo_dir = args.out.with_suffix("") / "hlo"
        hlo_dir.mkdir(parents=True, exist_ok=True)

    results = []
    header = (
        f"{'workload':40s} {'size':>8s} {'mode':4s} {'median':>10s} {'IQR%':>5s}"
        f" {'compile':>8s} {'temp MB':>8s} {'kernels':>7s} {'GFLOP':>8s}"
    )
    print(header)
    for w in select(args.k):
        for size in args.sizes or w.sizes:
            for mode in args.modes:
                try:
                    f, fargs = build(w, size, dtype, mode)
                    m, hlo = measure(f, fargs, samples=args.samples)
                except (
                    Exception
                ) as e:  # noqa: BLE001 - report and keep going (e.g. OOM)
                    msg = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
                    print(f"{w.name:40s} {size:8d} {mode:4s} FAILED {msg}")
                    results.append(
                        dict(workload=w.name, size=size, mode=mode, error=msg)
                    )
                    continue
                iqr = 100 * (m.time_q3_s - m.time_q1_s) / m.time_median_s
                print(
                    f"{w.name:40s} {size:8d} {mode:4s} {m.time_median_s * 1e3:8.3f}ms {iqr:5.1f}"
                    f" {m.compile_s:7.2f}s {m.temp_bytes / 2**20:8.1f} {m.num_kernels:7d} {m.flops / 1e9:8.2f}",
                    flush=True,
                )
                results.append(
                    dict(workload=w.name, size=size, mode=mode, **m.to_dict())
                )
                if hlo_dir is not None:
                    (
                        hlo_dir / f"{w.name.replace('/', '__')}__{size}__{mode}.hlo"
                    ).write_text(hlo)
                del f, fargs

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(meta=meta, results=results), indent=1))
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
