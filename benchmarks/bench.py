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


def run_one(name, size, mode, dtype_name, samples, hlo_path=None) -> dict:
    """Measure a single (workload, size, mode) in the current process."""
    if dtype_name == "float64":
        jax.config.update("jax_enable_x64", True)
    w = REGISTRY[name]
    f, fargs = build(w, size, jnp.dtype(dtype_name), mode)
    m, hlo = measure(f, fargs, samples=samples)
    if hlo_path:
        Path(hlo_path).write_text(hlo)
    stats = jax.devices()[0].memory_stats() or {}
    return dict(
        workload=name,
        size=size,
        mode=mode,
        peak_device_bytes=int(stats.get("peak_bytes_in_use", -1)),
        **m.to_dict(),
    )


def run_isolated(
    name, size, mode, dtype_name, samples, hlo_path=None, env=None
) -> dict:
    """Run :func:`run_one` in a fresh interpreter (clean GPU memory, clean peak stats)."""
    cmd = [sys.executable, "-m", "benchmarks.bench", "--one", name, str(size), mode]
    cmd += ["--dtype", dtype_name, "--samples", str(samples)]
    if hlo_path:
        cmd += ["--hlo-path", str(hlo_path)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    err = proc.stderr.strip().splitlines() or ["no output"]
    msg = next((ln for ln in reversed(err) if "Error" in ln), err[-1])
    return dict(workload=name, size=size, mode=mode, error=msg[:200])


def _fmt(r) -> str:
    head = f"{r['workload']:40s} {r['size']:8d} {r['mode']:4s}"
    if "error" in r:
        return f"{head} FAILED {r['error'][:110]}"
    iqr = 100 * (r["time_q3_s"] - r["time_q1_s"]) / r["time_median_s"]
    return (
        f"{head} {r['time_median_s'] * 1e3:8.3f}ms {iqr:5.1f} {r['compile_s']:7.2f}s"
        f" {r['temp_bytes'] / 2**20:8.1f} {r['peak_device_bytes'] / 2**20:8.1f}"
        f" {r['num_kernels']:7d} {r['flops'] / 1e9:8.2f}"
    )


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
    p.add_argument(
        "--no-isolate",
        action="store_true",
        help="run everything in this process (faster, but GPU memory accumulates)",
    )
    p.add_argument("--list", action="store_true")
    p.add_argument(
        "--one", nargs=3, metavar=("NAME", "SIZE", "MODE"), help=argparse.SUPPRESS
    )
    p.add_argument("--hlo-path", default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if args.one:
        name, size, mode = args.one
        r = run_one(name, int(size), mode, args.dtype, args.samples, args.hlo_path)
        print(json.dumps(r))
        return

    if args.list:
        for w in REGISTRY.values():
            print(f"{w.name:40s} sizes={w.sizes}  {w.description}")
        return

    meta = dict(
        git_ref=_git("rev-parse", "--abbrev-ref", "HEAD"),
        git_sha=_git("rev-parse", "--short", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
        jax=jax.__version__,
        dtype=args.dtype,
        python=platform.python_version(),
        time=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    if args.no_isolate:
        dev = jax.devices()[0]
        meta["device"] = f"{dev.platform}:{dev.device_kind}"
    print(json.dumps(meta), file=sys.stderr)

    hlo_dir = None
    if args.dump_hlo and args.out:
        hlo_dir = args.out.with_suffix("") / "hlo"
        hlo_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"{'workload':40s} {'size':>8s} {'mode':4s} {'median':>10s} {'IQR%':>5s}"
        f" {'compile':>8s} {'temp MB':>8s} {'peak MB':>8s} {'kernels':>7s} {'GFLOP':>8s}"
    )
    results = []
    for w in select(args.k):
        for size in args.sizes or w.sizes:
            for mode in args.modes:
                hlo_path = None
                if hlo_dir is not None:
                    hlo_path = (
                        hlo_dir / f"{w.name.replace('/', '__')}__{size}__{mode}.hlo"
                    )
                if args.no_isolate:
                    try:
                        r = run_one(
                            w.name, size, mode, args.dtype, args.samples, hlo_path
                        )
                    except Exception as e:  # noqa: BLE001 - report and keep going
                        r = dict(
                            workload=w.name,
                            size=size,
                            mode=mode,
                            error=f"{type(e).__name__}: {str(e).splitlines()[0][:150]}",
                        )
                else:
                    r = run_isolated(
                        w.name, size, mode, args.dtype, args.samples, hlo_path
                    )
                print(_fmt(r), flush=True)
                results.append(r)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(meta=meta, results=results), indent=1))
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
