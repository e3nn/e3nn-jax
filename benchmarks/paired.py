"""Drift-robust A/B: both source trees compiled in one process, timed in alternation.

Laptop/consumer GPUs change SM and memory clocks under power caps (we see
memory clocks move between 5.5 and 7.8 GHz within seconds), so two separate
measurements of the *same* program can differ by 25%. Here each workload is
compiled once from tree A and once from tree B inside one process (the
``e3nn_jax`` package is imported from each tree in turn; compiled executables
stay valid after the module is swapped out), then timed in short alternating
blocks ``A B B A A B ...``. The reported speedup is the median of the per-pair
ratios, so slow clock drift cancels.

Example::

    python -m benchmarks.paired --a ../baseline --b . -k tp_features_sh --sizes 10000

Each (workload, size, mode) runs in its own subprocess.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

_PURGE = ("e3nn_jax", "benchmarks.workloads", "benchmarks.bench")


def _load_tree(tree: str):
    """Import e3nn_jax from ``tree`` and (re)import the workload registry against it."""
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in _PURGE):
            del sys.modules[name]
    sys.path.insert(0, tree)
    try:
        import e3nn_jax  # noqa: F401
    finally:
        sys.path.remove(tree)
    import e3nn_jax as e3nn

    assert (
        Path(e3nn.__file__).resolve().is_relative_to(Path(tree).resolve())
    ), e3nn.__file__
    # importlib, not ``from benchmarks import ...``: the latter returns the stale
    # submodule attribute cached on the package and would keep tree A's registry.
    import importlib

    workloads = importlib.import_module("benchmarks.workloads")
    bench = importlib.import_module("benchmarks.bench")
    assert Path(workloads.e3nn.__file__).resolve().is_relative_to(Path(tree).resolve())
    return bench, workloads


def run_pair(
    a: str, b: str, workload: str, size: int, mode: str, pairs: int, block_s: float
) -> dict:
    import jax
    import jax.numpy as jnp

    from benchmarks.harness import device_time_s

    progs = {}
    for tag, tree in (("A", a), ("B", b)):
        bench, workloads = _load_tree(tree)
        f, args = bench.build(workloads.REGISTRY[workload], size, jnp.float32, mode)
        t0 = time.perf_counter()
        compiled = jax.jit(f).lower(*args).compile()
        compile_s = time.perf_counter() - t0
        mem = compiled.memory_analysis()
        progs[tag] = dict(
            compiled=compiled,
            args=args,
            compile_s=compile_s,
            temp_bytes=int(getattr(mem, "temp_size_in_bytes", -1)),
        )

    def block(tag, inner):
        p = progs[tag]
        c, args = p["compiled"], p["args"]
        t = time.perf_counter()
        for _ in range(inner):
            jax.block_until_ready(c(*args))
        return (time.perf_counter() - t) / inner

    # Warm-up both (autotuning, allocation, clock boost).
    t_end = time.perf_counter() + 0.5
    while time.perf_counter() < t_end:
        block("A", 3)
        block("B", 3)
    inner = {t: max(1, int(block_s / max(block(t, 3), 1e-7))) for t in progs}

    ta, tb, ratios = [], [], []
    for i in range(pairs):
        order = ("A", "B") if i % 2 == 0 else ("B", "A")
        r = {t: block(t, inner[t]) for t in order}
        ta.append(r["A"])
        tb.append(r["B"])
        ratios.append(r["A"] / r["B"])
    q = statistics.quantiles(ratios, n=4)
    return dict(
        workload=workload,
        size=size,
        mode=mode,
        a_s=statistics.median(ta),
        b_s=statistics.median(tb),
        speedup=statistics.median(ratios),
        speedup_q1=q[0],
        speedup_q3=q[2],
        a_dev_s=device_time_s(progs["A"]["compiled"], progs["A"]["args"]),
        b_dev_s=device_time_s(progs["B"]["compiled"], progs["B"]["args"]),
        a_temp=progs["A"]["temp_bytes"],
        b_temp=progs["B"]["temp_bytes"],
        a_compile_s=progs["A"]["compile_s"],
        b_compile_s=progs["B"]["compile_s"],
    )


def main(argv=None):
    p = argparse.ArgumentParser(prog="benchmarks.paired")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument(
        "-k", action="append", default=[], help="substring filter (repeatable)"
    )
    p.add_argument("--sizes", type=int, nargs="*", default=None)
    p.add_argument("--modes", nargs="*", default=["fwd", "bwd"])
    p.add_argument("--pairs", type=int, default=15)
    p.add_argument("--block-s", type=float, default=0.03)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--one", nargs=3, metavar=("WORKLOAD", "SIZE", "MODE"))
    args = p.parse_args(argv)
    a, b = str(Path(args.a).resolve()), str(Path(args.b).resolve())

    if args.one:
        w, size, mode = args.one
        try:
            r = run_pair(a, b, w, int(size), mode, args.pairs, args.block_s)
        except Exception as e:  # noqa: BLE001
            r = dict(
                workload=w,
                size=int(size),
                mode=mode,
                error=f"{type(e).__name__}: {e}"[:300],
            )
        print("RESULT " + json.dumps(r), flush=True)
        return

    from benchmarks.workloads import REGISTRY

    print(f"A = {a}\nB = {b}")
    print(
        f"{'workload':40s} {'size':>8s} {'mode':4s} {'A ms':>9s} {'B ms':>9s} {'speedup':>8s} {'[q1, q3]':>13s}"
        f" {'A dev':>8s} {'B dev':>8s} {'A tempMB':>9s} {'B tempMB':>9s} {'A comp':>7s} {'B comp':>7s}"
    )
    rows = []
    env = {k: v for k, v in os.environ.items() if k != "E3NN_JAX_PATH"}
    for w in REGISTRY.values():
        if args.k and not any(s in w.name for s in args.k):
            continue
        for size in args.sizes or w.sizes:
            for mode in args.modes:
                cmd = [
                    sys.executable,
                    "-m",
                    "benchmarks.paired",
                    "--a",
                    a,
                    "--b",
                    b,
                    "--pairs",
                    str(args.pairs),
                    "--block-s",
                    str(args.block_s),
                    "--one",
                    w.name,
                    str(size),
                    mode,
                ]
                out = subprocess.run(cmd, capture_output=True, text=True, env=env)
                line = next(
                    (ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")),
                    None,
                )
                r = (
                    json.loads(line[7:])
                    if line
                    else dict(
                        workload=w.name,
                        size=size,
                        mode=mode,
                        error=out.stderr.strip()[-300:],
                    )
                )
                rows.append(r)
                if "error" in r:
                    print(
                        f"{w.name:40s} {size:8d} {mode:4s} FAILED {r['error'][:90]}",
                        flush=True,
                    )
                    continue
                print(
                    f"{w.name:40s} {size:8d} {mode:4s} {r['a_s'] * 1e3:9.3f} {r['b_s'] * 1e3:9.3f}"
                    f" {r['speedup']:7.2f}x [{r['speedup_q1']:.2f}, {r['speedup_q3']:.2f}]"
                    f" {r['a_dev_s'] * 1e3:8.3f} {r['b_dev_s'] * 1e3:8.3f}"
                    f" {r['a_temp'] / 2**20:9.1f} {r['b_temp'] / 2**20:9.1f}"
                    f" {r['a_compile_s']:7.2f} {r['b_compile_s']:7.2f}",
                    flush=True,
                )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
