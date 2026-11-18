"""Interleaved A/B comparison of two e3nn_jax source trees.

Example::

    python -m benchmarks.ab --a /path/to/baseline-worktree --b . -k mul_scalar_weights --rounds 3

Each round runs every (workload, size, mode) once for A and once for B, each in
a fresh process, alternating the order between rounds to cancel drift (GPU
clocks/thermals). Reports the median over rounds of the per-run medians.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

from benchmarks.bench import run_isolated, select


def main(argv=None):
    p = argparse.ArgumentParser(prog="benchmarks.ab")
    p.add_argument("--a", required=True, help="source tree for A (baseline)")
    p.add_argument("--b", required=True, help="source tree for B (candidate)")
    p.add_argument("-k", action="append", default=[])
    p.add_argument("--modes", nargs="+", default=["fwd", "bwd"])
    p.add_argument("--sizes", nargs="+", type=int, default=None)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    envs = {}
    for tag, path in (("A", args.a), ("B", args.b)):
        env = dict(os.environ)
        env["E3NN_JAX_PATH"] = str(Path(path).resolve())
        envs[tag] = env

    print(f"A = {Path(args.a).resolve()}\nB = {Path(args.b).resolve()}")
    print(
        f"{'workload':40s} {'size':>8s} {'mode':4s} {'A ms':>9s} {'B ms':>9s} {'speedup':>8s}"
        f" {'A temp MB':>10s} {'B temp MB':>10s} {'A kern':>6s} {'B kern':>6s} {'A comp s':>8s} {'B comp s':>8s}"
        f" {'A dev ms':>9s} {'B dev ms':>9s} {'dev':>7s}"
    )
    rows = []
    for w in select(args.k):
        for size in args.sizes or w.sizes:
            for mode in args.modes:
                runs = {"A": [], "B": []}
                for r in range(args.rounds):
                    order = ("A", "B") if r % 2 == 0 else ("B", "A")
                    for tag in order:
                        runs[tag].append(
                            run_isolated(
                                w.name,
                                size,
                                mode,
                                "float32",
                                args.samples,
                                env=envs[tag],
                            )
                        )
                ok = {t: [x for x in runs[t] if "error" not in x] for t in runs}
                if not ok["A"] or not ok["B"]:
                    errs = {
                        t: next((x["error"] for x in runs[t] if "error" in x), "")
                        for t in runs
                    }
                    print(
                        f"{w.name:40s} {size:8d} {mode:4s} FAILED A:{errs['A'][:60]} B:{errs['B'][:60]}"
                    )
                    rows.append(dict(workload=w.name, size=size, mode=mode, runs=runs))
                    continue
                med = {
                    t: statistics.median(x["time_median_s"] for x in ok[t]) for t in ok
                }
                last = {t: ok[t][-1] for t in ok}
                dev = {
                    t: statistics.median(x.get("device_time_s", -1.0) for x in ok[t])
                    for t in ok
                }
                print(
                    f"{w.name:40s} {size:8d} {mode:4s} {med['A'] * 1e3:9.3f} {med['B'] * 1e3:9.3f}"
                    f" {med['A'] / med['B']:7.2f}x"
                    f" {last['A']['temp_bytes'] / 2**20:10.1f} {last['B']['temp_bytes'] / 2**20:10.1f}"
                    f" {last['A']['num_kernels']:6d} {last['B']['num_kernels']:6d}"
                    f" {last['A']['compile_s']:8.2f} {last['B']['compile_s']:8.2f}"
                    f" {dev['A'] * 1e3:9.3f} {dev['B'] * 1e3:9.3f} {dev['A'] / dev['B']:6.2f}x",
                    flush=True,
                )
                rows.append(
                    dict(
                        workload=w.name,
                        size=size,
                        mode=mode,
                        median_s=med,
                        device_s=dev,
                        runs=runs,
                    )
                )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
