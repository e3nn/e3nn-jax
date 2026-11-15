"""Per-kernel device-time breakdown of a workload, from a JAX profiler trace.

Example::

    python -m benchmarks.profile tp_features_sh/nequip 100000 bwd --top 20

Uses ``jax.profiler.trace(create_perfetto_trace=True)`` and parses the
Perfetto JSON, so it works without TensorBoard/XProf (CUDA and ROCm).
Kernel names map to fusion names in the optimised HLO (``--hlo`` writes it).
"""

import argparse
import collections
import glob
import gzip
import json
import os
import tempfile

import jax

from benchmarks.bench import build
from benchmarks.workloads import REGISTRY

import jax.numpy as jnp


def kernel_times(trace_path: str) -> tuple[dict, dict]:
    with gzip.open(trace_path, "rt") as f:
        trace = json.load(f)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    # Device (GPU) tracks: find pids whose process name mentions a GPU/stream.
    pid_names = {
        e["pid"]: e["args"]["name"]
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name"
    }
    device_pids = {
        pid
        for pid, name in pid_names.items()
        if "/device:GPU" in name or "/device:ROCM" in name or "GPU:" in name
    }
    total = collections.defaultdict(float)
    count = collections.Counter()
    for e in events:
        if e.get("ph") != "X" or e.get("pid") not in device_pids:
            continue
        name = e.get("name", "?")
        total[name] += e.get("dur", 0.0)  # microseconds
        count[name] += 1
    return total, count


def main(argv=None):
    p = argparse.ArgumentParser(prog="benchmarks.profile")
    p.add_argument("workload")
    p.add_argument("size", type=int)
    p.add_argument("mode", choices=["fwd", "bwd"])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--hlo", default=None, help="write optimised HLO to this path")
    args = p.parse_args(argv)

    f, fargs = build(REGISTRY[args.workload], args.size, jnp.float32, args.mode)
    compiled = jax.jit(f).lower(*fargs).compile()
    if args.hlo:
        with open(args.hlo, "w") as fh:
            fh.write(compiled.as_text())
    for _ in range(5):
        jax.block_until_ready(compiled(*fargs))

    with tempfile.TemporaryDirectory() as d:
        with jax.profiler.trace(d, create_perfetto_trace=True):
            for _ in range(args.iters):
                jax.block_until_ready(compiled(*fargs))
        paths = glob.glob(os.path.join(d, "**", "*.trace.json.gz"), recursive=True)
        total, count = kernel_times(paths[0])

    grand = sum(total.values())
    print(
        f"{args.workload} {args.size} {args.mode}: device time per call "
        f"{grand / args.iters / 1e3:.3f} ms over {sum(count.values()) // args.iters} kernels"
    )
    for name, t in sorted(total.items(), key=lambda kv: -kv[1])[: args.top]:
        print(
            f"  {t / args.iters / 1e3:8.3f} ms {100 * t / grand:5.1f}%  x{count[name] // args.iters:<3d} {name[:110]}"
        )


if __name__ == "__main__":
    main()
