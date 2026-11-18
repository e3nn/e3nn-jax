"""Per-kernel device-time breakdown of a workload, from a JAX profiler trace.

Example::

    python -m benchmarks.profile tp_features_sh/nequip 100000 bwd --top 20

Uses ``jax.profiler.trace(create_perfetto_trace=True)`` and parses the
Perfetto JSON, so it works without TensorBoard/XProf (CUDA and ROCm).
Kernel names map to fusion names in the optimised HLO (``--hlo`` writes it).
"""

import argparse

import jax

from benchmarks.bench import build
from benchmarks.harness import profile_kernels
from benchmarks.workloads import REGISTRY

import jax.numpy as jnp


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

    total, count = profile_kernels(compiled, fargs, args.iters)

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
