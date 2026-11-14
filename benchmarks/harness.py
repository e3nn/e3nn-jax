"""Measurement primitives: compile, time, and statically analyse a jitted function."""

import dataclasses
import re
import statistics
import time
from typing import Any, Callable

import jax

# Ops in the ENTRY computation that do not become device work.
_NON_KERNEL_OPS = {
    "parameter",
    "constant",
    "tuple",
    "get-tuple-element",
    "bitcast",
    "opt-barrier",
}
_TRACKED_OPS = (
    "fusion",
    "copy",
    "transpose",
    "concatenate",
    "gather",
    "scatter",
    "dot",
    "custom-call",
    "reduce",
    "dynamic-slice",
    "dynamic-update-slice",
)
_INSTR_RE = re.compile(r"^\s*(?:ROOT\s+)?%?[\w.\-]+\s*=\s*[^=]*?\s([a-z][a-z0-9\-]*)\(")


@dataclasses.dataclass
class Measurement:
    trace_s: float
    compile_s: float
    time_median_s: float
    time_q1_s: float
    time_q3_s: float
    samples: int
    inner_iters: int
    flops: float
    bytes_accessed: float
    temp_bytes: int
    argument_bytes: int
    output_bytes: int
    num_kernels: int
    op_counts: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def entry_op_counts(hlo_text: str) -> tuple[int, dict]:
    """Count ops in the ENTRY computation of optimised HLO.

    Each non-trivial top-level op is (approximately) one kernel launch on GPU.
    Also counts tracked op kinds anywhere in the module (inside fusions too).
    """
    counts = {k: 0 for k in _TRACKED_OPS}
    num_kernels = 0
    in_entry = False
    for line in hlo_text.splitlines():
        if line.startswith("ENTRY"):
            in_entry = True
            continue
        if in_entry and line.startswith("}"):
            in_entry = False
            continue
        m = _INSTR_RE.match(line)
        if not m:
            continue
        op = m.group(1)
        if in_entry and op not in _NON_KERNEL_OPS:
            num_kernels += 1
        if op in counts:
            counts[op] += 1
    return num_kernels, counts


def _block(x):
    return jax.block_until_ready(x)


def measure(
    fn: Callable,
    args: tuple[Any, ...],
    *,
    min_sample_s: float = 0.02,
    samples: int = 15,
    max_inner: int = 1000,
) -> tuple[Measurement, str]:
    """Jit ``fn``, compile it for ``args`` and time it on device.

    Returns the measurement and the optimised HLO text.
    """
    t0 = time.perf_counter()
    lowered = jax.jit(fn).lower(*args)
    t1 = time.perf_counter()
    compiled = lowered.compile()
    t2 = time.perf_counter()

    hlo = compiled.as_text()
    num_kernels, op_counts = entry_op_counts(hlo)

    cost = compiled.cost_analysis() or {}
    if isinstance(cost, (list, tuple)):  # older JAX versions
        cost = cost[0] if cost else {}
    mem = compiled.memory_analysis()

    # Timing never includes tracing/compilation: we call the already-compiled
    # executable. Every call is blocked on (JAX dispatch is asynchronous), so a
    # sample measures per-call latency, matching examples/tensor_product_benchmark.py.
    def run(inner):
        t = time.perf_counter()
        for _ in range(inner):
            _block(compiled(*args))
        return time.perf_counter() - t

    # Warm-up (lazy allocation, autotuning, clock ramp-up).
    run(10)

    # Calibrate the number of calls per sample so a sample lasts >= min_sample_s.
    inner = 1
    while True:
        dt = run(inner)
        if dt >= min_sample_s or inner >= max_inner:
            break
        inner = min(
            max_inner, max(inner * 2, int(inner * min_sample_s / max(dt, 1e-9)))
        )

    times = [run(inner) / inner for _ in range(samples)]

    q = statistics.quantiles(times, n=4)
    return (
        Measurement(
            trace_s=t1 - t0,
            compile_s=t2 - t1,
            time_median_s=statistics.median(times),
            time_q1_s=q[0],
            time_q3_s=q[2],
            samples=samples,
            inner_iters=inner,
            flops=float(cost.get("flops", -1.0)),
            bytes_accessed=float(cost.get("bytes accessed", -1.0)),
            temp_bytes=int(getattr(mem, "temp_size_in_bytes", -1)),
            argument_bytes=int(getattr(mem, "argument_size_in_bytes", -1)),
            output_bytes=int(getattr(mem, "output_size_in_bytes", -1)),
            num_kernels=num_kernels,
            op_counts=op_counts,
        ),
        hlo,
    )
