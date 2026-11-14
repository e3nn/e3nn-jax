"""Performance benchmarks for e3nn-jax (not part of the installed package)."""

import os

# Must happen before JAX initialises its backend: lets XLA's memory analysis and
# the process' real usage match, and avoids reserving 75% of the GPU up front.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
