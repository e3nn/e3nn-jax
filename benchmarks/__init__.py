"""Performance benchmarks for e3nn-jax (not part of the installed package)."""

import os
import sys

# Must happen before JAX initialises its backend: lets XLA's memory analysis and
# the process' real usage match, and avoids reserving 75% of the GPU up front.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# A/B testing: import e3nn_jax from another source tree (e.g. a baseline worktree).
if os.environ.get("E3NN_JAX_PATH"):
    sys.path.insert(0, os.environ["E3NN_JAX_PATH"])
