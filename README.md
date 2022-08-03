# e3nn-jax [![Coverage Status](https://coveralls.io/repos/github/e3nn/e3nn-jax/badge.svg?branch=main)](https://coveralls.io/github/e3nn/e3nn-jax?branch=main)

# [Documentation](https://e3nn-jax.readthedocs.io/en/latest) [![Documentation Status](https://readthedocs.org/projects/e3nn-jax/badge/?version=latest)](https://e3nn-jax.readthedocs.io/en/latest/?badge=latest)



# :boom: Warning :boom:
Please always check the ChangeLog for breaking changes.

# Installation

To install the latest released version:
```bash
pip install --upgrade e3nn-jax
```

To install the latest GitHub version:
```bash
pip install git+https://github.com/e3nn/e3nn-jax.git
```

To install from a local copy for development, we recommend creating a virtual enviroment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To check that the tests are running:
```bash
pip install pytest
pytest tests/tensor_products_test.py
```

# What is different from the PyTorch version?

- No more `shared_weights` and `internal_weights` in `TensorProduct`. Extensive use of `jax.vmap` instead (see example below)
- Support of python structure `IrrepsArray` that contains a contiguous version of the data and a list of `jnp.ndarray` for the data. This allows to avoid unnecessary `jnp.concatenante` followed by indexing to reverse the concatenation (even that `jax.jit` is probably able to unroll the concatenations)
- Support of `None` in the list of `jnp.ndarray` to avoid unnecessary computation with zeros (basically imposing `0 * x = 0`, which is not simplified by default by jax because `0 * nan = nan`)

# Examples

The examples are moved in the documentation.
