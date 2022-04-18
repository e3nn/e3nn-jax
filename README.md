# e3nn-jax
[![Coverage Status](https://coveralls.io/repos/github/e3nn/e3nn-jax/badge.svg?branch=main)](https://coveralls.io/github/e3nn/e3nn-jax?branch=main)

## Installation

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

## What is different from the PyTorch version?

- no more `shared_weights` and `internal_weights` in `TensorProduct`. Extensive use of `jax.vmap` instead (see example below)
- support of python structure `IrrepsData` that contains a contiguous version of the data and a list of `jnp.ndarray` for the data. This allows to avoid unnecessary `jnp.concatenante` followed by indexing to reverse the concatenation
- support of `None` in the list of `jnp.ndarray` to avoid unnecessary computation with zeros

## Example

Example with the `Irreps` class.
This class specifies a direct sum of irreducible representations.
It does not contain any actual data. It is use to specify the "type" of the data under rotation.

```python
import e3nn_jax as e3nn

irreps = e3nn.Irreps("2x0e + 3x1e")  # 2 even scalars and 3 even vectors
irreps = irreps + irreps  # 2x0e+3x1e+2x0e+3x1e
irreps.D_from_angles(alpha=1.57, beta=1.57, gamma=0.0)  # 22x22 matrix
```

It also includes the parity.
```python
irreps = e3nn.Irreps("0e + 0o")  # an even scalar and an odd scalar
irreps.D_from_angles(alpha=0.0, beta=0.0, gamma=0.0, k=1)  # the matrix that applies parity
# DeviceArray([[ 1.,  0.],
#              [ 0., -1.]], dtype=float32)
```

`IrrepsData` contains both the irreps and the data.
Here is the example of the tensor product of the two vectors.
```python
out = e3nn.full_tensor_product(
    e3nn.IrrepsData.from_contiguous("1o", jnp.array([2.0, 0.0, 0.0])),
    e3nn.IrrepsData.from_contiguous("1o", jnp.array([0.0, 2.0, 0.0]))
)
# out is of type `IrrepsData` and contains the following fields:

out.irreps
# 1x0e+1x1e+1x2e

out.contiguous
# DeviceArray([0.  , 0.  , 0.  , 2.83, 0.  , 2.83, 0.  , 0.  , 0.  ], dtype=float32)

out.list
# [DeviceArray([[0.]], dtype=float32),
#  DeviceArray([[0.  , 0.  , 2.83]], dtype=float32),
#  DeviceArray([[0.  , 2.83, 0.  , 0.  , 0.  ]], dtype=float32)]
```

The two fields `contiguous` and `list` contain the same information under different forms.
This is not a performence issue, we rely on `jax.jit` to optimize the code and get rid of the unused operations.

## Complete example

Usage of `FullyConnectedTensorProduct` in the `torch` version ([e3nn](github.com/e3nn/e3nn) repo):
```python
from e3nn import o3

irreps_in1 = o3.Irreps("1e")
irreps_in2 = o3.Irreps("1e")
irreps_out = o3.Irreps("1e")

tp = o3.FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)

x1 = irreps_in1.randn(10, -1)
x2 = irreps_in2.randn(10, -1)

out = tp(x1, x2)
```

and in the `jax` version (this repo):
```python
import jax
import e3nn_jax as e3nn
import haiku as hk

irreps_out = e3nn.Irreps("1e")

@hk.without_apply_rng
@hk.transform
def tp(x1, x2):
    return e3nn.FullyConnectedTensorProduct(irreps_out)(x1, x2)

irreps_in1 = e3nn.Irreps("1e")
irreps_in2 = e3nn.Irreps("1e")

x1 = e3nn.IrrepsData.randn(irreps_in1, jax.random.PRNGKey(0), (10,))
x2 = e3nn.IrrepsData.randn(irreps_in2, jax.random.PRNGKey(1), (10,))
w = tp.init(jax.random.PRNGKey(2), x1, x2)

out = tp.apply(w, x1, x2)
```

The `jax` version require more boiler-plate (haiku) and more verbose code (with the random keys).
However note that the input irreps does not need to be provided to `FullyConnectedTensorProduct` because it will obtain it from its inputs.
