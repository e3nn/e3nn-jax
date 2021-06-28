# e3nn-jax

## Shared weights

`torch` version:
```python
f = o3.FullyConnectedTensorProduct(irreps1, irreps2, irreps3, shared_weights=True)

f(x, y)
```

`jax` version:
```python
ins, nw, f, r = fully_connected_tensor_product(irreps1, irreps2, irreps3)
f = jax.vmap(f, (None, 0, 0), 0)
f = jax.jit(f)

f(w, x, y)
```

## Batch weights

`torch` version:
```python
f = o3.FullyConnectedTensorProduct(irreps1, irreps2, irreps3, shared_weights=False)

f(x, y, w)
```

`jax` version:
```python
ins, nw, f, r = fully_connected_tensor_product(irreps1, irreps2, irreps3)
f = jax.vmap(f, (0, 0, 0), 0)
f = jax.jit(f)

f(w, x, y)
```

## Convolution

```python
import jax
import jax.numpy as jnp
from e3nn_jax import fully_connected_tensor_product
from e3nn.o3 import Irreps  # TODO convert in e3nn_jax

# TODO make it a flax module
irreps_in = Irreps('0e + 1e')
irreps_sh = Irreps('0e + 1e')
irreps_out = Irreps('0e + 1e')

edge_src = jnp.array([0, 0, 1, 1])  # TODO use external lib to make it from positions
edge_dst = jnp.array([0, 1, 0, 1])
node = jnp.ones((2, irreps_in.dim))
sh = jnp.ones((edge_src.shape[0], irreps_sh.dim))  # TODO add sh in e3nn_jax

_, nw, tp, _ = fully_connected_tensor_product(irreps_in, irreps_sh, irreps_out)
tp = jax.vmap(tp, (None, 0, 0), 0)
w = jnp.ones((nw,))  # TODO add soft_one_hot_linspace in e3nn_jax and use a MLP

edge = tp(w, node[edge_src], sh)
node = jax.ops.index_add(jnp.zeros((node.shape[0], edge.shape[1])), edge_dst, edge)
```