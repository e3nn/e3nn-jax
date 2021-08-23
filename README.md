# e3nn-jax

## Shared weights

`torch` version:
```python
f = o3.FullyConnectedTensorProduct(irreps1, irreps2, irreps3, shared_weights=True)

f(x, y)
```

`jax` version:
```python
tp = FullyConnectedTensorProduct(irreps1, irreps2, irreps3)
w = [jax.random.normal(key, i.path_shape) for i in tp.instructions if i.has_weight]
f = jax.vmap(tp.left_right, (None, 0, 0), 0)
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
tp = FullyConnectedTensorProduct(irreps1, irreps2, irreps3)
w = [jax.random.normal(key, (10,) + i.path_shape) for i in tp.instructions if i.has_weight]
f = jax.vmap(tp.left_right, (0, 0, 0), 0)
f = jax.jit(f)

f(w, x, y)
```

## Extra channel index

`torch` version not implemented

`jax` version just needs an extra bunch of `vmap` calls:
```python
def compose(f, g):
    return lambda *x: g(f(*x))

def tp_extra_channels(irreps_in1, irreps_in2, irreps_out):
    tp = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)

    f = tp.left_right
    f = jax.vmap(f, (0, None, None), 0)  # channel_out
    f = jax.vmap(f, (0, None, 0), 0)  # channel_in2
    f = jax.vmap(f, (0, 0, None), 0)  # channel_in1
    f = compose(f, lambda z: jnp.sum(z, (0, 1)) / jnp.sqrt(z.shape[0] * z.shape[1]))
    tp.left_right = f

    return tp

tp = tp_extra_channels(irreps, irreps, irreps)
f = jax.vmap(tp.left_right, (None, 0, 0), 0)  # batch
f = jax.jit(f)

w = [jax.random.normal(k, (16, 32, 48) + i.path_shape) for i in tp.instructions if i.has_weight]
# x1.shape = (batch, ch_in1, irreps_in1)
# x2.shape = (batch, ch_in2, irreps_in2)
z = f(w, x1, x2)
# z.shape = (batch, ch_out, irreps_out)
```

## Convolution

```python
import jax
import jax.numpy as jnp
from e3nn_jax import FullyConnectedTensorProduct, Irreps, spherical_harmonics, soft_one_hot_linspace

# TODO make it a flax module
irreps_in = Irreps('2x0e + 3x1e')
irreps_sh = Irreps('3x0e + 2x1e')
irreps_out = Irreps('2x0e + 1e')

edge_src = jnp.array([0, 0, 1, 1])  # TODO use external lib to make it from positions
edge_dst = jnp.array([0, 1, 0, 1])
node = irreps_in.randn(key, (2, -1))

edge_vec = jax.random.normal(key, (edge_src.shape[0], 3))
sh = spherical_harmonics(irreps_sh, edge_vec, True)
edge_len_emb = soft_one_hot_linspace(
    x=jnp.linalg.norm(edge_vec, axis=-1),
    start=0.0,
    end=2.0,
    number=10,
    basis='smooth_finite',
    cutoff=True
) * 10**0.5

tp = FullyConnectedTensorProduct(irreps_in, irreps_sh, irreps_out)
f = jax.vmap(tp.left_right, (0, 0, 0), 0)

w = [
    jnp.einsum(
        "ei,i...->e...",
        edge_len_emb,
        jax.random.normal(key, (10,) + i.path_shape)
    )
    for i in tp.instructions if i.has_weight
]
# TODO use a MLP

edge = f(w, node[edge_src], sh)
node = jax.ops.index_add(jnp.zeros((node.shape[0], edge.shape[1])), edge_dst, edge)
```