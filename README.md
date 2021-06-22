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
