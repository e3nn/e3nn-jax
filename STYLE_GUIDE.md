# Broadcasting
Favor `jax.vmap` to broadcasting

# IrrepsArray
`IrrepsArray` is a triplet of `(irreps, array, list)` where

- `irreps: Irreps`
- `array: jnp.ndarray` of shape `shape` with `shape[-1] == irreps.dim`
- `list: List[Optional[jnp.ndarray]]` with one entry per entry in `irreps`. If the entry is `None` is means that the block is filled with zeros. The `i`th entry is shape `shape[:-1] + (mul, ir.dim)` where `mul, ir = irreps[i]`

- Favor function to get `IrrepsArray` as input/output
- Ideally implement the function for both `.array` and `.list` and output a new `IrrepsArray`
- If not, use either `.array` or `.list` and create a new `IrrepsArray` using `IrrepsArray(irreps, array)` or `IrrepsArray.from_list`

The idea is to rely on `jax.jit` to remove the dead code during compilation.

# Class vs Function
Prefer functions to classes. Implement as a class...

- When the operation contains parameters (`hk.Module`), this all to create an instance and use it multiple times with the same parameters.
- When the input format has to be determined before feeding the data. (It is the case of `FunctionalTensorProduct`)

If possible, decorate the function with `e3nn_jax.util.no_data.overload_for_irreps_without_data` to allow the function to determine the output `Irreps` without providing input data.
