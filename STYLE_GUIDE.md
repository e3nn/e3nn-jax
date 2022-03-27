# Broadcasting
Favor `jax.vmap` to broadcasting

# IrrepsData
`IrrepsData` is a triplet of `(irreps, contiguous, list)` where

- `irreps: Irreps`
- `contiguous: jnp.ndarray` of shape `shape + (irreps.dim,)`
- `list: List[Optional[jnp.ndarray]]` with one entry per entry in `irreps`. If the entry is `None` is means that the block is filled with zeros. The `i`th entry is shape `shape + (mul, ir.dim)` where `mul, ir = irreps[i]`

- Favor function to get `IrrepsData` as input/output
- Ideally implement the function for both `.contiguous` and `.list` and output a new `IrrepsData`
- If not, use either `.contiguous` or `.list` and create a new `IrrepsData` using `IrrepsData.from_contiguous` or `IrrepsData.from_list`

The idea is to rely on `jax.jit` to remove the dead code during compilation.

# Class vs Function
Prefer functions to classes. Implement as a class...

- When the operation contains parameters (`hk.Module`), this all to create an instance and use it multiple times with the same parameters.
- When the input format has to be determined before feeding the data. (It is the case of `FunctionalTensorProduct`)

If possible, decorate the function with `e3nn_jax.util.no_data.overload_for_irreps_without_data` to allow the function to determine the output `Irreps` without providing input data.
