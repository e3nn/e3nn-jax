# Class vs Function
Prefer functions to classes. Implement as a class...

- When the operation contains parameters (`hk.Module`), this all to create an instance and use it multiple times with the same parameters.
- When the input format has to be determined before feeding the data. (It is the case of `Gate` and `FunctionalTensorProduct`)

If possible, decorate the function with `e3nn_jax.util.no_data.overload_for_irreps_without_data` to allow the function to determine the output `Irreps` without providing input data.
