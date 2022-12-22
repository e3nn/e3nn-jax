# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `e3nn.s2grid_vectors` and `e3nn.pad_to_plot_on_s2grid` to help plotting signals on the sphere
- `e3nn.util.assert_output_dtype` to check the output dtype of a function

## [0.14.0] - 2022-12-16
### Added
- `IrrepsArray.astype` to cast the underlying array
- `e3nn.flax.MultiLayerPerceptron` and `e3nn.haiku.MultiLayerPerceptron`
- `e3nn.IrrepsArray.from_list(..., dtype)`
- Add sparse tensor product as an option in `e3nn.tensor_product` and related functions. It sparsify the clebsch gordan coefficients. It has more inpact when `fused=True`. It is disabled by default because no improvement was observed in the benchmarks.
- Add `log_coordinates` along the other parameterizations of SO(3). `e3nn.log_coordinates_to_matrix`, `e3nn.rand_log_coordinates`, etc.

### Fixed
- set dtype for all `jnp.zeros(..., dtype)` calls in the codebase
- set dtype for all `jnp.ones(..., dtype)` calls in the codebase

### Removed
- **[BREAKING]** `e3nn.full_tensor_product` in favor of `e3nn.tensor_product`
- **[BREAKING]** `e3nn.FunctionalTensorSquare` in favor of `e3nn.tensor_square`
- **[BREAKING]** `e3nn.TensorSquare` in favor of `e3nn.tensor_square`
- **[BREAKING]** `e3nn.IrrepsArray.cat` in favor of `e3nn.concatenate`
- **[BREAKING]** `e3nn.IrrepsArray.randn` in favor of `e3nn.normal`
- **[BREAKING]** `e3nn.Irreps.randn` in favor of `e3nn.normal`
- **[BREAKING]** `e3nn.Irreps.transform_by_*` in favor of `e3nn.IrrepsArray.transform_by_*`

## Changed
- moves `BatchNorm` and `Dropout` to `e3nn.haiku` submodule, will remove them from the main module in the future.
- move `e3nn.haiku.FullyConnectedTensorProduct` in `haiku` submodule. Undeprecate it because it's faster than `e3nn.tensor_product` followed by `e3nn.Linear`. This is because `opteinsum` optimizes the contraction of the two operations.

## [0.13.1] - 2022-12-14
### Added
- `e3nn.scatter_sum` to replace `e3nn.index_add`. `e3nn.index_add` is deprecated.
- add `flax` and `haiku` submodules. Plan to migrate all modules to `flax` and `haiku` in the future.
- Implement `e3nn.flax.Linear` and move `e3nn.Linear` in `e3nn.haiku.Linear`.

## [0.13.0] - 2022-12-07
### Changed
- **[BREAKING]** `3 * e3nn.Irreps("0e + 1o")` now returns `3x0e + 3x1o` instead of `1x0e + 1x1o + 1x0e + 1x1o + 1x0e + 1x1o`
- **[BREAKING]** in Linear, renamed `num_weights` to `num_indexed_weights` because it was confusing.

### Added
- `e3nn.Irreps("3x0e + 6x1o") // 3` returns `1x0e + 2x1o`

### Fixed
- `s2grid` is now jitable

## [0.12.0] - 2022-11-16
### Added
- `e3nn.Irreps.regroup` and `e3nn.IrrepsArray.regroup` to regroup irreps. Equivalent to `sort` followed by `simplify`.
- add `regroup_output` parameter to `e3nn.tensor_product` and `e3nn.tensor_square` to regroup the output irreps.

### Changed
- `e3nn.IrrepsArray.convert` is now private (`e3nn.IrrepsArray._convert`) because it's recommended to other methods instead.
- **breaking change** use `input.regroup()` in `e3nn.Linear` which can change the structure of the parameters dictionary.
- **breaking change** `regroup_output` is `True` by default in `e3nn.tensor_product` and `e3nn.tensor_square`.
- To facilitate debugging, if not `key` is provided to `e3nn.normal` it will use the hash of the irreps.
- **breaking change** changed normalization of `e3nn.tensor_square` in the case of `normalized_input=True`

### Removed
- Deprecate `e3nn.TensorSquare`

## [0.11.1] - 2022-11-13
### Added
- `e3nn.Linear` now supports integer "weights" inputs.
- `e3nn.Linear` now supports `name` argument.
- Add `.dtype` to `IrrepsArray` to get the dtype of the underlying array.

### Changed
- `e3nn.MultiLayerPerceptron` names its layers `linear_0`, `linear_1`, etc.

## [0.11.0] - 2022-11-08
### Added
- s2grid: `e3nn.from_s2grid` and `e3nn.to_s2grid` thanks to @songk42 for the contribution
- argument `max_order: int` to function `reduced_tensor_product_basis` to be able to limit the polynomial order of the basis
- `MultiLayerPerceptron` accepts `IrrepsArray` as input and output
- `e3nn.Linear` accepts optional weights as arguments that will be internally mixed with the free parameters. Very usefyul to implement the depthwise convolution

### Changed
- **breaking change** `e3nn.normal` has a new argument to get normalized vectors.
- **breaking change** `e3nn.tensor_square` now distinguishes between `normalization=norm` and `normalized_input=True`.

## [0.10.1] - 2022-10-24
### Added
- `e3nn.SymmetricTensorProduct` operation: a parameterized version of `x + x^2 + x^3 + ...`.
- `e3nn.soft_envelope` a smooth `C^inf` envelope radial function.
- `e3nn.tensor_square`

## [0.10.0] - 2022-10-05
### Added
- `Irrep.generators` and `Irreps.generators` functions to get the generators of the representations.
- `e3nn.bessel` function
- `slice_by_mul`, `slice_by_dim` and `slice_by_chunk` functions to `Irreps` and `IrrepsArray`

### Changed
- **breaking change** `e3nn.soft_one_hot_linspace` does not support `bessel` anymore. Use `e3nn.bessel` instead.
- `e3nn.gate` is now more flexible of the input format, see examples in the docstring.

### Removed
- **breaking change** `IrrepsArray.split`

## [0.9.2] - 2022-09-29
### Fixed
- fix `IrrepsArray.zeros().at[...].add`

## [0.9.1] - 2022-09-27
### Added
- `e3nn.reduced_symmetric_tensor_product_basis(irreps: Irreps, order: int)`
- `e3nn.IrrepsArray.filtered(keep: List[Irrep])`
- `e3nn.reduced_tensor_product_basis(formula_or_irreps_list: Union[str, List[e3nn.Irreps]], ...)`
- `IrrepsArray.at[i].set(v)` and `IrrepsArray.at[i].add(v)`
- add `Irreps.is_scalar`

## [0.9.0] - 2022-09-04
### Added
- Simple irreps indexing of `IrrepsArray`: like `x[..., "10x0e"]` but not `x[..., "0e + 1e"]`
- `e3nn.concatenate, e3nn.mean, e3nn.sum`
- `e3nn.norm` for `IrrepsArray`
- `e3nn.tensor_product`
- `e3nn.normal`
- Better support of `+ - * /` operators for `IrrepsArray`
- Add **new operator** `e3nn.grad`: it takes an `IrrepsArray -> IrrepsArray` function and returns a `IrrepsArray -> IrrepsArray` function
- Add support of operator `IrrepsArray ** scalar`
- Add support of `x[..., 3:6]` for `IrrepsArray`
- Add `e3nn.reduced_tensor_product_basis`
- Add `e3nn.stack`

### Removed
- `IrrepsArray.cat` is now deprecated and replaced by `e3nn.concatenate`
- `e3nn.full_tensor_product` is now deprecated and replaced by `e3nn.tensor_product`
- `e3nn.FullyConnectedTensorProduct` is now deprecated in favor of `e3nn.tensor_product` and `e3nn.Linear`
- **breaking change** remove `IrrepsArray.from_any`
- **breaking change** remove option `optimize_einsums`, (it is now always `True`)

### Changed
- **breaking change** rewrite the `equivariance_error` and `assert_equivariant` functions

## [0.8.0] - 2022-08-11
### Changed
- **breaking change** change the ordering of `Irrep`. Now it matches with `Irrep.iterator`.
- **breaking change** `Irrep("1e") == "1e"` and `Irreps("1e + 2e") == "1e + 2e"` are now `True`.
- **breaking change** `Linear` simplify the `irreps_out` which might cause reshape of the parameters.
- `index_add` supports `IrrepArray`

### Added
- broadcast for `Linear`
- argument `channel_out` to `Linear` for convenience
- `Irreps` can be created from a `MulIrrep`
- `"0e" + Irreps("1e")` is now supported
- `"0e" + Irrep("1e")` is now supported
- `map_back` argument to `index_add`
- `IrrepsArray.split(list of irreps)`
- `poly_envelope` function

## [0.7.0] - 2022-08-03
### Changed
- **breaking change** rename `IrrepsData` into `IrrepsArray`
- **breaking change** `IrrepsArray.shape` is now equal to `contiguous.shape` (instead of `contiguous.shape[:-1]`)
- **breaking change** `IrrepsArray * array` requires `array.shape[-1]` to be 1 or `array` to be a scalar
- **breaking change** `IrrepsArray.contiguous` is renamed in `IrrepsArray.array`
- **breaking change** `IrrepsArray.new` is renamed in `IrrepsArray.from_any`
- `spherical_harmonics` normalization is now set to `component` like everything else.

### Removed
- **breaking change** `IrrepsArray.from_contiguous` is removed. Use `IrrepsArray(irreps, array)` instead.

### Added
- add `e3nn.config` to set global defaults parameters
- `__getindex__` to `IrrepsData`
- `gradient_normalization` argument that can be `element` or `path`
- `path_normalization` can be a number between 0 and 1
- add nearest interpolation for `zoom`, default is linear
- implement `custom_jvp` for spherical harmonics

## [0.6.3] - 2022-06-29
### Added
- Docker image

## [0.6.2] - 2022-06-29
### Added
- add the `sh` function that does not use `IrrepsData` as input/output
- `legendre` algorithm to compute spherical harmonics
- add flag `algorithm` to specify the algorithm to use for computing spherical harmonics, use `legendre` for large L.
- `experimental.voxel_convolution`: add optional dynamic steps (not static for jit)

## [0.6.1] - 2022-06-09
### Changed
- fix a bug in `experimental.voxel_convolution` constructor

## [0.6.0] - 2022-06-09
### Added
- Function `matrix` to `FunctionalLinear`
- `experimental.voxel_convolution`: `padding` and add self-connection into the convolution kernel
- `experimental.voxel_pooling`: add `output_size` argument to the `zoom` function
- `IrrepsData`: `list` attribute is now lazily initialized
- `experimental.voxel_convolution`: add possibility to have different radial functions depenfing on the spherical harmonic degree

### Changed
- Behavior of `eps` in `BatchNorm`. Now `input / sqrt((1 - eps) * norm^2 + eps)` instead of `input / sqrt(norm^2 + eps)`
- Optimized `spherical_harmonics` by decomposing the order in powers of 2. It is supposed to improve stability because less operations are performed for high orders. It improves the performance when computing a single order.
- Optimized `spherical_harmonics` by using dense matrix multiplication instead of sparse matrix multiplication.

## [0.5.0] - 2022-05-24
### Added
- add `loop` argument to `radius_graph`

### Changed
- use `dataclasses.dataclass` instead of custom `dataclass`
- Get Clebsch-Gordan coefficients from qutip and a change of basis
- Add `start_zero` and `end_zero` arguments to function `soft_one_hot_linspace`

## [0.4.3] - 2022-03-26
### Added
- `IrrepsData` can be given as argument of `spherical_harmonics`
- added broadcasting of `IrrepsData`, `elementwise_tensor_product`, `FullyConnectedTensorProduct`, `full_tensor_product`

### Changed
- `BatchNorm` supports None
- `BatchNorm` supports change default value of `eps` from `1e-5` to `1e-4`
- `gate` change default odd activation to (1 - exp(x^2)) * x

## [0.4.2] - 2022-03-23
### Changed
- `gate` list of activations argument is now optional
- `experimental.transformer.Transformer` simplified interface using `IrrepsData` and swap two arguments order

## [0.4.1] - 2022-03-21
### Added
- `IrrepsData.repeat_irreps_by_last_axis`
- `IrrepsData.repeat_mul_by_last_axis`
- `IrrepsData.factor_mul_to_last_axis`
- add `axis` argument to `IrrepsData.cat`
- `IrrepsData.remove_nones`
- `IrrepsData.ones`

### Changed
- `experimental.point_convolution.Convolution` simplified interface using `IrrepsData`

## [0.4.0] - 2022-03-19

### Added
- Changelog
