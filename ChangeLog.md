# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
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

### Removed
- `IrrepsArray.cat` is now deprecated and replaced by `e3nn.concatenate`
- `e3nn.full_tensor_product` is now deprecated and replaced by `e3nn.tensor_product`
- `e3nn.FullyConnectedTensorProduct` is now deprecated in favor of `e3nn.tensor_product` and `e3nn.Linear`
- **breaking change** remove `IrrepsArray.from_any`

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

## Changed
- use `dataclasses.dataclass` instead of custom `dataclass`
- Get Clebsch-Gordan coefficients from qutip and a change of basis
- Add `start_zero` and `end_zero` arguments to function `soft_one_hot_linspace`

## [0.4.3] - 2022-03-26
### Added
- `IrrepsData` can be given as argument of `spherical_harmonics`
- added broadcasting of `IrrepsData`, `elementwise_tensor_product`, `FullyConnectedTensorProduct`, `full_tensor_product`

## Changed
- `BatchNorm` supports None
- `BatchNorm` supports change default value of `eps` from `1e-5` to `1e-4`
- `gate` change default odd activation to (1 - exp(x^2)) * x

## [0.4.2] - 2022-03-23
## Changed
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

## Changed
- `experimental.point_convolution.Convolution` simplified interface using `IrrepsData`

## [0.4.0] - 2022-03-19

### Added
- Changelog
