# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `legendre` algorithm to compute spherical harmonics
- add flag `algorithm` to specify the algorithm to use for computing spherical harmonics

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
