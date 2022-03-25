# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `IrrepsData` can be given as argument of `spherical_harmonics`

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
