__version__ = "0.20.6"

from e3nn_jax._src.config import config
from e3nn_jax._src.rotation import (
    rand_matrix,
    rotation_angle_from_matrix,
    identity_angles,
    rand_angles,
    compose_angles,
    inverse_angles,
    rotation_angle_from_angles,
    identity_quaternion,
    rand_quaternion,
    compose_quaternion,
    inverse_quaternion,
    rotation_angle_from_quaternion,
    rand_axis_angle,
    compose_axis_angle,
    rotation_angle_from_axis_angle,
    matrix_x,
    matrix_y,
    matrix_z,
    angles_to_matrix,
    matrix_to_angles,
    angles_to_quaternion,
    matrix_to_quaternion,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    matrix_to_axis_angle,
    angles_to_axis_angle,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    quaternion_to_angles,
    axis_angle_to_angles,
    rand_log_coordinates,
    identity_log_coordinates,
    compose_log_coordinates,
    inverse_log_coordinates,
    rotation_angle_from_log_coordinates,
    log_coordinates_to_matrix,
    matrix_to_log_coordinates,
    log_coordinates_to_quaternion,
    quaternion_to_log_coordinates,
    log_coordinates_to_axis_angle,
    axis_angle_to_log_coordinates,
    log_coordinates_to_angles,
    angles_to_log_coordinates,
    angles_to_xyz,
    xyz_to_angles,
)
from e3nn_jax._src.su2 import su2_clebsch_gordan, su2_generators
from e3nn_jax._src.so3 import clebsch_gordan, generators
from e3nn_jax._src.irreps import Irrep, MulIrrep, Irreps
from e3nn_jax._src.irreps_array import IrrepsArray
from e3nn_jax._src.basic import (
    from_chunks,
    as_irreps_array,
    zeros,
    zeros_like,
    ones,
    ones_like,
    concatenate,
    stack,
    mean,
    norm,
    normal,
    dot,
    cross,
    where,
)
from e3nn_jax._src.basic import sum_ as sum
from e3nn_jax._src.spherical_harmonics import spherical_harmonics, sh, legendre
from e3nn_jax._src.radial import (
    sus,
    soft_one_hot_linspace,
    bessel,
    poly_envelope,
    soft_envelope,
)
from e3nn_jax._src.linear import FunctionalLinear
from e3nn_jax._src.tensor_products import (
    tensor_product,
    elementwise_tensor_product,
    tensor_square,
)
from e3nn_jax._src.grad import grad
from e3nn_jax._src.activation import (
    soft_odd,
    scalar_activation,
    normalize_function,
    norm_activation,
    soft_normalization,  # not in docs
)
from e3nn_jax._src.gate import gate
from e3nn_jax._src.radius_graph import radius_graph
from e3nn_jax._src.scatter import index_add, scatter_sum, scatter_mean, scatter_max
from e3nn_jax._src.reduced_tensor_product import (
    reduced_tensor_product_basis,
    reduced_symmetric_tensor_product_basis,
    reduced_antisymmetric_tensor_product_basis,
)
from e3nn_jax._src.s2grid import (
    s2_irreps,
    to_s2grid,
    to_s2point,
    from_s2grid,
    legendre_transform_to_s2grid,
    legendre_transform_from_s2grid,
    betas_to_spherical_signal,
    m0_values_to_irrepsarray,
    s2_dirac,
    SphericalSignal,
    get_s2fft_grid_resolution,
)
from e3nn_jax._src.tensor_product_with_spherical_harmonics import (
    tensor_product_with_spherical_harmonics,
)
from e3nn_jax._src.utils.vmap import vmap


# make submodules flax and haiku available
from e3nn_jax import flax, haiku, equinox
from e3nn_jax import utils

__all__ = [
    "config",  # not in docs
    "rand_matrix",
    "rotation_angle_from_matrix",
    "identity_angles",
    "rand_angles",
    "compose_angles",
    "inverse_angles",
    "rotation_angle_from_angles",
    "identity_quaternion",
    "rand_quaternion",
    "compose_quaternion",
    "inverse_quaternion",
    "rotation_angle_from_quaternion",
    "rand_axis_angle",
    "compose_axis_angle",
    "rotation_angle_from_axis_angle",
    "matrix_x",
    "matrix_y",
    "matrix_z",
    "angles_to_matrix",
    "matrix_to_angles",
    "angles_to_quaternion",
    "matrix_to_quaternion",
    "axis_angle_to_quaternion",
    "quaternion_to_axis_angle",
    "matrix_to_axis_angle",
    "angles_to_axis_angle",
    "axis_angle_to_matrix",
    "quaternion_to_matrix",
    "quaternion_to_angles",
    "axis_angle_to_angles",
    "rand_log_coordinates",
    "identity_log_coordinates",
    "compose_log_coordinates",
    "inverse_log_coordinates",
    "rotation_angle_from_log_coordinates",
    "log_coordinates_to_matrix",
    "matrix_to_log_coordinates",
    "log_coordinates_to_quaternion",
    "quaternion_to_log_coordinates",
    "log_coordinates_to_axis_angle",
    "axis_angle_to_log_coordinates",
    "log_coordinates_to_angles",
    "angles_to_log_coordinates",
    "angles_to_xyz",
    "xyz_to_angles",
    "su2_clebsch_gordan",  # not in docs
    "su2_generators",  # not in docs
    "clebsch_gordan",
    "generators",  # TODO could be moved into Irrep
    "Irrep",
    "MulIrrep",  # not in docs
    "Irreps",
    "IrrepsArray",
    "from_chunks",
    "as_irreps_array",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "concatenate",
    "stack",
    "mean",
    "norm",
    "normal",
    "dot",
    "cross",
    "where",
    "sum",
    "spherical_harmonics",
    "sh",
    "legendre",  # not in docs
    "sus",
    "soft_one_hot_linspace",
    "bessel",
    "FunctionalLinear",  # not in docs
    "tensor_product",
    "elementwise_tensor_product",
    "tensor_square",
    "grad",
    "soft_odd",
    "scalar_activation",
    "normalize_function",
    "norm_activation",
    "soft_normalization",
    "gate",
    "radius_graph",
    "index_add",
    "scatter_sum",
    "scatter_mean",
    "scatter_max",
    "poly_envelope",
    "soft_envelope",
    "reduced_tensor_product_basis",
    "reduced_symmetric_tensor_product_basis",
    "reduced_antisymmetric_tensor_product_basis",
    "s2_irreps",
    "to_s2grid",
    "to_s2point",
    "from_s2grid",
    "get_s2fft_grid_resolution",
    "legendre_transform_from_s2grid",
    "legendre_transform_to_s2grid",
    "betas_to_spherical_signal",
    "m0_values_to_irrepsarray",
    "s2_dirac",
    "SphericalSignal",
    "tensor_product_with_spherical_harmonics",
    "vmap",
    "flax",
    "haiku",
    "equinox",
    "utils",
]
