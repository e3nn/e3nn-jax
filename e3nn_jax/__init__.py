__version__ = "0.13.0"

from e3nn_jax._src.config import config
from e3nn_jax._src.rotation import (
    rand_matrix,
    identity_angles,
    rand_angles,
    compose_angles,
    inverse_angles,
    identity_quaternion,
    rand_quaternion,
    compose_quaternion,
    inverse_quaternion,
    rand_axis_angle,
    compose_axis_angle,
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
    angles_to_xyz,
    xyz_to_angles,
)
from e3nn_jax._src.su2 import su2_clebsch_gordan, su2_generators
from e3nn_jax._src.so3 import clebsch_gordan, wigner_D, generators
from e3nn_jax._src.instruction import Instruction
from e3nn_jax._src.irreps import Irrep, MulIrrep, Irreps
from e3nn_jax._src.irreps_array import IrrepsArray, concatenate, stack, mean, norm, normal
from e3nn_jax._src.irreps_array import sum_ as sum
from e3nn_jax._src.spherical_harmonics import spherical_harmonics, sh, legendre
from e3nn_jax._src.radial import sus, soft_one_hot_linspace, bessel, poly_envelope, soft_envelope
from e3nn_jax._src.linear import FunctionalLinear, Linear
from e3nn_jax._src.core_tensor_product import FunctionalTensorProduct
from e3nn_jax._src.tensor_products import (
    FunctionalFullyConnectedTensorProduct,
    FullyConnectedTensorProduct,
    full_tensor_product,
    tensor_product,
    elementwise_tensor_product,
    FunctionalTensorSquare,
    TensorSquare,
    tensor_square,
)
from e3nn_jax._src.grad import grad
from e3nn_jax._src.activation import scalar_activation, normalize_function
from e3nn_jax._src.gate import gate
from e3nn_jax._src.batchnorm import BatchNorm
from e3nn_jax._src.dropout import Dropout
from e3nn_jax._src.mlp import MultiLayerPerceptron
from e3nn_jax._src.graph_util import index_add, radius_graph
from e3nn_jax._src.reduced_tensor_product import reduced_tensor_product_basis, reduced_symmetric_tensor_product_basis
from e3nn_jax._src.symmetric_tensor_product import SymmetricTensorProduct
from e3nn_jax._src.s2grid import from_s2grid, to_s2grid


__all__ = [
    "config",  # not in docs
    "rand_matrix",
    "identity_angles",
    "rand_angles",
    "compose_angles",
    "inverse_angles",
    "identity_quaternion",
    "rand_quaternion",
    "compose_quaternion",
    "inverse_quaternion",
    "rand_axis_angle",
    "compose_axis_angle",
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
    "angles_to_xyz",
    "xyz_to_angles",
    "su2_clebsch_gordan",  # not in docs
    "su2_generators",  # not in docs
    "clebsch_gordan",
    "wigner_D",  # TODO could be moved into Irrep
    "generators",  # TODO could be moved into Irrep
    "Instruction",  # not in docs
    "Irrep",
    "MulIrrep",  # not in docs
    "Irreps",
    "IrrepsArray",
    "concatenate",
    "stack",
    "mean",
    "norm",
    "normal",
    "sum",
    "spherical_harmonics",
    "sh",
    "legendre",  # not in docs
    "sus",
    "soft_one_hot_linspace",
    "bessel",
    "FunctionalLinear",  # not in docs
    "Linear",
    "FunctionalTensorProduct",
    "FunctionalFullyConnectedTensorProduct",  # deprecated
    "FullyConnectedTensorProduct",  # deprecated
    "full_tensor_product",  # deprecated
    "tensor_product",
    "elementwise_tensor_product",
    "FunctionalTensorSquare",  # not in docs
    "TensorSquare",  # not in docs
    "tensor_square",
    "grad",
    "scalar_activation",
    "normalize_function",
    "gate",
    "BatchNorm",
    "Dropout",
    "MultiLayerPerceptron",
    "index_add",
    "radius_graph",
    "poly_envelope",
    "soft_envelope",
    "reduced_tensor_product_basis",
    "reduced_symmetric_tensor_product_basis",
    "SymmetricTensorProduct",
    "from_s2grid",
    "to_s2grid",
]
