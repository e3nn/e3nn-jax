__version__ = "0.9.0"

from ._config import config
from ._rotation import (
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
from ._su2 import su2_clebsch_gordan, su2_generators
from ._so3 import clebsch_gordan, wigner_D, generators
from ._instruction import Instruction
from ._irreps import Irrep, MulIrrep, Irreps
from ._irreps_array import IrrepsArray, concatenate, stack, mean, norm, normal
from ._irreps_array import sum_ as sum
from ._spherical_harmonics import spherical_harmonics, sh, legendre
from ._soft_one_hot_linspace import sus, soft_one_hot_linspace
from ._linear import FunctionalLinear, Linear
from ._core_tensor_product import FunctionalTensorProduct
from ._tensor_products import (
    FunctionalFullyConnectedTensorProduct,
    FullyConnectedTensorProduct,
    full_tensor_product,
    tensor_product,
    elementwise_tensor_product,
    FunctionalTensorSquare,
    TensorSquare,
)
from ._grad import grad
from ._activation import scalar_activation, normalize_function
from ._gate import gate
from ._batchnorm import BatchNorm
from ._dropout import Dropout
from ._nn import MultiLayerPerceptron
from ._graph_util import index_add, radius_graph
from ._poly_envelope import poly_envelope
from ._reduced_tensor_product import reduced_tensor_product_basis

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
    "FunctionalLinear",  # not in docs
    "Linear",
    "FunctionalTensorProduct",
    "FunctionalFullyConnectedTensorProduct",  # deprecated
    "FullyConnectedTensorProduct",  # deprecated
    "full_tensor_product",  # deprecated
    "tensor_product",
    "elementwise_tensor_product",
    "FunctionalTensorSquare",  # not in docs
    "TensorSquare",
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
    "reduced_tensor_product_basis",
]
