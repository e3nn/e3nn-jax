__version__ = "0.3.1"

from ._wigner import wigner_3j, wigner_D
from ._irreps import Irrep, Irreps
from ._spherical_harmonics import spherical_harmonics
from ._soft_one_hot_linspace import sus, soft_one_hot_linspace
from ._linear import linear
from ._tensor_product import tensor_product, fully_connected_tensor_product

__all__ = [
    "wigner_3j", "wigner_D",
    "Irrep", "Irreps",
    "spherical_harmonics",
    "sus", "soft_one_hot_linspace",
    "linear",
    "tensor_product", "fully_connected_tensor_product",
]
