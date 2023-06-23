from typing import List, Optional

import e3nn_jax as e3nn
from e3nn_jax.legacy import FunctionalTensorProduct


def FunctionalFullyConnectedTensorProduct(
    irreps_in1: e3nn.Irreps,
    irreps_in2: e3nn.Irreps,
    irreps_out: e3nn.Irreps,
    in1_var: Optional[List[float]] = None,
    in2_var: Optional[List[float]] = None,
    out_var: Optional[List[float]] = None,
    irrep_normalization: str = None,
    path_normalization: str = None,
    gradient_normalization: str = None,
):
    irreps_in1 = e3nn.Irreps(irreps_in1)
    irreps_in2 = e3nn.Irreps(irreps_in2)
    irreps_out = e3nn.Irreps(irreps_out)

    instructions = [
        (i_1, i_2, i_out, "uvw", True)
        for i_1, (_, ir_1) in enumerate(irreps_in1)
        for i_2, (_, ir_2) in enumerate(irreps_in2)
        for i_out, (_, ir_out) in enumerate(irreps_out)
        if ir_out in ir_1 * ir_2
    ]
    return FunctionalTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        in1_var,
        in2_var,
        out_var,
        irrep_normalization,
        path_normalization,
        gradient_normalization,
    )
