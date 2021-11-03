from typing import Any, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from e3nn_jax import Irreps

from ._tensor_product import _sum_tensors


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple
    path_weight: float


class Linear:
    irreps_in: Irreps
    irreps_out: Irreps
    instructions: List[Instruction]
    output_mask: jnp.ndarray

    def __init__(
        self,
        irreps_in: Any,
        irreps_out: Any,
        instructions: Optional[List[Tuple[int, int]]] = None,
        biases: Optional[Union[List[bool], bool]] = None,
    ):
        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)

        if instructions is None:
            # By default, make all possible connections
            instructions = [
                (i_in, i_out)
                for i_in, (_, ir_in) in enumerate(irreps_in)
                for i_out, (_, ir_out) in enumerate(irreps_out)
                if ir_in == ir_out
            ]

        instructions = [
            Instruction(
                i_in=i_in,
                i_out=i_out,
                path_shape=(irreps_in[i_in].mul, irreps_out[i_out].mul),
                path_weight=1,
            )
            for i_in, i_out in instructions
        ]
        instructions = [ins for ins in instructions if 0 not in ins.path_shape]

        instructions = [
            Instruction(
                i_in=ins.i_in,
                i_out=ins.i_out,
                path_shape=ins.path_shape,
                path_weight=(
                    irreps_in[ins.i_in].mul * sum(
                        1 if other_ins.i_out == ins.i_out else 0
                        for other_ins in instructions
                    )
                )**(-0.5)
            )
            for ins in instructions
        ]

        if biases is None:
            biases = len(irreps_out) * (False,)
        if isinstance(biases, bool):
            biases = [biases and ir.is_scalar() for _, ir in irreps_out]

        assert len(biases) == len(irreps_out)
        assert all(ir.is_scalar() or (not b) for b, (_, ir) in zip(biases, irreps_out))

        instructions += [
            Instruction(
                i_in=-1,
                i_out=i_out,
                path_shape=(mul_ir.dim,),
                path_weight=1.0
            )
            for i_out, (bias, mul_ir) in enumerate(zip(biases, irreps_out)) if bias
        ]

        with jax.core.eval_context():
            if irreps_out.dim > 0:
                output_mask = jnp.concatenate([
                    jnp.ones(mul_ir.dim)
                    if any(
                        (ins.i_out == i_out) and (0 not in ins.path_shape)
                        for ins in instructions
                    )
                    else jnp.zeros(mul_ir.dim)
                    for i_out, mul_ir in enumerate(irreps_out)
                ])
            else:
                output_mask = jnp.ones(0)

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.instructions = instructions
        self.output_mask = output_mask

    def __call__(self, ws, x, output_list=False):
        """
        ws: List of arrays
        x: array
        """
        x_list = self.irreps_in.to_list(x)  # [[mul, ir.dim], ...]

        out_list = [
            ins.path_weight * w
            if ins.i_in == -1 else
            (
                None
                if x_list[ins.i_in] is None else
                ins.path_weight * jnp.einsum("uw,ui->wi", w, x_list[ins.i_in])
            )
            for ins, w in zip(self.instructions, ws)
        ]

        out = [
            _sum_tensors(
                [out for ins, out in zip(self.instructions, out_list) if ins.i_out == i_out],
                shape=(mul_ir_out.mul, mul_ir_out.ir.dim,),
                empty_return_none=output_list,
            )
            for i_out, mul_ir_out in enumerate(self.irreps_out)
            if mul_ir_out.mul > 0
        ]
        if output_list:
            return out
        return jnp.concatenate([x.flatten() for x in out])
