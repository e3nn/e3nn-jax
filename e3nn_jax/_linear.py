from typing import Any, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from e3nn_jax import Irreps, IrrepsData

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
        path_normalization: str = 'element',
    ):
        assert path_normalization in ['element', 'path']

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

        def alpha(ins):
            x = sum(
                irreps_in[i.i_in if path_normalization == 'element' else ins.i_in].mul
                for i in instructions
                if i.i_out == ins.i_out
            )
            return 1.0 if x == 0 else x

        instructions = [
            Instruction(
                i_in=ins.i_in,
                i_out=ins.i_out,
                path_shape=ins.path_shape,
                path_weight=alpha(ins)**(-0.5)
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

    def __call__(self, ws, x):
        """
        ws: List of arrays
        x: input
        """
        x = IrrepsData.new(self.irreps_in, x)
        assert all(x is None or x.ndim == 2 for x in x.list), "the input of Linear must be a list of 2D arrays"

        out_list = [
            ins.path_weight * w
            if ins.i_in == -1 else
            (
                None
                if x.list[ins.i_in] is None else
                ins.path_weight * jnp.einsum("uw,ui->wi", w, x.list[ins.i_in])
            )
            for ins, w in zip(self.instructions, ws)
        ]

        out = [
            _sum_tensors(
                [out for ins, out in zip(self.instructions, out_list) if ins.i_out == i_out],
                shape=(mul_ir_out.mul, mul_ir_out.ir.dim,),
                empty_return_none=True,
            )
            for i_out, mul_ir_out in enumerate(self.irreps_out)
        ]
        return IrrepsData.from_list(self.irreps_out, out)
