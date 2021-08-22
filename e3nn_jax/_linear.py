from typing import Any, List, NamedTuple, Optional, Tuple, Union

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
    biases: List[int]

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

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.instructions = instructions

    def __call__(self, ws, x):
        """
        ws: List of arrays
        x: array
        """
        x_list = self.irreps_in.as_list(x)  # [[mul, ir.dim], ...]

        out_list = [
            ins.path_weight * w
            if ins.i_in == -1 else
            ins.path_weight * jnp.einsum("uw,ui->wi", w, x_list[ins.i_in])
            for ins, w in zip(self.instructions, ws)
        ]

        return jnp.concatenate([
            _sum_tensors(
                [out for ins, out in zip(self.instructions, out_list) if ins.i_out == i_out],
                shape=(mul_ir_out.dim,),
            )
            for i_out, mul_ir_out in enumerate(self.irreps_out)
            if mul_ir_out.mul > 0
        ])


from typing import Callable

import flax
import jax


class FlaxLinear(flax.linen.Module):
    irreps_in: Irreps
    irreps_out: Irreps
    instructions: Optional[Tuple[int, int]] = None
    biases: Union[bool, List[bool]] = False
    weight_init: Callable = jax.random.normal
    bias_init: Callable = flax.linen.initializers.zeros

    @flax.linen.compact
    def __call__(self, x):
        lin = Linear(self.irreps_in, self.irreps_out, self.instructions, biases=self.biases)
        w = [
            self.param(f'bias {ins.i_out}', self.bias_init, ins.path_shape)
            if ins.i_in == -1 else
            self.param(f'weight {ins.i_in} -> {ins.i_out}', self.weight_init, ins.path_shape)
            for ins in lin.instructions
        ]
        return lin(w, x)
