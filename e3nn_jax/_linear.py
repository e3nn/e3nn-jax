import math
from functools import partial
from math import prod
from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp
import opt_einsum as oe

from e3nn_jax import Irreps

from ._tensor_product import _as_list, _sum_tensors


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple


def linear(
    irreps_in: Any,
    irreps_out: Any,
    instructions: Optional[List[Tuple[int, int]]] = None,
    biases: Optional[Union[List[bool], bool]] = None,
    optimize_einsums: bool = True,
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
            path_shape=(irreps_in[i_in].mul, irreps_out[i_out].mul)
        )
        for i_in, i_out in instructions
    ]
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]

    if biases is None:
        biases = len(irreps_out) * (False,)
    if isinstance(biases, bool):
        biases = [biases and ir.is_scalar() for _, ir in irreps_out]

    assert len(biases) == len(irreps_out)
    assert all(ir.is_scalar() or (not b) for b, (_, ir) in zip(biases, irreps_out))

    einsum = partial(oe.contract, backend='jax') if optimize_einsums else jnp.einsum
    weight_numel = sum(prod(ins.path_shape) for ins in instructions)
    bias_numel = sum(mul_ir.dim for bias, mul_ir in zip(biases, irreps_out) if bias)

    # = Function definitions =
    def f(ws, bs, x):
        """
        ws: [w_index]
        bs: [b_index]
        x: [lm]
        """
        out_bias_list = []
        bias_index = 0
        for bias, mul_ir_out in zip(biases, irreps_out):
            if bias:
                if sum(biases) == 1:
                    b = bs
                else:
                    b = bs[bias_index:bias_index + mul_ir_out.dim]
                    bias_index += mul_ir_out.dim
                out_bias_list += [[b]]
            else:
                out_bias_list += [[]]

        # = extract individual input irreps =
        x_list = _as_list(irreps_in, x)  # [[mul, ir.dim], ...]

        flat_weight_index = 0

        out_list = []

        for ins in instructions:
            mul_ir_in = irreps_in[ins.i_in]
            mul_ir_out = irreps_out[ins.i_out]

            path_nweight = prod(ins.path_shape)
            if len(instructions) == 1:
                w = ws
            else:
                w = ws[flat_weight_index:flat_weight_index + path_nweight]
            w = w.reshape(ins.path_shape)
            flat_weight_index += path_nweight

            out_list += [
                einsum("uw,ui->wi", w, x_list[ins.i_in]) / math.sqrt(
                    mul_ir_in.mul * sum(
                        1 if other_ins.i_out == ins.i_out else 0
                        for other_ins in instructions
                    )
                )
            ]

        return jnp.concatenate([
            _sum_tensors(
                [out for ins, out in zip(instructions, out_list) if ins.i_out == i_out] + out_bias_list[i_out],
                shape=(mul_ir_out.dim,),
            )
            for i_out, mul_ir_out in enumerate(irreps_out)
            if mul_ir_out.mul > 0
        ])

    return instructions, weight_numel, bias_numel, f


class Linear(flax.linen.Module):
    irreps_in: Irreps
    irreps_out: Irreps
    instructions: Optional[Tuple[int, int]] = None
    biases: Union[bool, List[bool]] = False
    weight_init: Callable = jax.random.normal
    bias_init: Callable = flax.linen.initializers.zeros

    @flax.linen.compact
    def __call__(self, x):
        _, nw, nb, f = linear(self.irreps_in, self.irreps_out, self.instructions, biases=self.biases)
        w = self.param('weight', self.weight_init, (nw,))
        b = self.param('bias', self.bias_init, (nb,))
        return f(w, b, x)
