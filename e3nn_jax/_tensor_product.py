from math import sqrt
from typing import List, Optional, Any

from e3nn import o3
from e3nn.util import prod

from ._instruction import Instruction

import opt_einsum as oe
import jax.numpy as jnp


def _sum_tensors(xs, shape):
    if len(xs) > 0:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out
    return jnp.zeros(shape)


def tensor_product(
    irreps_in1: Any,
    irreps_in2: Any,
    irreps_out: Any,
    instructions: List[Any],
    in1_var: Optional[List[float]] = None,
    in2_var: Optional[List[float]] = None,
    out_var: Optional[List[float]] = None,
    normalization: str = 'component',
    specialized_code: bool = True,
    optimize_einsums: bool = True,
):
    irreps_in1 = o3.Irreps(irreps_in1)
    irreps_in2 = o3.Irreps(irreps_in2)
    irreps_out = o3.Irreps(irreps_out)

    instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]
    instructions = [
        Instruction(
            i_in1, i_in2, i_out, connection_mode, has_weight, path_weight,
            {
                'uvw': (irreps_in1[i_in1].mul, irreps_in2[i_in2].mul, irreps_out[i_out].mul),
                'uvu': (irreps_in1[i_in1].mul, irreps_in2[i_in2].mul),
                'uvv': (irreps_in1[i_in1].mul, irreps_in2[i_in2].mul),
                'uuw': (irreps_in1[i_in1].mul, irreps_out[i_out].mul),
                'uuu': (irreps_in1[i_in1].mul,),
                'uvuv': (irreps_in1[i_in1].mul, irreps_in2[i_in2].mul),
            }[connection_mode],
        )
        for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in instructions
    ]

    einsum = oe.contract if optimize_einsums else jnp.einsum
    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)

    if in1_var is None:
        in1_var = [1.0 for _ in range(len(irreps_in1))]

    if in2_var is None:
        in2_var = [1.0 for _ in range(len(irreps_in2))]

    if out_var is None:
        out_var = [1.0 for _ in range(len(irreps_out))]

    def f(weights, input1, input2):
        # = Short-circut for zero dimensional =
        if irreps_in1.dim == 0 or irreps_in2.dim == 0 or irreps_out.dim == 0:
            return jnp.zeros((irreps_out.dim,))

        # = wigners cache =
        w3j_dict = dict()

        # = extract individual input irreps =
        x1_list = [
            input1[i].reshape(mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(irreps_in1.slices(), irreps_in1)
        ]

        x2_list = [
            input2[i].reshape(mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(irreps_in2.slices(), irreps_in2)
        ]

        xx_dict = dict()

        flat_weight_index = 0

        out_list = []

        for ins in instructions:
            mul_ir_in1 = irreps_in1[ins.i_in1]
            mul_ir_in2 = irreps_in2[ins.i_in2]
            mul_ir_out = irreps_out[ins.i_out]

            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l

            if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
                continue

            alpha = ins.path_weight * out_var[ins.i_out] / sum(in1_var[i.i_in1] * in2_var[i.i_in2] for i in instructions if i.i_out == ins.i_out)

            x1 = x1_list[ins.i_in1]
            x2 = x2_list[ins.i_in2]

            assert ins.connection_mode in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv']

            alpha = sqrt(alpha / {
                'uvw': (mul_ir_in1.mul * mul_ir_in2.mul),
                'uvu': mul_ir_in2.mul,
                'uvv': mul_ir_in1.mul,
                'uuw': mul_ir_in1.mul,
                'uuu': 1,
                'uvuv': 1,
            }[ins.connection_mode])

            if ins.has_weight:
                # Extract the weight from the flattened weight tensor
                w = weights[flat_weight_index:flat_weight_index + prod(ins.path_shape)].reshape(ins.path_shape)
                flat_weight_index += prod(ins.path_shape)

            # We didn't make this instruction specialized, so do the general case
            key = (ins.i_in1, ins.i_in2, ins.connection_mode[:2])
            if key not in xx_dict:
                if ins.connection_mode[:2] == 'uv':
                    xx_dict[key] = einsum('ui,vj->uvij', x1, x2)
                if ins.connection_mode[:2] == 'uu':
                    xx_dict[key] = einsum('ui,uj->uij', x1, x2)
            xx = xx_dict[key]

            key = (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
            if key not in w3j_dict:
                wig = o3.wigner_3j(*key).numpy()

                if normalization == 'component':
                    wig *= mul_ir_out.ir.dim**0.5
                if normalization == 'norm':
                    wig *= (mul_ir_in1.ir.dim * mul_ir_in2.ir.dim)**0.5

                w3j_dict[key] = jnp.array(wig)
            w3j = w3j_dict[key]

            exp = {'component': 1, 'norm': -1}[normalization]

            if ins.connection_mode == 'uvw':
                assert ins.has_weight
                if specialized_code and key == (0, 0, 0):
                    ein_out = einsum("uvw,u,v->w", w, x1.reshape(mul_ir_in1.dim), x2.reshape(mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    ein_out = einsum("uvw,u,vj->wj", w, x1.reshape(mul_ir_in1.dim), x2)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    ein_out = einsum("uvw,ui,v->wi", w, x1, x2.reshape(mul_ir_in2.dim))
                elif specialized_code and mul_ir_out.ir.l == 0:
                    ein_out = einsum("uvw,ui,vi->w", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)**exp
                else:
                    ein_out = einsum("uvw,ijk,uvij->wk", w, w3j, xx)
            if ins.connection_mode == 'uvu':
                assert mul_ir_in1.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and key == (0, 0, 0):
                        ein_out = einsum("uv,u,v->u", w, x1.reshape(mul_ir_in1.dim), x2.reshape(mul_ir_in2.dim))
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        ein_out = einsum("uv,u,vj->uj", w, x1.reshape(mul_ir_in1.dim), x2)
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        ein_out = einsum("uv,ui,v->ui", w, x1, x2.reshape(mul_ir_in2.dim))
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        ein_out = einsum("uv,ui,vi->u", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)**exp
                    else:
                        ein_out = einsum("uv,ijk,uvij->uk", w, w3j, xx)
                else:
                    # not so useful operation because v is summed
                    ein_out = einsum("ijk,uvij->uk", w3j, xx)
            if ins.connection_mode == 'uvv':
                assert mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and key == (0, 0, 0):
                        ein_out = einsum("uv,u,v->v", w, x1.reshape(mul_ir_in1.dim), x2.reshape(mul_ir_in2.dim))
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        ein_out = einsum("uv,u,vj->vj", w, x1.reshape(mul_ir_in1.dim), x2)
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        ein_out = einsum("uv,ui,v->vi", w, x1, x2.reshape(mul_ir_in2.dim))
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        ein_out = einsum("uv,ui,vi->v", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)**exp
                    else:
                        ein_out = einsum("uv,ijk,uvij->vk", w, w3j, xx)
                else:
                    # not so useful operation because u is summed
                    ein_out = einsum("ijk,uvij->vk", w3j, xx)
            if ins.connection_mode == 'uuw':
                assert mul_ir_in1.mul == mul_ir_in2.mul
                if ins.has_weight:
                    # TODO implement specialized code
                    ein_out = einsum("uw,ijk,uij->wk", w, w3j, xx)
                else:
                    # equivalent to tp(x, y, 'uuu').sum('u')
                    assert mul_ir_out.mul == 1
                    ein_out = einsum("ijk,uij->k", w3j, xx)
            if ins.connection_mode == 'uuu':
                assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    # TODO implement specialized code
                    ein_out = einsum("u,ijk,uij->uk", w, w3j, xx)
                else:
                    # TODO implement specialized code
                    ein_out = einsum("ijk,uij->uk", w3j, xx)
            if ins.connection_mode == 'uvuv':
                assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    # TODO implement specialized code
                    ein_out = einsum("uv,ijk,uvij->uvk", w, w3j, xx)
                else:
                    # TODO implement specialized code
                    ein_out = einsum("ijk,uvij->uvk", w3j, xx)

            ein_out = alpha * ein_out

            out_list += [ein_out.reshape(mul_ir_out.dim)]

        # = Return the result =
        out_out = jnp.concatenate([
            _sum_tensors(
                [out for ins, out in zip(instructions, out_list) if ins.i_out == i_out],
                shape=(mul_ir_out.dim),
            )
            for i_out, mul_ir_out in enumerate(irreps_out)
        ])

        return out_out.reshape(irreps_out.dim)
    return instructions, weight_numel, f


def fully_connected_tensor_product(
    irreps_in1: Any,
    irreps_in2: Any,
    irreps_out: Any,
    **kwargs
):
    irreps_in1 = o3.Irreps(irreps_in1)
    irreps_in2 = o3.Irreps(irreps_in2)
    irreps_out = o3.Irreps(irreps_out)

    instr = [
        (i_1, i_2, i_out, 'uvw', True, 1.0)
        for i_1, (_, ir_1) in enumerate(irreps_in1)
        for i_2, (_, ir_2) in enumerate(irreps_in2)
        for i_out, (_, ir_out) in enumerate(irreps_out)
        if ir_out in ir_1 * ir_2
    ]
    return tensor_product(irreps_in1, irreps_in2, irreps_out, instr, **kwargs)
