from functools import lru_cache, partial
from math import sqrt, prod
from typing import Any, List, NamedTuple, Optional
import itertools

import jax
import jax.numpy as jnp

from e3nn_jax import Irrep, Irreps, wigner_3j

from ._einsum import einsum as opt_einsum


def _sum_tensors(xs, shape):
    if len(xs) > 0:
        out = xs[0].reshape(shape)
        for x in xs[1:]:
            out = out + x.reshape(shape)
        return out
    return jnp.zeros(shape)


def _flat_concatenate(xs):
    if len(xs) > 0:
        return jnp.concatenate([x.flatten() for x in xs])
    return jnp.zeros((0,))


class Instruction(NamedTuple):
    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    path_shape: tuple


class TensorProduct:
    irreps_in1: Irreps
    irreps_in2: Irreps
    irreps_out: Irreps
    instructions: List[Instruction]
    output_mask: jnp.ndarray

    def __init__(
        self,
        irreps_in1: Any,
        irreps_in2: Any,
        irreps_out: Any,
        instructions: List[Any],
        in1_var: Optional[List[float]] = None,
        in2_var: Optional[List[float]] = None,
        out_var: Optional[List[float]] = None,
        normalization: str = 'component',
    ):
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        del irreps_in1, irreps_in2, irreps_out

        instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]
        instructions = [
            Instruction(
                i_in1, i_in2, i_out, connection_mode, has_weight, path_weight,
                {
                    'uvw': (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul, self.irreps_out[i_out].mul),
                    'uvu': (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    'uvv': (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    'uuw': (self.irreps_in1[i_in1].mul, self.irreps_out[i_out].mul),
                    'uuu': (self.irreps_in1[i_in1].mul,),
                    'uvuv': (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                }[connection_mode],
            )
            for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in instructions
        ]

        if in1_var is None:
            in1_var = [1.0 for _ in range(len(self.irreps_in1))]

        if in2_var is None:
            in2_var = [1.0 for _ in range(len(self.irreps_in2))]

        if out_var is None:
            out_var = [1.0 for _ in range(len(self.irreps_out))]

        normalization_coefficients = []
        for ins in instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]
            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
            assert ins.connection_mode in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv']

            alpha = ins.path_weight * out_var[ins.i_out] / sum(in1_var[i.i_in1] * in2_var[i.i_in2] for i in instructions if i.i_out == ins.i_out)
            alpha = sqrt(alpha / {
                'uvw': (mul_ir_in1.mul * mul_ir_in2.mul),
                'uvu': mul_ir_in2.mul,
                'uvv': mul_ir_in1.mul,
                'uuw': mul_ir_in1.mul,
                'uuu': 1,
                'uvuv': 1,
            }[ins.connection_mode])

            if normalization == 'component':
                alpha *= mul_ir_out.ir.dim**0.5
            if normalization == 'norm':
                alpha *= (mul_ir_in1.ir.dim * mul_ir_in2.ir.dim)**0.5

            normalization_coefficients += [alpha]

        self.instructions = [
            Instruction(ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight, alpha, ins.path_shape)
            for ins, alpha in zip(instructions, normalization_coefficients)
        ]

        if self.irreps_out.dim > 0:
            self.output_mask = jnp.concatenate([
                jnp.ones(mul_ir.dim)
                if any(
                    (ins.i_out == i_out) and (ins.path_weight != 0) and (0 not in ins.path_shape)
                    for ins in self.instructions
                )
                else jnp.zeros(mul_ir.dim)
                for i_out, mul_ir in enumerate(self.irreps_out)
            ])
        else:
            self.output_mask = jnp.ones(0)

    @partial(jax.jit, static_argnums=(0,), static_argnames=('specialized_code', 'optimize_einsums', 'custom_einsum_vjp', 'fuse_all', 'output_list'))
    def left_right(self, weights, input1, input2=None, *, specialized_code=False, optimize_einsums=True, custom_einsum_vjp=False, fuse_all=False, output_list=False):
        if input2 is None:
            weights, input1, input2 = [], weights, input1

        # = Short-circut for zero dimensional =
        if self.irreps_in1.dim == 0 or self.irreps_in2.dim == 0 or self.irreps_out.dim == 0:
            if output_list:
                return [jnp.zeros((mul, ir.dim,)) for mul, ir in self.irreps_out]
            return jnp.zeros((self.irreps_out.dim,))

        if custom_einsum_vjp:
            assert optimize_einsums
            einsum = opt_einsum
        else:
            einsum = partial(jnp.einsum, optimize='optimal' if optimize_einsums else 'greedy')

        if isinstance(input1, list):
            input1_list = self.irreps_in1.as_list(input1)
            input1 = _flat_concatenate(input1_list)
        else:
            input1_list = self.irreps_in1.as_list(input1)

        if isinstance(input2, list):
            input2_list = self.irreps_in2.as_list(input2)
            input2 = _flat_concatenate(input2_list)
        else:
            input2_list = self.irreps_in2.as_list(input2)

        if isinstance(weights, list):
            weights_flat = _flat_concatenate(weights)
        else:
            weights_flat = weights
            weights = []
            i = 0
            for ins in self.instructions:
                if ins.has_weight:
                    n = prod(ins.path_shape)
                    weights.append(weights_flat[i:i+n].reshape(ins.path_shape))
                    i += n

        if fuse_all:
            with jax.core.eval_context():
                # TODO make it for non weights, even simpler
                num_path = weights_flat.size
                big_w3j = jnp.zeros((num_path, self.irreps_in1.dim, self.irreps_in2.dim, self.irreps_out.dim))
                i = 0
                for ins in self.instructions:
                    assert ins.has_weight
                    assert ins.connection_mode == 'uvw'

                    mul_ir_in1 = self.irreps_in1[ins.i_in1]
                    mul_ir_in2 = self.irreps_in2[ins.i_in2]
                    mul_ir_out = self.irreps_out[ins.i_out]
                    m1, m2, mo = mul_ir_in1.mul, mul_ir_in2.mul, mul_ir_out.mul
                    d1, d2, do = mul_ir_in1.ir.dim, mul_ir_in2.ir.dim, mul_ir_out.ir.dim
                    s1 = self.irreps_in1[:ins.i_in1].dim
                    s2 = self.irreps_in2[:ins.i_in2].dim
                    so = self.irreps_out[:ins.i_out].dim

                    assert ins.path_shape == (m1, m2, mo)

                    w3j = wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
                    for u, v, w in itertools.product(range(m1), range(m2), range(mo)):
                        big_w3j = big_w3j.at[i, s1+u*d1: s1+(u+1)*d1, s2+v*d2: s2+(v+1)*d2, so+w*do: so+(w+1)*do].set(ins.path_weight * w3j)
                        i += 1

            out = einsum("p,pijk,i,j->k", weights_flat, big_w3j, input1, input2)
            if output_list:
                return self.irreps_out.as_list(out)
            return out

        @lru_cache(maxsize=None)
        def multiply(in1, in2, mode):
            if mode == 'uv':
                return einsum('ui,vj->uvij', input1_list[in1], input2_list[in2])
            if mode == 'uu':
                return einsum('ui,uj->uij', input1_list[in1], input2_list[in2])

        weight_index = 0

        out_list = []

        for ins in self.instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]

            if ins.has_weight:
                w = weights[weight_index]
                assert w.shape == ins.path_shape
                weight_index += 1

            if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
                continue

            x1 = input1_list[ins.i_in1]
            x2 = input2_list[ins.i_in2]
            xx = multiply(ins.i_in1, ins.i_in2, ins.connection_mode[:2])
            with jax.core.eval_context():
                w3j = wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
                w3j = ins.path_weight * w3j

            if ins.connection_mode == 'uvw':
                assert ins.has_weight
                if specialized_code and (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l) == (0, 0, 0):
                    out = ins.path_weight * einsum("uvw,uv->w", w, xx.reshape(mul_ir_in1.dim, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    out = ins.path_weight * einsum("uvw,u,vj->wj", w, x1.reshape(mul_ir_in1.dim), x2) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    out = ins.path_weight * einsum("uvw,ui,v->wi", w, x1, x2.reshape(mul_ir_in2.dim)) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0:
                    out = ins.path_weight * einsum("uvw,ui,vi->w", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                else:
                    out = einsum("uvw,ijk,uvij->wk", w, w3j, xx)
            if ins.connection_mode == 'uvu':
                assert mul_ir_in1.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l) == (0, 0, 0):
                        out = ins.path_weight * einsum("uv,u,v->u", w, x1.reshape(mul_ir_in1.dim), x2.reshape(mul_ir_in2.dim))
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        out = ins.path_weight * einsum("uv,u,vj->uj", w, x1.reshape(mul_ir_in1.dim), x2) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        out = ins.path_weight * einsum("uv,ui,v->ui", w, x1, x2.reshape(mul_ir_in2.dim)) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        out = ins.path_weight * einsum("uv,ui,vi->u", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        out = einsum("uv,ijk,uvij->uk", w, w3j, xx)
                else:
                    # not so useful operation because v is summed
                    out = einsum("ijk,uvij->uk", w3j, xx)
            if ins.connection_mode == 'uvv':
                assert mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l) == (0, 0, 0):
                        out = ins.path_weight * einsum("uv,u,v->v", w, x1.reshape(mul_ir_in1.dim), x2.reshape(mul_ir_in2.dim))
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        out = ins.path_weight * einsum("uv,u,vj->vj", w, x1.reshape(mul_ir_in1.dim), x2) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        out = ins.path_weight * einsum("uv,ui,v->vi", w, x1, x2.reshape(mul_ir_in2.dim)) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        out = ins.path_weight * einsum("uv,ui,vi->v", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        out = einsum("uv,ijk,uvij->vk", w, w3j, xx)
                else:
                    # not so useful operation because u is summed
                    out = einsum("ijk,uvij->vk", w3j, xx)
            if ins.connection_mode == 'uuw':
                assert mul_ir_in1.mul == mul_ir_in2.mul
                if ins.has_weight:
                    out = einsum("uw,ijk,uij->wk", w, w3j, xx)
                else:
                    # equivalent to tp(x, y, 'uuu').sum('u')
                    assert mul_ir_out.mul == 1
                    out = einsum("ijk,uij->k", w3j, xx)
            if ins.connection_mode == 'uuu':
                assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    out = einsum("u,ijk,uij->uk", w, w3j, xx)
                else:
                    out = einsum("ijk,uij->uk", w3j, xx)
            if ins.connection_mode == 'uvuv':
                assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    out = einsum("uv,ijk,uvij->uvk", w, w3j, xx)
                else:
                    out = einsum("ijk,uvij->uvk", w3j, xx)

            out_list += [out]

        out = [
            _sum_tensors(
                [out for ins, out in zip(self.instructions, out_list) if ins.i_out == i_out],
                shape=(mul_ir_out.mul, mul_ir_out.ir.dim),
            )
            for i_out, mul_ir_out in enumerate(self.irreps_out)
        ]
        if output_list:
            return out
        return _flat_concatenate(out)

    @partial(jax.jit, static_argnums=(0,), static_argnames=('optimize_einsums', 'custom_einsum_vjp'))
    def right(self, weights, input2=None, *, optimize_einsums=False, custom_einsum_vjp=False):
        if input2 is None:
            weights, input2 = [], weights

        # = Short-circut for zero dimensional =
        if self.irreps_in1.dim == 0 or self.irreps_in2.dim == 0 or self.irreps_out.dim == 0:
            return jnp.zeros((self.irreps_in1.dim, self.irreps_out.dim,))

        # = extract individual input irreps =
        x2_list = self.irreps_in2.as_list(input2)

        if custom_einsum_vjp:
            assert optimize_einsums
            einsum = opt_einsum
        else:
            einsum = partial(jnp.einsum, optimize='optimal' if optimize_einsums else 'greedy')

        weight_index = 0

        out_list = []

        for ins in self.instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]

            if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
                continue

            x2 = x2_list[ins.i_in2]

            if ins.has_weight:
                w = weights[weight_index]
                assert w.shape == ins.path_shape
                weight_index += 1

            with jax.core.eval_context():
                w3j = wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)

            if ins.connection_mode == 'uvw':
                assert ins.has_weight
                out = einsum("uvw,ijk,vj->uiwk", w, w3j, x2)
            if ins.connection_mode == 'uvu':
                assert mul_ir_in1.mul == mul_ir_out.mul
                if ins.has_weight:
                    out = einsum("uv,ijk,vj,uw->uiwk", w, w3j, x2, jnp.eye(mul_ir_in1.mul))
                else:
                    # not so useful operation because v is summed
                    out = einsum("ijk,vj,uw->uiwk", w3j, x2, jnp.eye(mul_ir_in1.mul))
            if ins.connection_mode == 'uvv':
                assert mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    out = einsum("uv,ijk,vj->uivk", w, w3j, x2)
                else:
                    # not so useful operation because u is summed
                    out = einsum("ijk,vj,u->uivk", w3j, x2, jnp.ones((mul_ir_in1.mul,)))
            if ins.connection_mode == 'uuw':
                assert mul_ir_in1.mul == mul_ir_in2.mul
                if ins.has_weight:
                    out = einsum("uw,ijk,uj->uiwk", w, w3j, x2)
                else:
                    # equivalent to tp(x, y, 'uuu').sum('u')
                    assert mul_ir_out.mul == 1
                    out = einsum("ijk,uj->uik", w3j, x2)
            if ins.connection_mode == 'uuu':
                assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    out = einsum("u,ijk,uj,uw->uiwk", w, w3j, x2, jnp.eye(mul_ir_in1.mul))
                else:
                    out = einsum("ijk,uj,uw->uiwk", w3j, x2, jnp.eye(mul_ir_in1.mul))
            if ins.connection_mode == 'uvuv':
                assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    out = einsum("uv,ijk,vj,uw->uiwvk", w, w3j, x2, jnp.eye(mul_ir_in1.mul))
                else:
                    out = einsum("ijk,vj,uw->uiwvk", w3j, x2, jnp.eye(mul_ir_in1.mul))

            out = ins.path_weight * out

            out_list += [out.reshape(mul_ir_in1.dim, mul_ir_out.dim)]

        return jnp.concatenate([
            jnp.concatenate([
                _sum_tensors(
                    [out for ins, out in zip(self.instructions, out_list) if (ins.i_in1, ins.i_out) == (i_in1, i_out)],
                    shape=(mul_ir_in1.dim, mul_ir_out.dim),
                )
                for i_out, mul_ir_out in enumerate(self.irreps_out)
                if mul_ir_out.mul > 0
            ], axis=1)
            for i_in1, mul_ir_in1 in enumerate(self.irreps_in1)
            if mul_ir_in1.mul > 0
        ], axis=0)


class FullyConnectedTensorProduct(TensorProduct):
    def __init__(
        self,
        irreps_in1: Any,
        irreps_in2: Any,
        irreps_out: Any,
        in1_var: Optional[List[float]] = None,
        in2_var: Optional[List[float]] = None,
        out_var: Optional[List[float]] = None,
        normalization: str = 'component',
    ):
        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)

        instructions = [
            (i_1, i_2, i_out, 'uvw', True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(irreps_in1, irreps_in2, irreps_out, instructions, in1_var, in2_var, out_var, normalization)


class ElementwiseTensorProduct(TensorProduct):
    def __init__(
        self,
        irreps_in1: Any,
        irreps_in2: Any,
        filter_ir_out=None,
        normalization: str = 'component',
    ):
        irreps_in1 = Irreps(irreps_in1).simplify()
        irreps_in2 = Irreps(irreps_in2).simplify()
        if filter_ir_out is not None:
            filter_ir_out = [Irrep(ir) for ir in filter_ir_out]

        assert irreps_in1.num_irreps == irreps_in2.num_irreps

        irreps_in1 = list(irreps_in1)
        irreps_in2 = list(irreps_in2)

        i = 0
        while i < len(irreps_in1):
            mul_1, ir_1 = irreps_in1[i]
            mul_2, ir_2 = irreps_in2[i]

            if mul_1 < mul_2:
                irreps_in2[i] = (mul_1, ir_2)
                irreps_in2.insert(i + 1, (mul_2 - mul_1, ir_2))

            if mul_2 < mul_1:
                irreps_in1[i] = (mul_2, ir_1)
                irreps_in1.insert(i + 1, (mul_1 - mul_2, ir_1))
            i += 1

        irreps_out = []
        instructions = []
        for i, ((mul, ir_1), (mul_2, ir_2)) in enumerate(zip(irreps_in1, irreps_in2)):
            assert mul == mul_2
            for ir in ir_1 * ir_2:

                if filter_ir_out is not None and ir not in filter_ir_out:
                    continue

                i_out = len(irreps_out)
                irreps_out.append((mul, ir))
                instructions += [
                    (i, i, i_out, 'uuu', False)
                ]

        super().__init__(irreps_in1, irreps_in2, irreps_out, instructions, normalization=normalization)
