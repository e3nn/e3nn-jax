"""Defines the functional tensor product."""

import collections
import itertools
from dataclasses import dataclass, field, replace
from functools import partial
from math import prod, sqrt
from typing import Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, sparsify

import e3nn_jax as e3nn
from e3nn_jax._src.utils.dtype import get_pytree_dtype
from e3nn_jax._src.utils.einsum import einsum as opt_einsum
from e3nn_jax._src.utils.sum_tensors import sum_tensors


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=True)
class Instruction:
    """Defines an instruction for a tensor product."""

    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    weight_std: float
    first_input_multiplicity: int
    second_input_multiplicity: int
    output_multiplicity: int
    path_shape: Tuple[int, ...] = field(init=False)
    num_elements: int = field(init=False)

    def __post_init__(self):
        if self.connection_mode not in [
            "uvw",
            "uvu",
            "uvv",
            "uuw",
            "uuu",
            "uvuv",
            "uvu<v",
            "u<vw",
        ]:
            raise ValueError(
                f"Unsupported connection_mode {self.connection_mode} for instruction."
            )

        path_shape = {
            "uvw": (
                self.first_input_multiplicity,
                self.second_input_multiplicity,
                self.output_multiplicity,
            ),
            "uvu": (self.first_input_multiplicity, self.second_input_multiplicity),
            "uvv": (self.first_input_multiplicity, self.second_input_multiplicity),
            "uuw": (self.first_input_multiplicity, self.output_multiplicity),
            "uuu": (self.first_input_multiplicity,),
            "uvuv": (self.first_input_multiplicity, self.second_input_multiplicity),
            "uvu<v": (
                self.first_input_multiplicity
                * (self.second_input_multiplicity - 1)
                // 2,
            ),
            "u<vw": (
                self.first_input_multiplicity
                * (self.second_input_multiplicity - 1)
                // 2,
                self.output_multiplicity,
            ),
        }[self.connection_mode]
        super().__setattr__("path_shape", path_shape)

        num_elements = {
            "uvw": (self.first_input_multiplicity * self.second_input_multiplicity),
            "uvu": self.second_input_multiplicity,
            "uvv": self.first_input_multiplicity,
            "uuw": self.first_input_multiplicity,
            "uuu": 1,
            "uvuv": 1,
            "uvu<v": 1,
            "u<vw": self.first_input_multiplicity
            * (self.second_input_multiplicity - 1)
            // 2,
        }[self.connection_mode]
        super().__setattr__("num_elements", num_elements)

    def replace(self, **changes) -> "Instruction":
        return replace(self, **changes)

    def __repr__(self) -> str:
        return (
            "Instruction("
            + ", ".join(
                [
                    f"i={self.i_in1},{self.i_in2},{self.i_out}",
                    f"mode={self.connection_mode}",
                    f"has_weight={self.has_weight}",
                    f"path_weight={self.path_weight}",
                    f"weight_std={self.weight_std}",
                    f"mul={self.first_input_multiplicity},{self.second_input_multiplicity},{self.output_multiplicity}",
                    f"path_shape={self.path_shape}",
                    f"num_elements={self.num_elements}",
                ]
            )
            + ")"
        )


class FunctionalTensorProduct:
    r"""Tensor product of two tensors.

    Args:
        irreps_in1: :class:`~e3nn_jax.Irreps` of the first tensor.
        irreps_in2: :class:`~e3nn_jax.Irreps` of the second tensor.
        irreps_out: :class:`~e3nn_jax.Irreps` of the output tensor.
        instructions: List of instructions.
            ``[(i_in1, i_in2, i_out, connection_mode, has_weight, (path_weight)), ...]``
            - i_in1, i_in2, i_out are indices of the irreps_in1, irreps_in2, irreps_out.
            - connection_mode is one of ``uvw``, ``uvu``, ``uvv``, ``uuw``, ``uuu``, ``uvuv``
            - has_weight is a boolean indicating whether the instruction has a weight.
            - path_weight (optional, 1.0 by default) is the weight of the path.

        in1_var: Variance of the first tensor.
        in2_var: Variance of the second tensor.
        out_var: Variance of the output tensor.
        irrep_normalization: Normalization of the tensors. ``component`` or ``norm``.
        path_normalization (str or float): Normalization of the paths, ``element`` or ``path``.
            0/1 corresponds to a normalization where each element/path has an equal contribution to the forward propagation.
        gradient_normalization (str or float): Normalization of the gradients, ``element`` or ``path``.
            0/1 corresponds to a normalization where each element/path has an equal contribution to the learning.
    """
    irreps_in1: e3nn.Irreps
    irreps_in2: e3nn.Irreps
    irreps_out: e3nn.Irreps
    instructions: List[Instruction]
    output_mask: jnp.ndarray

    def __init__(
        self,
        irreps_in1: e3nn.Irreps,
        irreps_in2: e3nn.Irreps,
        irreps_out: e3nn.Irreps,
        instructions: List[Tuple[int, int, int, str, bool, Optional[float]]],
        in1_var: Optional[List[float]] = None,
        in2_var: Optional[List[float]] = None,
        out_var: Optional[List[float]] = None,
        irrep_normalization: str = None,
        path_normalization: Union[str, float] = None,
        gradient_normalization: Union[str, float] = None,
    ):
        self.irreps_in1 = e3nn.Irreps(irreps_in1)
        self.irreps_in2 = e3nn.Irreps(irreps_in2)
        self.irreps_out = e3nn.Irreps(irreps_out)
        del irreps_in1, irreps_in2, irreps_out

        instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]
        instructions = [
            Instruction(
                i_in1,
                i_in2,
                i_out,
                connection_mode,
                has_weight,
                path_weight,
                None,
                self.irreps_in1[i_in1].mul,
                self.irreps_in2[i_in2].mul,
                self.irreps_out[i_out].mul,
            )
            for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in instructions
        ]

        if in1_var is None:
            in1_var = [1.0 for _ in self.irreps_in1]

        if in2_var is None:
            in2_var = [1.0 for _ in self.irreps_in2]

        if out_var is None:
            out_var = [1.0 for _ in self.irreps_out]

        if irrep_normalization is None:
            irrep_normalization = e3nn.config("irrep_normalization")

        if path_normalization is None:
            path_normalization = e3nn.config("path_normalization")
        if isinstance(path_normalization, str):
            path_normalization = {"element": 0.0, "path": 1.0}[path_normalization]

        if gradient_normalization is None:
            gradient_normalization = e3nn.config("gradient_normalization")
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[
                gradient_normalization
            ]

        self.instructions = _normalize_instruction_path_weights(
            instructions,
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            in1_var,
            in2_var,
            out_var,
            irrep_normalization,
            path_normalization,
            gradient_normalization,
        )

        with jax.ensure_compile_time_eval():
            if self.irreps_out.dim > 0:
                self.output_mask = jnp.concatenate(
                    [
                        jnp.ones(mul_ir.dim, dtype=bool)
                        if any(
                            (ins.i_out == i_out)
                            and (ins.path_weight != 0)
                            and (0 not in ins.path_shape)
                            for ins in self.instructions
                        )
                        else jnp.zeros(mul_ir.dim, dtype=bool)
                        for i_out, mul_ir in enumerate(self.irreps_out)
                    ]
                )
            else:
                self.output_mask = jnp.ones(0, dtype=bool)

    def left_right(
        self,
        weights: Union[List[jnp.ndarray], jnp.ndarray],
        input1: e3nn.IrrepsArray,
        input2: e3nn.IrrepsArray = None,
        *,
        custom_einsum_jvp: bool = None,
        fused: bool = None,
        sparse: bool = None,
    ) -> e3nn.IrrepsArray:
        r"""Compute the tensor product of two input tensors.

        Args:
            weights (array or list of arrays): The weights of the tensor product.
            input1 (IrrepsArray): The first input tensor.
            input2 (IrrepsArray): The second input tensor.
            custom_einsum_jvp (bool): If True, use the custom jvp for the einsum code.
            fused (bool): If True, fuse all the einsums.

        Returns:
            `IrrepsArray`: The output tensor.
        """
        if custom_einsum_jvp is None:
            custom_einsum_jvp = e3nn.config("custom_einsum_jvp")
        if fused is None:
            fused = e3nn.config("fused")
        if sparse is None:
            sparse = e3nn.config("sparse_tp")

        if input2 is None:
            weights, input1, input2 = [], weights, input1

        return _left_right(
            self,
            weights,
            input1.rechunk(self.irreps_in1),
            input2.rechunk(self.irreps_in2),
            custom_einsum_jvp=custom_einsum_jvp,
            fused=fused,
            sparse=sparse,
        )

    def right(
        self,
        weights: List[jnp.ndarray],
        input2: e3nn.IrrepsArray = None,
        *,
        custom_einsum_jvp=None,
    ) -> jnp.ndarray:
        r"""Compute the right contraction of the tensor product.

        Args:
            weights (array or list of arrays): The weights of the tensor product.
            input2 (IrrepsArray): The second input tensor.
            custom_einsum_jvp (bool): If True, use the custom jvp for the einsum code.

        Returns:
            A matrix of shape ``(irreps_in1.dim, irreps_out.dim)``.
        """
        if custom_einsum_jvp is None:
            custom_einsum_jvp = e3nn.config("custom_einsum_jvp")

        if input2 is None:
            weights, input2 = [], weights

        return _right(
            self,
            weights,
            input2.rechunk(self.irreps_in2),
            custom_einsum_jvp=custom_einsum_jvp,
        )

    def __repr__(self):
        npath = sum(prod(i.path_shape) for i in self.instructions)
        nweight = sum(prod(i.path_shape) for i in self.instructions if i.has_weight)
        return (
            f"{self.__class__.__name__}"
            f"({self.irreps_in1.simplify()} x {self.irreps_in2.simplify()} "
            f"-> {self.irreps_out.simplify()} | {npath} paths | {nweight} weights)"
        )


def _flat_concatenate(xs):
    if any(x is None for x in xs):
        return None
    if len(xs) > 0:
        return jnp.concatenate([x.flatten() for x in xs])
    return jnp.zeros((0,), dtype=jnp.float32)


def _normalize_instruction_path_weights(
    instructions: List[Instruction],
    first_input_irreps: e3nn.Irreps,
    second_input_irreps: e3nn.Irreps,
    output_irreps: e3nn.Irreps,
    first_input_variance: List[float],
    second_input_variance: List[float],
    output_variance: List[float],
    irrep_normalization: str,
    path_normalization_exponent: float,
    gradient_normalization_exponent: float,
) -> List[Instruction]:
    """Returns instructions with normalized path weights."""

    def var(instruction):
        return (
            first_input_variance[instruction.i_in1]
            * second_input_variance[instruction.i_in2]
            * instruction.num_elements
        )

    # Precompute normalization factors.
    path_normalization_sums = collections.defaultdict(lambda: 0.0)
    for instruction in instructions:
        path_normalization_sums[instruction.i_out] += var(instruction) ** (
            1.0 - path_normalization_exponent
        )

    path_normalization_factors = {
        instruction: var(instruction) ** path_normalization_exponent
        * path_normalization_sums[instruction.i_out]
        for instruction in instructions
    }

    def update(instruction: Instruction) -> float:
        """Computes normalized path weight for a single instructions, with precomputed path normalization factors."""

        if irrep_normalization not in ["component", "norm", "none"]:
            raise ValueError(f"Unsupported irrep normalization: {irrep_normalization}.")

        mul_ir_in1 = first_input_irreps[instruction.i_in1]
        mul_ir_in2 = second_input_irreps[instruction.i_in2]
        mul_ir_out = output_irreps[instruction.i_out]

        assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
        assert (
            abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l)
            <= mul_ir_out.ir.l
            <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
        )

        if irrep_normalization == "component":
            alpha = mul_ir_out.ir.dim
        if irrep_normalization == "norm":
            alpha = mul_ir_in1.ir.dim * mul_ir_in2.ir.dim
        if irrep_normalization == "none":
            alpha = 1

        x = path_normalization_factors[instruction]
        if x > 0.0:
            alpha /= x

        alpha *= output_variance[instruction.i_out]
        alpha *= instruction.path_weight

        if instruction.has_weight:
            return instruction.replace(
                path_weight=sqrt(alpha) ** gradient_normalization_exponent,
                weight_std=sqrt(alpha) ** (1.0 - gradient_normalization_exponent),
            )
        else:
            return instruction.replace(path_weight=sqrt(alpha))

    return [update(instruction) for instruction in instructions]


@partial(jax.profiler.annotate_function, name="TensorProduct.left_right")
def _left_right(
    self: FunctionalTensorProduct,
    weights: Union[List[jnp.ndarray], jnp.ndarray],
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    *,
    custom_einsum_jvp: bool = False,
    fused: bool = False,
    sparse: bool = False,
):
    dtype = get_pytree_dtype(weights, input1, input2)
    if dtype.kind == "i":
        dtype = jnp.float32

    if self.irreps_in1.dim == 0 or self.irreps_in2.dim == 0 or self.irreps_out.dim == 0:
        return e3nn.zeros(self.irreps_out, (), dtype)

    if sparse:
        assert (
            not custom_einsum_jvp
        ), "custom_einsum_jvp does not support sparse tensors."

        def einsum(op, *args):
            f = sparsify(lambda *args: jnp.einsum(op, *args))
            return f(*args)

    else:
        einsum = opt_einsum if custom_einsum_jvp else jnp.einsum

    if isinstance(weights, list):
        assert len(weights) == len(
            [ins for ins in self.instructions if ins.has_weight]
        ), (
            len(weights),
            len([ins for ins in self.instructions if ins.has_weight]),
        )
        weights_flat = _flat_concatenate(weights)
        weights_list = weights
    else:
        weights_flat = weights
        weights_list = []
        i = 0
        for ins in self.instructions:
            if ins.has_weight:
                n = prod(ins.path_shape)
                weights_list.append(weights[i : i + n].reshape(ins.path_shape))
                i += n
        assert i == weights.size
    del weights

    assert (
        input1.ndim == 1
    ), f"input1 is shape {input1.shape}. Execting ndim to be 1. Use jax.vmap to map over input1"
    assert (
        input2.ndim == 1
    ), f"input2 is shape {input2.shape}. Execting ndim to be 1. Use jax.vmap to map over input2"

    if fused:
        output = _fused_left_right(
            self, weights_flat, input1, input2, einsum, sparse, dtype
        )
    else:
        output = _block_left_right(
            self, weights_list, input1, input2, einsum, sparse, dtype
        )

    assert (
        output.dtype == dtype
    ), f"output.dtype {output.dtype} != dtype {dtype}, Please report this bug."
    return output


def _block_left_right(
    self: FunctionalTensorProduct,
    weights_list: List[jnp.ndarray],
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    einsum: Callable,
    sparse: bool,
    dtype: jnp.dtype,
) -> e3nn.IrrepsArray:
    weight_index = 0

    out_list = []

    for ins in self.instructions:
        mul_ir_in1 = self.irreps_in1[ins.i_in1]
        mul_ir_in2 = self.irreps_in2[ins.i_in2]
        mul_ir_out = self.irreps_out[ins.i_out]

        if ins.has_weight:
            w = weights_list[weight_index]
            assert w.shape == ins.path_shape
            weight_index += 1

        if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            # TODO verify that there is no need for
            # out_list += [None]
            continue

        x1 = input1.chunks[ins.i_in1]
        x2 = input2.chunks[ins.i_in2]

        if x1 is None or x2 is None:
            out_list += [None]
            continue

        with jax.ensure_compile_time_eval():
            w3j = e3nn.clebsch_gordan(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
            w3j = ins.path_weight * w3j
            w3j = w3j.astype(dtype)
            if sparse:
                w3j = BCOO.fromdense(w3j)

        if ins.connection_mode == "uvw":
            assert ins.has_weight
            out = einsum("uvw,ijk,ui,vj->wk", w, w3j, x1, x2)
        if ins.connection_mode == "uvu":
            assert mul_ir_in1.mul == mul_ir_out.mul
            if ins.has_weight:
                out = einsum("uv,ijk,ui,vj->uk", w, w3j, x1, x2)
            else:
                # not so useful operation because v is summed
                out = einsum("ijk,ui,vj->uk", w3j, x1, x2)
        if ins.connection_mode == "uvv":
            assert mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                out = einsum("uv,ijk,ui,vj->vk", w, w3j, x1, x2)
            else:
                # not so useful operation because u is summed
                out = einsum("ijk,ui,vj->vk", w3j, x1, x2)
        if ins.connection_mode == "uuw":
            assert mul_ir_in1.mul == mul_ir_in2.mul
            if ins.has_weight:
                out = einsum("uw,ijk,ui,uj->wk", w, w3j, x1, x2)
            else:
                # equivalent to tp(x, y, 'uuu').sum('u')
                assert mul_ir_out.mul == 1
                out = einsum("ijk,ui,uj->k", w3j, x1, x2)
        if ins.connection_mode == "uuu":
            assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                out = einsum("u,ijk,ui,uj->uk", w, w3j, x1, x2)
            else:
                out = einsum("ijk,ui,uj->uk", w3j, x1, x2)
        if ins.connection_mode == "uvuv":
            assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                out = einsum("uv,ijk,ui,vj->uvk", w, w3j, x1, x2)
            else:
                out = einsum("ijk,ui,vj->uvk", w3j, x1, x2)
        if ins.connection_mode == "uvu<v":
            assert mul_ir_in1.mul == mul_ir_in2.mul
            assert mul_ir_in1.mul * (mul_ir_in1.mul - 1) // 2 == mul_ir_out.mul
            i = jnp.triu_indices(mul_ir_in1.mul, 1)
            xx = jnp.einsum("ui,vj->uvij", x1, x2)[i[0], i[1]]  # uvij -> wij
            if ins.has_weight:
                out = einsum("w,ijk,wij->wk", w, w3j, xx)
            else:
                out = einsum("ijk,wij->wk", w3j, xx)
        if ins.connection_mode == "u<vw":
            assert mul_ir_in1.mul == mul_ir_in2.mul
            assert ins.has_weight
            i = jnp.triu_indices(mul_ir_in1.mul, 1)
            out = einsum(
                "qw,ijk,qij->wk", w, w3j, jnp.einsum("ui,vj->uvij", x1, x2)[i[0], i[1]]
            )

        out_list += [out]

    out = [
        sum_tensors(
            [
                out
                for ins, out in zip(self.instructions, out_list)
                if ins.i_out == i_out
            ],
            shape=(mul_ir_out.mul, mul_ir_out.ir.dim),
            empty_return_none=True,
            dtype=dtype,
        )
        for i_out, mul_ir_out in enumerate(self.irreps_out)
    ]
    return e3nn.from_chunks(self.irreps_out, out, (), dtype)


def _fused_left_right(
    self: FunctionalTensorProduct,
    weights_flat: jnp.ndarray,
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    einsum: Callable,
    sparse: bool,
    dtype: jnp.dtype,
) -> e3nn.IrrepsArray:
    with jax.ensure_compile_time_eval():
        num_path = weights_flat.size
        has_path_with_no_weights = any(not ins.has_weight for ins in self.instructions)
        i = 0

        if has_path_with_no_weights:
            num_path += 1
            i += 1

        big_w3j = jnp.zeros(
            (
                num_path,
                self.irreps_in1.dim,
                self.irreps_in2.dim,
                self.irreps_out.dim,
            ),
            dtype=dtype,
        )
        for ins in self.instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]
            m1, m2, mo = mul_ir_in1.mul, mul_ir_in2.mul, mul_ir_out.mul
            d1, d2, do = mul_ir_in1.ir.dim, mul_ir_in2.ir.dim, mul_ir_out.ir.dim
            s1 = self.irreps_in1[: ins.i_in1].dim
            s2 = self.irreps_in2[: ins.i_in2].dim
            so = self.irreps_out[: ins.i_out].dim

            w3j = e3nn.clebsch_gordan(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)

            def set_w3j(x, i, u, v, w):
                return x.at[
                    i,
                    s1 + u * d1 : s1 + (u + 1) * d1,
                    s2 + v * d2 : s2 + (v + 1) * d2,
                    so + w * do : so + (w + 1) * do,
                ].add((ins.path_weight * w3j).astype(dtype))

            if ins.connection_mode == "uvw":
                assert ins.has_weight
                for u, v, w in itertools.product(range(m1), range(m2), range(mo)):
                    big_w3j = set_w3j(big_w3j, i, u, v, w)
                    i += 1
            elif ins.connection_mode == "uvu":
                assert ins.has_weight
                for u, v in itertools.product(range(m1), range(m2)):
                    big_w3j = set_w3j(big_w3j, i, u, v, u)
                    i += 1
            elif ins.connection_mode == "uvv":
                assert ins.has_weight
                for u, v in itertools.product(range(m1), range(m2)):
                    big_w3j = set_w3j(big_w3j, i, u, v, v)
                    i += 1
            elif ins.connection_mode == "uuu":
                for u in range(m1):
                    if ins.has_weight:
                        big_w3j = set_w3j(big_w3j, i, u, u, u)
                        i += 1
                    else:
                        big_w3j = set_w3j(big_w3j, 0, u, u, u)
            elif ins.connection_mode == "uvuv":
                for u in range(m1):
                    for v in range(m2):
                        if ins.has_weight:
                            big_w3j = set_w3j(big_w3j, i, u, v, u * m2 + v)
                            i += 1
                        else:
                            big_w3j = set_w3j(big_w3j, 0, u, v, u * m2 + v)
            else:
                assert False

        assert i == num_path

        if sparse:
            big_w3j = BCOO.fromdense(big_w3j)

    if has_path_with_no_weights and big_w3j.shape[0] == 1:
        if sparse:
            f = sparsify(lambda w3j, x1, x2: einsum("pijk,i,j->k", w3j, x1, x2))
            out = f(big_w3j, input1.array, input2.array)
        else:
            out = einsum("pijk,i,j->k", big_w3j, input1.array, input2.array)
    else:
        if has_path_with_no_weights:
            weights_flat = jnp.concatenate(
                [jnp.ones((1,), weights_flat.dtype), weights_flat]
            )

        if sparse:
            f = sparsify(lambda w, w3j, x1, x2: einsum("p,pijk,i,j->k", w, w3j, x1, x2))
            out = f(weights_flat, big_w3j, input1.array, input2.array)
        else:
            out = einsum(
                "p,pijk,i,j->k", weights_flat, big_w3j, input1.array, input2.array
            )
    return e3nn.IrrepsArray(self.irreps_out, out)


@partial(jax.profiler.annotate_function, name="TensorProduct.right")
def _right(
    self: FunctionalTensorProduct,
    weights: List[jnp.ndarray],
    input2: e3nn.IrrepsArray,
    *,
    custom_einsum_jvp: bool = False,
) -> jnp.ndarray:
    dtype = get_pytree_dtype(weights, input2)
    if dtype.kind == "i":
        dtype = jnp.float32

    # = Short-circut for zero dimensional =
    if self.irreps_in1.dim == 0 or self.irreps_in2.dim == 0 or self.irreps_out.dim == 0:
        return jnp.zeros(
            (
                self.irreps_in1.dim,
                self.irreps_out.dim,
            ),
            dtype=dtype,
        )

    einsum = opt_einsum if custom_einsum_jvp else jnp.einsum

    weight_index = 0

    out_list = []

    for ins in self.instructions:
        mul_ir_in1 = self.irreps_in1[ins.i_in1]
        mul_ir_in2 = self.irreps_in2[ins.i_in2]
        mul_ir_out = self.irreps_out[ins.i_out]

        x2 = input2.chunks[ins.i_in2]

        if ins.has_weight:
            w = weights[weight_index]
            assert w.shape == ins.path_shape, (
                w.shape,
                ins.path_shape,
                weight_index,
                ins,
            )
            weight_index += 1

        if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            # TODO add tests for this case
            out_list += [jnp.zeros((mul_ir_in1.dim, mul_ir_out.dim), dtype=dtype)]
            continue

        with jax.ensure_compile_time_eval():
            w3j = e3nn.clebsch_gordan(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
            w3j = w3j.astype(dtype)

        if ins.connection_mode == "uvw":
            assert ins.has_weight
            out = einsum("uvw,ijk,vj->uiwk", w, w3j, x2)
        if ins.connection_mode == "uvu":
            assert mul_ir_in1.mul == mul_ir_out.mul
            if ins.has_weight:
                out = einsum("uv,ijk,vj,uw->uiwk", w, w3j, x2, jnp.eye(mul_ir_in1.mul))
            else:
                # not so useful operation because v is summed
                out = einsum("ijk,vj,uw->uiwk", w3j, x2, jnp.eye(mul_ir_in1.mul))
        if ins.connection_mode == "uvv":
            assert mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                out = einsum("uv,ijk,vj->uivk", w, w3j, x2)
            else:
                # not so useful operation because u is summed
                out = einsum(
                    "ijk,vj,u->uivk", w3j, x2, jnp.ones((mul_ir_in1.mul,), dtype)
                )
        if ins.connection_mode == "uuw":
            assert mul_ir_in1.mul == mul_ir_in2.mul
            if ins.has_weight:
                out = einsum("uw,ijk,uj->uiwk", w, w3j, x2)
            else:
                # equivalent to tp(x, y, 'uuu').sum('u')
                assert mul_ir_out.mul == 1
                out = einsum("ijk,uj->uik", w3j, x2)
        if ins.connection_mode == "uuu":
            assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                out = einsum("u,ijk,uj,uw->uiwk", w, w3j, x2, jnp.eye(mul_ir_in1.mul))
            else:
                out = einsum("ijk,uj,uw->uiwk", w3j, x2, jnp.eye(mul_ir_in1.mul))
        if ins.connection_mode == "uvuv":
            assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                out = einsum("uv,ijk,vj,uw->uiwvk", w, w3j, x2, jnp.eye(mul_ir_in1.mul))
            else:
                out = einsum("ijk,vj,uw->uiwvk", w3j, x2, jnp.eye(mul_ir_in1.mul))

        out = ins.path_weight * out

        out_list += [out.reshape(mul_ir_in1.dim, mul_ir_out.dim)]

    output = jnp.concatenate(
        [
            jnp.concatenate(
                [
                    sum_tensors(
                        [
                            out
                            for ins, out in zip(self.instructions, out_list)
                            if (ins.i_in1, ins.i_out) == (i_in1, i_out)
                        ],
                        shape=(mul_ir_in1.dim, mul_ir_out.dim),
                        dtype=dtype,
                    )
                    for i_out, mul_ir_out in enumerate(self.irreps_out)
                    if mul_ir_out.mul > 0
                ],
                axis=1,
            )
            for i_in1, mul_ir_in1 in enumerate(self.irreps_in1)
            if mul_ir_in1.mul > 0
        ],
        axis=0,
    )

    assert output.dtype == dtype, f"{output.dtype} != {dtype}, Please report this bug."
    return output
