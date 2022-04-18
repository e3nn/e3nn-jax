from typing import Any, List, Optional

import haiku as hk
import jax
import jax.numpy as jnp

from e3nn_jax import FunctionalTensorProduct, Irrep, Irreps, IrrepsData
from e3nn_jax.util.decorators import overload_for_irreps_without_data


def naive_broadcast_decorator(func):
    def wrapper(*args):
        shape = jnp.broadcast_shapes(*(arg.shape for arg in args))
        args = [arg.broadcast_to(shape) for arg in args]
        f = func
        for _ in range(len(shape)):
            f = jax.vmap(f)
        return f(*args)

    return wrapper


def FunctionalFullyConnectedTensorProduct(
    irreps_in1: Any,
    irreps_in2: Any,
    irreps_out: Any,
    in1_var: Optional[List[float]] = None,
    in2_var: Optional[List[float]] = None,
    out_var: Optional[List[float]] = None,
    irrep_normalization: str = "component",
    path_normalization: str = "element",
):
    irreps_in1 = Irreps(irreps_in1)
    irreps_in2 = Irreps(irreps_in2)
    irreps_out = Irreps(irreps_out)

    instructions = [
        (i_1, i_2, i_out, "uvw", True)
        for i_1, (_, ir_1) in enumerate(irreps_in1)
        for i_2, (_, ir_2) in enumerate(irreps_in2)
        for i_out, (_, ir_out) in enumerate(irreps_out)
        if ir_out in ir_1 * ir_2
    ]
    return FunctionalTensorProduct(
        irreps_in1, irreps_in2, irreps_out, instructions, in1_var, in2_var, out_var, irrep_normalization, path_normalization
    )


class FullyConnectedTensorProduct(hk.Module):
    def __init__(self, irreps_out, *, irreps_in1=None, irreps_in2=None):
        super().__init__()

        self.irreps_out = Irreps(irreps_out)
        self.irreps_in1 = Irreps(irreps_in1) if irreps_in1 is not None else None
        self.irreps_in2 = Irreps(irreps_in2) if irreps_in2 is not None else None

    def __call__(self, x1: IrrepsData, x2: IrrepsData) -> IrrepsData:
        if self.irreps_in1 is not None:
            x1 = IrrepsData.new(self.irreps_in1, x1)
        if self.irreps_in2 is not None:
            x2 = IrrepsData.new(self.irreps_in2, x2)

        x1 = x1.remove_nones().simplify()
        x2 = x2.remove_nones().simplify()

        tp = FunctionalFullyConnectedTensorProduct(x1.irreps, x2.irreps, self.irreps_out.simplify())
        ws = [
            hk.get_parameter(
                f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] {tp.irreps_in1[ins.i_in1]},{tp.irreps_in2[ins.i_in2]},{tp.irreps_out[ins.i_out]}",
                shape=ins.path_shape,
                init=hk.initializers.RandomNormal(),
            )
            for ins in tp.instructions
        ]
        f = naive_broadcast_decorator(lambda x1, x2: tp.left_right(ws, x1, x2))
        output = f(x1, x2)
        return output.convert(self.irreps_out)


@overload_for_irreps_without_data((0, 1))
def full_tensor_product(
    input1: IrrepsData,
    input2: IrrepsData,
    filter_ir_out=None,
    irrep_normalization: str = "component",
):
    if filter_ir_out is not None:
        filter_ir_out = [Irrep(ir) for ir in filter_ir_out]

    irreps_out = []
    instructions = []
    for i_1, (mul_1, ir_1) in enumerate(input1.irreps):
        for i_2, (mul_2, ir_2) in enumerate(input2.irreps):
            for ir_out in ir_1 * ir_2:

                if filter_ir_out is not None and ir_out not in filter_ir_out:
                    continue

                i_out = len(irreps_out)
                irreps_out.append((mul_1 * mul_2, ir_out))
                instructions += [(i_1, i_2, i_out, "uvuv", False)]

    irreps_out = Irreps(irreps_out)
    irreps_out, p, _ = irreps_out.sort()

    instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

    tp = FunctionalTensorProduct(
        input1.irreps, input2.irreps, irreps_out, instructions, irrep_normalization=irrep_normalization
    )

    return naive_broadcast_decorator(tp.left_right)(input1, input2)


@overload_for_irreps_without_data((0, 1))
def elementwise_tensor_product(
    input1: IrrepsData,
    input2: IrrepsData,
    filter_ir_out=None,
    irrep_normalization: str = "component",
    path_normalization: str = "element",
) -> IrrepsData:
    if filter_ir_out is not None:
        filter_ir_out = [Irrep(ir) for ir in filter_ir_out]

    assert input1.irreps.num_irreps == input2.irreps.num_irreps

    irreps_in1 = list(input1.irreps)
    irreps_in2 = list(input2.irreps)

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

    input1 = input1.convert(irreps_in1)
    input2 = input2.convert(irreps_in2)

    irreps_out = []
    instructions = []
    for i, ((mul, ir_1), (mul_2, ir_2)) in enumerate(zip(irreps_in1, irreps_in2)):
        assert mul == mul_2
        for ir in ir_1 * ir_2:

            if filter_ir_out is not None and ir not in filter_ir_out:
                continue

            i_out = len(irreps_out)
            irreps_out.append((mul, ir))
            instructions += [(i, i, i_out, "uuu", False)]

    tp = FunctionalTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        irrep_normalization=irrep_normalization,
        path_normalization=path_normalization,
    )

    return naive_broadcast_decorator(tp.left_right)(input1, input2)


def FunctionalTensorSquare(irreps_in: Irreps, irreps_out: Irreps, irrep_normalization: str = None, **kwargs):
    if irrep_normalization is None:
        irrep_normalization = "component"

    assert irrep_normalization in ["component", "norm", "none"]

    irreps_in = Irreps(irreps_in)
    irreps_out = Irreps(irreps_out)

    instructions = []
    for i_1, (mul_1, ir_1) in enumerate(irreps_in):
        for i_2, (mul_2, ir_2) in enumerate(irreps_in):
            for i_out, (mul_out, ir_out) in enumerate(irreps_out):
                if ir_out in ir_1 * ir_2:

                    if irrep_normalization == "component":
                        alpha = ir_out.dim
                    elif irrep_normalization == "norm":
                        alpha = ir_1.dim * ir_2.dim
                    elif irrep_normalization == "none":
                        alpha = 1
                    else:
                        raise ValueError(f"irrep_normalization={irrep_normalization}")

                    if i_1 < i_2:
                        instructions += [(i_1, i_2, i_out, "uvw", True, alpha)]
                    elif i_1 == i_2:
                        i = i_1
                        mul = mul_1

                        if mul > 1:
                            instructions += [(i, i, i_out, "u<vw", True, alpha)]

                        if ir_out.l % 2 == 0:
                            if irrep_normalization == "component":
                                if ir_out.l == 0:
                                    alpha = ir_out.dim / (ir_1.dim + 2)
                                else:
                                    alpha = ir_out.dim / 2
                            if irrep_normalization == "norm":
                                if ir_out.l == 0:
                                    alpha = ir_out.dim * ir_1.dim
                                else:
                                    alpha = ir_1.dim * (ir_1.dim + 2) / 2

                            instructions += [(i, i, i_out, "uuw", True, alpha)]

    return FunctionalTensorProduct(irreps_in, irreps_in, irreps_out, instructions, irrep_normalization="none", **kwargs)


class TensorSquare(hk.Module):
    def __init__(self, irreps_out, *, irreps_in=None, init=None):
        super().__init__()

        self.irreps_in = Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_out = Irreps(irreps_out)

        if init is None:
            init = hk.initializers.RandomNormal()
        self.init = init

    def __call__(self, input: IrrepsData) -> IrrepsData:
        if self.irreps_in is not None:
            input = IrrepsData.new(self.irreps_in, input)

        input = input.remove_nones().simplify()

        tp = FunctionalTensorSquare(input.irreps, self.irreps_out)
        ws = [
            hk.get_parameter(
                f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] {tp.irreps_in1[ins.i_in1]},{tp.irreps_in2[ins.i_in2]},{tp.irreps_out[ins.i_out]}",
                shape=ins.path_shape,
                init=self.init,
            )
            for ins in tp.instructions
        ]
        return tp.left_right(ws, input, input)
