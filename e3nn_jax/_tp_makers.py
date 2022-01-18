import functools
import operator
from typing import Any, List, Optional

from e3nn_jax import Irrep, Irreps, TensorProduct


def _prod(xs):
    return functools.reduce(operator.mul, xs, 1)


def FullyConnectedTensorProduct(
    irreps_in1: Any,
    irreps_in2: Any,
    irreps_out: Any,
    in1_var: Optional[List[float]] = None,
    in2_var: Optional[List[float]] = None,
    out_var: Optional[List[float]] = None,
    irrep_normalization: str = 'component',
    path_normalization: str = 'element',
):
    irreps_in1 = Irreps(irreps_in1)
    irreps_in2 = Irreps(irreps_in2)
    irreps_out = Irreps(irreps_out)

    instructions = [
        (i_1, i_2, i_out, 'uvw', True)
        for i_1, (_, ir_1) in enumerate(irreps_in1)
        for i_2, (_, ir_2) in enumerate(irreps_in2)
        for i_out, (_, ir_out) in enumerate(irreps_out)
        if ir_out in ir_1 * ir_2
    ]
    return TensorProduct(irreps_in1, irreps_in2, irreps_out, instructions, in1_var, in2_var, out_var, irrep_normalization, path_normalization)


def FullTensorProduct(
    irreps_in1: Any,
    irreps_in2: Any,
    filter_ir_out=None,
    irrep_normalization: str = 'component',
):
    irreps_in1 = Irreps(irreps_in1)
    irreps_in2 = Irreps(irreps_in2)
    if filter_ir_out is not None:
        filter_ir_out = [Irrep(ir) for ir in filter_ir_out]

    irreps_out = []
    instructions = []
    for i_1, (mul_1, ir_1) in enumerate(irreps_in1):
        for i_2, (mul_2, ir_2) in enumerate(irreps_in2):
            for ir_out in ir_1 * ir_2:

                if filter_ir_out is not None and ir_out not in filter_ir_out:
                    continue

                i_out = len(irreps_out)
                irreps_out.append((mul_1 * mul_2, ir_out))
                instructions += [
                    (i_1, i_2, i_out, 'uvuv', False)
                ]

    irreps_out = Irreps(irreps_out)
    irreps_out, p, _ = irreps_out.sort()

    instructions = [
        (i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions
    ]

    return TensorProduct(irreps_in1, irreps_in2, irreps_out, instructions, irrep_normalization=irrep_normalization)


def ElementwiseTensorProduct(
    irreps_in1: Any,
    irreps_in2: Any,
    filter_ir_out=None,
    irrep_normalization: str = 'component',
    path_normalization: str = 'element',
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

    return TensorProduct(irreps_in1, irreps_in2, irreps_out, instructions, irrep_normalization=irrep_normalization, path_normalization=path_normalization)


def TensorSquare(
    irreps_in: Irreps,
    irreps_out: Irreps,
    irrep_normalization: str = None,
    **kwargs
):
    if irrep_normalization is None:
        irrep_normalization = 'component'

    assert irrep_normalization in ['component', 'norm', 'none']

    irreps_in = Irreps(irreps_in).simplify()
    irreps_out = Irreps(irreps_out).simplify()

    instructions = []
    for i_1, (mul_1, ir_1) in enumerate(irreps_in):
        for i_2, (mul_2, ir_2) in enumerate(irreps_in):
            for i_out, (mul_out, ir_out) in enumerate(irreps_out):
                if ir_out in ir_1 * ir_2:

                    if irrep_normalization == 'component':
                        alpha = ir_out.dim
                    elif irrep_normalization == 'norm':
                        alpha = ir_1.dim * ir_2.dim
                    elif irrep_normalization == 'none':
                        alpha = 1
                    else:
                        raise ValueError(f"irrep_normalization={irrep_normalization}")

                    if i_1 < i_2:
                        instructions += [
                            (i_1, i_2, i_out, 'uvw', True, alpha)
                        ]
                    elif i_1 == i_2:
                        i = i_1
                        mul = mul_1

                        if mul > 1:
                            instructions += [
                                (i, i, i_out, 'u<vw', True, alpha)
                            ]

                        if ir_out.l % 2 == 0:
                            if irrep_normalization == 'component':
                                if ir_out.l == 0:
                                    alpha = ir_out.dim / (ir_1.dim + 2)
                                else:
                                    alpha = ir_out.dim / 2
                            if irrep_normalization == 'norm':
                                if ir_out.l == 0:
                                    alpha = ir_out.dim * ir_1.dim
                                else:
                                    alpha = ir_1.dim * (ir_1.dim + 2) / 2

                            instructions += [
                                (i, i, i_out, 'uuw', True, alpha)
                            ]

    return TensorProduct(irreps_in, irreps_in, irreps_out, instructions, irrep_normalization='none', **kwargs)
