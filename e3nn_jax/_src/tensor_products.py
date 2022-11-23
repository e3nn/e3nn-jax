import warnings
from functools import partial
from typing import List, Optional

import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import FunctionalTensorProduct, Irrep, Irreps, IrrepsArray, config
from e3nn_jax._src.util.decorators import overload_for_irreps_without_array


def naive_broadcast_decorator(func):
    def wrapper(*args):
        leading_shape = jnp.broadcast_shapes(*(arg.shape[:-1] for arg in args))
        args = [arg.broadcast_to(leading_shape + (-1,)) for arg in args]
        f = func
        for _ in range(len(leading_shape)):
            f = jax.vmap(f)
        return f(*args)

    return wrapper


@overload_for_irreps_without_array((0, 1))
def tensor_product(
    input1: IrrepsArray,
    input2: IrrepsArray,
    *,
    filter_ir_out: Optional[List[Irrep]] = None,
    irrep_normalization: Optional[str] = None,
    custom_einsum_jvp: bool = None,
    fused: bool = None,
    regroup_output: bool = True,
) -> IrrepsArray:
    """Tensor product reduced into irreps.

    Args:
        input1 (IrrepsArray): First input
        input2 (IrrepsArray): Second input
        filter_ir_out (list of Irrep, optional): Filter the output irreps. Defaults to None.
        irrep_normalization (str, optional): Irrep normalization, ``"component"`` or ``"norm"``. Defaults to ``"component"``.
        regroup_output (bool, optional): Regroup the outputs into irreps. Defaults to True.

    Returns:
        IrrepsArray: Tensor product of the two inputs.
            The irreps are sorted (``0e, 0o, 1o, 1e, 2e, 2o, ...``) but not simplified, see example below.

    Examples:
        >>> jnp.set_printoptions(precision=2, suppress=True)
        >>> import e3nn_jax as e3nn
        >>> x = e3nn.IrrepsArray("2x0e + 1o", jnp.arange(5))
        >>> y = e3nn.IrrepsArray("0o + 2o", jnp.arange(6))
        >>> e3nn.tensor_product(x, y)
        2x0o+2x1e+1x2e+2x2o+1x3e
        [  0.     0.     0.     0.     0.    -1.9   16.65  14.83   7.35 -12.57
           0.    -0.66   4.08   0.     0.     0.     0.     0.     1.     2.
           3.     4.     5.     9.9   10.97   9.27  -1.97  12.34  15.59  12.73]

        Usage in combination with `Linear`:

        >>> @hk.without_apply_rng
        ... @hk.transform
        ... def fully_connected_tensor_product(x, y):
        ...     return e3nn.Linear("3x1e")(e3nn.tensor_product(x, y))
        >>> params = fully_connected_tensor_product.init(jax.random.PRNGKey(0), x, y)
        >>> jax.tree_util.tree_structure(params)
        PyTreeDef({'linear': {'w[1,0] 2x1e,3x1e': *}})
        >>> z = fully_connected_tensor_product.apply(params, x, y)

        The irreps can be determined without providing input data:

        >>> e3nn.tensor_product("2x1e + 2e", "2e")
        1x0e+3x1e+3x2e+3x3e+1x4e
    """
    if regroup_output:
        input1 = input1.regroup()
        input2 = input2.regroup()

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

    output = naive_broadcast_decorator(partial(tp.left_right, fused=fused, custom_einsum_jvp=custom_einsum_jvp))(
        input1, input2
    )
    if regroup_output:
        output = output.regroup()
    return output


@overload_for_irreps_without_array((0, 1))
def elementwise_tensor_product(
    input1: IrrepsArray,
    input2: IrrepsArray,
    *,
    filter_ir_out=None,
    irrep_normalization: str = None,
) -> IrrepsArray:
    r"""Elementwise tensor product of two `IrrepsArray`.

    Args:
        input1 (IrrepsArray): First input
        input2 (IrrepsArray): Second input with the same number of irreps as ``input1``,
            ``input1.irreps.num_irreps == input2.irreps.num_irreps``.
        filter_ir_out (list of Irrep, optional): Filter the output irreps. Defaults to None.
        irrep_normalization (str, optional): Irrep normalization, ``"component"`` or ``"norm"``. Defaults to ``"component"``.

    Returns:
        IrrepsArray: Elementwise tensor product of the two inputs.
            The irreps are not sorted and not simplified.

    Examples:
        >>> jnp.set_printoptions(precision=2, suppress=True)
        >>> import e3nn_jax as e3nn
        >>> x = e3nn.IrrepsArray("2x0e + 1o", jnp.arange(5))
        >>> y = e3nn.IrrepsArray("1e + 0o + 0o", jnp.arange(5))
        >>> e3nn.elementwise_tensor_product(x, y)
        1x1e+1x0o+1x1e [ 0.  0.  0.  3.  8. 12. 16.]
    """
    if filter_ir_out is not None:
        filter_ir_out = [Irrep(ir) for ir in filter_ir_out]

    if input1.irreps.num_irreps != input2.irreps.num_irreps:
        raise ValueError(f"Number of irreps must be the same, got {input1.irreps.num_irreps} and {input2.irreps.num_irreps}")

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

    input1 = input1._convert(irreps_in1)
    input2 = input2._convert(irreps_in2)

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
    )

    return naive_broadcast_decorator(tp.left_right)(input1, input2)


@overload_for_irreps_without_array((0,))
def tensor_square(
    input: IrrepsArray,
    *,
    irrep_normalization: Optional[str] = None,
    normalized_input: bool = False,
    custom_einsum_jvp: bool = None,
    fused: bool = None,
    regroup_output: bool = True,
) -> IrrepsArray:
    r"""Tensor product of a `IrrepsArray` with itself.

    Args:
        input (IrrepsArray): Input to be squared
        irrep_normalization (str, optional): Irrep normalization, ``"component"`` or ``"norm"``.
        normalized_input (bool, optional): If True, the input is assumed to be striclty normalized.
            Note that this is different from ``irrep_normalization="norm"`` for which the input is
            of norm 1 in average. Defaults to False.
        custom_einsum_jvp (bool, optional): If True, use a custom implementation of the jvp of einsum.
        fused (bool, optional): If True, use a fused implementation of the tensor product.

    Returns:
        IrrepsArray: Tensor product of the input with itself.

    Examples:
        >>> jnp.set_printoptions(precision=2, suppress=True)
        >>> import e3nn_jax as e3nn
        >>> x = e3nn.IrrepsArray("0e + 1o", jnp.array([10, 1, 2, 3.0]))
        >>> e3nn.tensor_square(x)
        2x0e+1x1o+1x2e [57.74  3.61 10.   20.   30.    3.    2.   -0.58  6.    4.  ]

        >>> e3nn.tensor_square(x, normalized_input=True)
        2x0e+1x1o+1x2e [100.    14.    17.32  34.64  51.96  11.62   7.75  -2.24  23.24  15.49]
    """
    if regroup_output:
        input = input.regroup()

    if irrep_normalization is None:
        irrep_normalization = config("irrep_normalization")

    assert irrep_normalization in ["component", "norm", "none"]

    irreps_out = []

    instructions = []
    for i_1, (mul_1, ir_1) in enumerate(input.irreps):
        for i_2, (mul_2, ir_2) in enumerate(input.irreps):
            for ir_out in ir_1 * ir_2:

                if normalized_input:
                    if irrep_normalization == "component":
                        alpha = ir_1.dim * ir_2.dim * ir_out.dim
                    elif irrep_normalization == "norm":
                        alpha = ir_1.dim * ir_2.dim
                    elif irrep_normalization == "none":
                        alpha = 1
                    else:
                        raise ValueError(f"irrep_normalization={irrep_normalization}")
                else:
                    if irrep_normalization == "component":
                        alpha = ir_out.dim
                    elif irrep_normalization == "norm":
                        alpha = ir_1.dim * ir_2.dim
                    elif irrep_normalization == "none":
                        alpha = 1
                    else:
                        raise ValueError(f"irrep_normalization={irrep_normalization}")

                if i_1 < i_2:
                    i_out = len(irreps_out)
                    irreps_out.append((mul_1 * mul_2, ir_out))
                    instructions += [(i_1, i_2, i_out, "uvuv", False, alpha)]
                elif i_1 == i_2:
                    i = i_1
                    mul = mul_1

                    if mul > 1:
                        i_out = len(irreps_out)
                        irreps_out.append((mul * (mul - 1) // 2, ir_out))
                        instructions += [(i, i, i_out, "uvu<v", False, alpha)]

                    if ir_out.l % 2 == 0:
                        if normalized_input:
                            if irrep_normalization == "component":
                                if ir_out.l == 0:
                                    alpha = ir_out.dim * ir_1.dim
                                else:
                                    alpha = ir_1.dim * (ir_1.dim + 2) / 2 * ir_out.dim
                            elif irrep_normalization == "norm":
                                if ir_out.l == 0:
                                    alpha = ir_out.dim * ir_1.dim
                                else:
                                    alpha = ir_1.dim * (ir_1.dim + 2) / 2
                            elif irrep_normalization == "none":
                                alpha = 1
                            else:
                                raise ValueError(f"irrep_normalization={irrep_normalization}")
                        else:
                            if irrep_normalization == "component":
                                if ir_out.l == 0:
                                    alpha = ir_out.dim / (ir_1.dim + 2)
                                else:
                                    alpha = ir_out.dim / 2
                            elif irrep_normalization == "norm":
                                if ir_out.l == 0:
                                    alpha = ir_1.dim * ir_2.dim / (ir_1.dim + 2)
                                else:
                                    alpha = ir_1.dim * ir_2.dim / 2
                            elif irrep_normalization == "none":
                                alpha = 1
                            else:
                                raise ValueError(f"irrep_normalization={irrep_normalization}")

                        i_out = len(irreps_out)
                        irreps_out.append((mul, ir_out))
                        instructions += [(i, i, i_out, "uuu", False, alpha)]

    irreps_out = Irreps(irreps_out)
    irreps_out, p, _ = irreps_out.sort()

    instructions = [(i_1, i_2, p[i_out], mode, train, alpha) for i_1, i_2, i_out, mode, train, alpha in instructions]

    tp = FunctionalTensorProduct(
        input.irreps,
        input.irreps,
        irreps_out,
        instructions,
        irrep_normalization="none",
    )

    output = naive_broadcast_decorator(partial(tp.left_right, fused=fused, custom_einsum_jvp=custom_einsum_jvp))(input, input)
    if regroup_output:
        output = output.regroup()
    return output


# Deprecated functions:


def FunctionalFullyConnectedTensorProduct(
    irreps_in1: Irreps,
    irreps_in2: Irreps,
    irreps_out: Irreps,
    in1_var: Optional[List[float]] = None,
    in2_var: Optional[List[float]] = None,
    out_var: Optional[List[float]] = None,
    irrep_normalization: str = None,
    path_normalization: str = None,
    gradient_normalization: str = None,
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


class FullyConnectedTensorProduct(hk.Module):
    def __init__(self, irreps_out, *, irreps_in1=None, irreps_in2=None):
        super().__init__()

        warnings.warn(
            "e3nn.FullyConnectedTensorProduct is deprecated. Use e3nn.tensor_product followed by e3nn.Linear instead.",
            DeprecationWarning,
        )

        self.irreps_out = Irreps(irreps_out)
        self.irreps_in1 = Irreps(irreps_in1) if irreps_in1 is not None else None
        self.irreps_in2 = Irreps(irreps_in2) if irreps_in2 is not None else None

    def __call__(self, x1: IrrepsArray, x2: IrrepsArray, **kwargs) -> IrrepsArray:
        if self.irreps_in1 is not None:
            x1 = x1._convert(self.irreps_in1)
        if self.irreps_in2 is not None:
            x2 = x2._convert(self.irreps_in2)

        x1 = x1.remove_nones().simplify()
        x2 = x2.remove_nones().simplify()

        tp = FunctionalFullyConnectedTensorProduct(x1.irreps, x2.irreps, self.irreps_out.simplify())
        ws = [
            hk.get_parameter(
                (
                    f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] "
                    f"{tp.irreps_in1[ins.i_in1]},{tp.irreps_in2[ins.i_in2]},{tp.irreps_out[ins.i_out]}"
                ),
                shape=ins.path_shape,
                init=hk.initializers.RandomNormal(stddev=ins.weight_std),
            )
            for ins in tp.instructions
        ]
        f = naive_broadcast_decorator(lambda x1, x2: tp.left_right(ws, x1, x2, **kwargs))
        output = f(x1, x2)
        return output._convert(self.irreps_out)


def full_tensor_product(
    input1: IrrepsArray,
    input2: IrrepsArray,
    filter_ir_out: Optional[List[Irrep]] = None,
    irrep_normalization: Optional[str] = None,
):
    warnings.warn("e3nn.full_tensor_product is deprecated. Use e3nn.tensor_product instead.", DeprecationWarning)

    return tensor_product(input1, input2, filter_ir_out=filter_ir_out, irrep_normalization=irrep_normalization)


def FunctionalTensorSquare(irreps_in: Irreps, irreps_out: Irreps, irrep_normalization: str = None, **kwargs):
    if irrep_normalization is None:
        irrep_normalization = config("irrep_normalization")

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
    """Tensor product of a tensor with itself.

    The difference with :func:`e3nn_jax.reduced_symmetric_tensor_product_basis` and :func:`e3nn_jax.SymmetricTensorProduct`
    is the proper normalization.

    Args:
        irreps_out (Irreps): Irreps of the output
    """

    def __init__(self, irreps_out: Irreps, *, irreps_in=None, init=None):
        super().__init__()

        warnings.warn(
            "TensorSquare is deprecated, use e3nn.tensor_square followed by e3nn.Linear instead",
            DeprecationWarning,
        )

        self.irreps_in = Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_out = Irreps(irreps_out)

        if init is None:
            init = hk.initializers.RandomNormal
        self.init = init

    def __call__(self, input: IrrepsArray) -> IrrepsArray:
        if self.irreps_in is not None:
            input = input._convert(self.irreps_in)

        input = input.remove_nones().simplify()

        tp = FunctionalTensorSquare(input.irreps, self.irreps_out.simplify())
        ws = [hk.get_parameter(str(ins), shape=ins.path_shape, init=self.init(ins.weight_std)) for ins in tp.instructions]
        f = naive_broadcast_decorator(lambda x: tp.left_right(ws, x, x))
        output = f(input)
        return output._convert(self.irreps_out)
