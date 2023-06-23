from functools import partial
from typing import List, Optional

import jax.numpy as jnp

import e3nn_jax as e3nn
from e3nn_jax._src.basic import _align_two_irreps_arrays
from e3nn_jax._src.utils.decorators import overload_for_irreps_without_array
from e3nn_jax._src.utils.dtype import get_pytree_dtype


def _prepare_inputs(input1, input2):
    input1 = e3nn.as_irreps_array(input1)
    input2 = e3nn.as_irreps_array(input2)

    dtype = get_pytree_dtype(input1, input2)
    input1 = input1.astype(dtype)
    input2 = input2.astype(dtype)

    leading_shape = jnp.broadcast_shapes(input1.shape[:-1], input2.shape[:-1])
    input1 = input1.broadcast_to(leading_shape + (-1,))
    input2 = input2.broadcast_to(leading_shape + (-1,))
    return input1, input2, leading_shape


def _validate_filter_ir_out(filter_ir_out):
    if filter_ir_out is not None:
        if isinstance(filter_ir_out, str):
            filter_ir_out = e3nn.Irreps(filter_ir_out)
        if isinstance(filter_ir_out, e3nn.Irrep):
            filter_ir_out = [filter_ir_out]
        filter_ir_out = [e3nn.Irrep(ir) for ir in filter_ir_out]
    return filter_ir_out


@overload_for_irreps_without_array((0, 1))
def tensor_product(
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    *,
    filter_ir_out: Optional[List[e3nn.Irrep]] = None,
    irrep_normalization: Optional[str] = None,
    regroup_output: bool = True,
) -> e3nn.IrrepsArray:
    """Tensor product reduced into irreps.

    Args:
        input1 (IrrepsArray): First input
        input2 (IrrepsArray): Second input
        filter_ir_out (list of Irrep, optional): Filter the output irreps. Defaults to None.
        irrep_normalization (str, optional): Irrep normalization, ``"component"`` or ``"norm"``. Defaults to ``"component"``.
        regroup_output (bool, optional): Regroup the outputs into irreps. Defaults to True.

    Returns:
        IrrepsArray: Tensor product of the two inputs.

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

        Usage in combination with `haiku.Linear` or `flax.Linear`:

        >>> import jax
        >>> import flax.linen as nn
        >>> linear = e3nn.flax.Linear("3x1e")
        >>> params = linear.init(jax.random.PRNGKey(0), e3nn.tensor_product(x, y))
        >>> jax.tree_util.tree_structure(params)
        PyTreeDef(CustomNode(FrozenDict[('params',)], [{'w[1,0] 2x1e,3x1e': *}]))
        >>> z = linear.apply(params, e3nn.tensor_product(x, y))

        The irreps can be determined without providing input data:

        >>> e3nn.tensor_product("2x1e + 2e", "2e")
        1x0e+3x1e+3x2e+3x3e+1x4e
    """
    input1, input2, leading_shape = _prepare_inputs(input1, input2)
    filter_ir_out = _validate_filter_ir_out(filter_ir_out)

    if irrep_normalization is None:
        irrep_normalization = e3nn.config("irrep_normalization")

    if regroup_output:
        input1 = input1.regroup()
        input2 = input2.regroup()

    irreps_out = []
    chunks = []
    for (mul_1, ir_1), x1 in zip(input1.irreps, input1.chunks):
        for (mul_2, ir_2), x2 in zip(input2.irreps, input2.chunks):
            for ir_out in ir_1 * ir_2:
                if filter_ir_out is not None and ir_out not in filter_ir_out:
                    continue

                irreps_out.append((mul_1 * mul_2, ir_out))

                if x1 is not None and x2 is not None:
                    cg = e3nn.clebsch_gordan(ir_1.l, ir_2.l, ir_out.l)

                    if irrep_normalization == "component":
                        cg = cg * jnp.sqrt(ir_out.dim)
                    elif irrep_normalization == "norm":
                        cg = cg * jnp.sqrt(ir_1.dim * ir_2.dim)
                    elif irrep_normalization == "none":
                        pass
                    else:
                        raise ValueError(
                            f"irrep_normalization={irrep_normalization} not supported"
                        )

                    chunk = jnp.einsum("...ui , ...vj , ijk -> ...uvk", x1, x2, cg)
                    chunk = jnp.reshape(
                        chunk, chunk.shape[:-3] + (mul_1 * mul_2, ir_out.dim)
                    )
                else:
                    chunk = None

                chunks.append(chunk)

    output = e3nn.from_chunks(irreps_out, chunks, leading_shape, input1.dtype)
    output = output.sort()
    if regroup_output:
        output = output.regroup()
    return output


@overload_for_irreps_without_array((0, 1))
def elementwise_tensor_product(
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    *,
    filter_ir_out: Optional[List[e3nn.Irrep]] = None,
    irrep_normalization: Optional[str] = None,
) -> e3nn.IrrepsArray:
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
    input1, input2, leading_shape = _prepare_inputs(input1, input2)
    filter_ir_out = _validate_filter_ir_out(filter_ir_out)

    if irrep_normalization is None:
        irrep_normalization = e3nn.config("irrep_normalization")

    if input1.irreps.num_irreps != input2.irreps.num_irreps:
        raise ValueError(
            "e3nn.elementwise_tensor_product: inputs must have the same number of irreps, "
            f"got {input1.irreps.num_irreps} and {input2.irreps.num_irreps}"
        )

    input1, input2 = _align_two_irreps_arrays(input1, input2)

    irreps_out = []
    chunks = []
    for (mul, ir_1), x1, (_, ir_2), x2 in zip(
        input1.irreps, input1.chunks, input2.irreps, input2.chunks
    ):
        for ir_out in ir_1 * ir_2:
            if filter_ir_out is not None and ir_out not in filter_ir_out:
                continue

            irreps_out.append((mul, ir_out))
            cg = e3nn.clebsch_gordan(ir_1.l, ir_2.l, ir_out.l)

            if irrep_normalization == "component":
                cg = cg * jnp.sqrt(ir_out.dim)
            elif irrep_normalization == "norm":
                cg = cg * jnp.sqrt(ir_1.dim * ir_2.dim)
            elif irrep_normalization == "none":
                pass
            else:
                raise ValueError(
                    f"irrep_normalization={irrep_normalization} not supported"
                )

            chunk = jnp.einsum("...ui , ...uj , ijk -> ...uk", x1, x2, cg)
            chunks.append(chunk)

    return e3nn.from_chunks(irreps_out, chunks, leading_shape, input1.dtype)


@overload_for_irreps_without_array((0,))
def tensor_square(
    input: e3nn.IrrepsArray,
    *,
    irrep_normalization: Optional[str] = None,
    normalized_input: bool = False,
    custom_einsum_jvp: bool = None,
    fused: bool = None,
    regroup_output: bool = True,
) -> e3nn.IrrepsArray:
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
    input = e3nn.as_irreps_array(input)

    if regroup_output:
        input = input.regroup()

    if irrep_normalization is None:
        irrep_normalization = e3nn.config("irrep_normalization")

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
                                raise ValueError(
                                    f"irrep_normalization={irrep_normalization}"
                                )
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
                                raise ValueError(
                                    f"irrep_normalization={irrep_normalization}"
                                )

                        i_out = len(irreps_out)
                        irreps_out.append((mul, ir_out))
                        instructions += [(i, i, i_out, "uuu", False, alpha)]

    irreps_out = e3nn.Irreps(irreps_out)
    irreps_out, p, _ = irreps_out.sort()

    instructions = [
        (i_1, i_2, p[i_out], mode, train, alpha)
        for i_1, i_2, i_out, mode, train, alpha in instructions
    ]

    from e3nn_jax.legacy import FunctionalTensorProduct

    tp = FunctionalTensorProduct(
        input.irreps,
        input.irreps,
        irreps_out,
        instructions,
        irrep_normalization="none",
    )

    f = partial(tp.left_right, fused=fused, custom_einsum_jvp=custom_einsum_jvp)
    for _ in range(input.ndim - 1):
        f = e3nn.utils.vmap(f)

    output = f(input, input)

    if regroup_output:
        output = output.regroup()
    return output
