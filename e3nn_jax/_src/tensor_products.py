import itertools
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np

import e3nn_jax as e3nn
from e3nn_jax._src.basic import _align_two_irreps_arrays
from e3nn_jax._src.utils.decorators import overload_for_irreps_without_array
from e3nn_jax._src.utils.dtype import get_pytree_dtype


def _prepare_inputs(input1, input2):
    input1 = e3nn.as_irreps_array(input1)
    input2 = e3nn.as_irreps_array(input2)

    dtype = get_pytree_dtype(input1, input2)
    if dtype.kind == "i":
        dtype = jnp.ones(()).dtype

    input1 = input1.astype(dtype)
    input2 = input2.astype(dtype)

    leading_shape = jnp.broadcast_shapes(input1.shape[:-1], input2.shape[:-1])
    input1 = input1.broadcast_to(leading_shape + (-1,))
    input2 = input2.broadcast_to(leading_shape + (-1,))
    return input1, input2, leading_shape


def _cg_contract(x1: jax.Array, x2: jax.Array, cg: np.ndarray, mode: str) -> jax.Array:
    r"""Contract two irrep chunks with a Clebsch-Gordan tensor.

    Computes, for ``cg`` of shape ``(d1, d2, d3)``:

    * ``mode="uv"``: ``out[..., u, v, k] = sum_ij x1[..., u, i] x2[..., v, j] cg[i, j, k]``
      with ``x1: (..., u, d1)``, ``x2: (..., v, d2)``;
    * ``mode="uu"``: ``out[..., u, k] = sum_ij x1[..., u, i] x2[..., u, j] cg[i, j, k]``
      with ``x1: (..., u, d1)``, ``x2: (..., u, d2)``.

    This is the same maths as ``jnp.einsum("...ui,...vj,ijk->...uvk", ...)`` but evaluated
    in two broadcast-multiply-and-sum stages:

    .. math::

        M_{vik} = \sum_j x2_{vj} C_{ijk}, \qquad out_{uvk} = \sum_i x1_{ui} M_{vik}

    with the CG tensor contracted first against the operand of smaller multiplicity.
    On GPU this lowers to a few fused elementwise/reduction kernels instead of batched
    GEMMs over tiny matrices plus layout transposes, the output is produced directly in
    the ``(..., mul, ir.dim)`` layout, and reverse-mode autodiff yields single fused
    reductions. Arithmetic is plain floating point (no reduced-precision matmul
    units), so it is at least as accurate as the einsum.

    When both multiplicities are 1 (e.g. channels kept on a separate axis) there is
    nothing to broadcast over and the plain einsum is faster, so it is used instead.
    """
    cg = np.asarray(cg)
    if mode == "uv" and x1.shape[-2] < x2.shape[-2]:
        out = _cg_contract(x2, x1, cg.transpose(1, 0, 2), "uv")
        return jnp.swapaxes(out, -2, -3)

    c = jnp.asarray(cg, dtype=x1.dtype)
    if x1.shape[-2] == 1 and x2.shape[-2] == 1:
        # Multiplicity 1 on both sides (e.g. channels on a separate axis): nothing to
        # broadcast over, and the reduction form lets XLA fuse every path into one
        # large output kernel, which is slower than the plain contraction.
        if mode == "uv":
            return jnp.einsum("...ui , ...vj , ijk -> ...uvk", x1, x2, c)
        return jnp.einsum("...ui , ...uj , ijk -> ...uk", x1, x2, c)
    # M[..., v, i, k] = sum_j x2[..., v, j] cg[i, j, k]
    m = jnp.sum(x2[..., :, None, :, None] * c, axis=-2)
    if mode == "uv":
        # out[..., u, v, k] = sum_i x1[..., u, i] M[..., v, i, k]
        return jnp.sum(x1[..., :, None, :, None] * m[..., None, :, :, :], axis=-2)
    if mode == "uu":
        # out[..., u, k] = sum_i x1[..., u, i] M[..., u, i, k]
        return jnp.sum(x1[..., :, :, None] * m, axis=-2)
    raise ValueError(f"unknown mode {mode}")


def _normalized_cg(l1: int, l2: int, l3: int, sqrt_of, dtype) -> np.ndarray:
    """Clebsch-Gordan tensor times ``sqrt(sqrt_of)``, cast to ``dtype``, as concrete numpy.

    Evaluated eagerly (even under ``jit``) with the same jnp operations the einsum-based
    implementation used, so the coefficients are bit-identical.
    """
    with jax.ensure_compile_time_eval():
        cg = e3nn.clebsch_gordan(l1, l2, l3)
        if sqrt_of is not None:
            cg = cg * jnp.sqrt(sqrt_of)
        return np.asarray(jnp.asarray(cg).astype(dtype))


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
        PyTreeDef({'params': {'w[1,0] 2x1e,3x1e': *}})
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
                    if irrep_normalization == "component":
                        sqrt_of = ir_out.dim
                    elif irrep_normalization == "norm":
                        sqrt_of = ir_1.dim * ir_2.dim
                    elif irrep_normalization == "none":
                        sqrt_of = None
                    else:
                        raise ValueError(
                            f"irrep_normalization={irrep_normalization} not supported"
                        )

                    cg = _normalized_cg(ir_1.l, ir_2.l, ir_out.l, sqrt_of, x1.dtype)
                    chunk = _cg_contract(x1, x2, cg, "uv")
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

            if x1 is not None and x2 is not None:
                with jax.ensure_compile_time_eval():
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

                    cg = cg.astype(x1.dtype)

                chunk = None
                if ir_1.l == 0 or ir_2.l == 0:
                    chunk = _scalar_irrep_product(x1, x2, cg, ir_1.l == 0)
                if chunk is None:
                    chunk = _cg_contract(x1, x2, np.asarray(cg), "uu")
            else:
                chunk = None

            chunks.append(chunk)

    return e3nn.from_chunks(irreps_out, chunks, leading_shape, input1.dtype)


def _scalar_irrep_product(
    x1: jax.Array, x2: jax.Array, cg: jax.Array, first_is_scalar: bool
) -> Optional[jax.Array]:
    """Product of a scalar with an irrep as a broadcast multiply.

    When one of the irreps is a scalar, the Clebsch-Gordan tensor is diagonal
    (``cg[0, j, k] = c_k delta_jk`` or ``cg[i, 0, k] = c_k delta_ik``), so
    ``einsum("ui,uj,ijk->uk")`` reduces to ``x1 * x2 * c``. This avoids a
    contraction against a (mostly zero) dense matrix and lets XLA fuse the
    whole operation into a single elementwise kernel. The skipped terms are
    exact zeros, so the result is mathematically identical.

    Returns ``None`` if ``cg`` does not have the expected diagonal structure.
    """
    cg = np.asarray(cg)
    m = cg[0] if first_is_scalar else cg[:, 0, :]
    diag = np.diagonal(m)
    if not np.array_equal(m, np.diag(diag)):
        return None
    return x1 * x2 * jnp.asarray(diag)


@overload_for_irreps_without_array((0,))
def tensor_square(
    input: e3nn.IrrepsArray,
    *,
    irrep_normalization: Optional[str] = None,
    normalized_input: bool = False,
    regroup_output: bool = True,
) -> e3nn.IrrepsArray:
    r"""Tensor product of a `IrrepsArray` with itself.

    Args:
        input (IrrepsArray): Input to be squared
        irrep_normalization (str, optional): Irrep normalization, ``"component"`` or ``"norm"``.
        normalized_input (bool, optional): If True, the input is assumed to be striclty normalized.
            Note that this is different from ``irrep_normalization="norm"`` for which the input is
            of norm 1 in average. Defaults to False.
        regroup_output (bool, optional): If True, the output irreps are regrouped. Defaults to True.

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
    chunks = []

    for i_1, ((mul_1, ir_1), x1) in enumerate(zip(input.irreps, input.chunks)):
        for i_2, ((mul_2, ir_2), x2) in enumerate(zip(input.irreps, input.chunks)):
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
                    irreps_out.append((mul_1 * mul_2, ir_out))

                    if x1 is None or x2 is None:
                        chunks.append(None)
                    else:
                        cg = _square_cg(ir_1.l, ir_2.l, ir_out.l, alpha, x1.dtype)
                        chunk = _cg_contract(x1, x2, cg, "uv")
                        chunk = jnp.reshape(
                            chunk, chunk.shape[:-3] + (mul_1 * mul_2, ir_out.dim)
                        )
                        chunks.append(chunk)

                elif i_1 == i_2:
                    mul = mul_1
                    ir = ir_1
                    x = x1

                    if mul > 1:
                        irreps_out.append((mul * (mul - 1) // 2, ir_out))

                        if x is None:
                            chunks.append(None)
                        else:
                            cg = _square_cg(ir.l, ir.l, ir_out.l, alpha, x.dtype)
                            # Pairs u < v in itertools.combinations order. Gathering
                            # just these pairs is O(mul^2), instead of computing all
                            # mul^2 products and contracting them with a one-hot
                            # (mul, mul, mul (mul - 1) / 2) selector (O(mul^4)).
                            i, j = zip(*itertools.combinations(range(mul), 2))
                            i, j = np.array(i), np.array(j)
                            chunk = _cg_contract(x[..., i, :], x[..., j, :], cg, "uu")
                            chunks.append(chunk)

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

                        irreps_out.append((mul, ir_out))

                        if x is None:
                            chunks.append(None)
                        else:
                            cg = _square_cg(ir.l, ir.l, ir_out.l, alpha, x.dtype)
                            chunk = _cg_contract(x, x, cg, "uu")
                            chunks.append(chunk)

    irreps_out = e3nn.Irreps(irreps_out)

    output = e3nn.from_chunks(irreps_out, chunks, input.shape[:-1], input.dtype)
    output = output.sort()
    if regroup_output:
        output = output.regroup()
    return output


def _square_cg(l1: int, l2: int, l3: int, alpha, dtype) -> np.ndarray:
    """``clebsch_gordan(l1, l2, l3).astype(dtype) * sqrt(alpha)`` as concrete numpy.

    Same operations, in the same order, as the einsum-based implementation.
    """
    with jax.ensure_compile_time_eval():
        cg = jnp.asarray(e3nn.clebsch_gordan(l1, l2, l3).astype(dtype))
        return np.asarray(cg * jnp.sqrt(alpha))
