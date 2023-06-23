from math import sqrt
from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

import e3nn_jax as e3nn
from e3nn_jax import Irreps, IrrepsArray, config
from e3nn_jax._src.utils.sum_tensors import sum_tensors
from e3nn_jax._src.utils.dtype import get_pytree_dtype


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple
    path_weight: float
    weight_std: float


class FunctionalLinear:
    irreps_in: Irreps
    irreps_out: Irreps
    instructions: List[Instruction]
    output_mask: jnp.ndarray

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        instructions: Optional[List[Tuple[int, int]]] = None,
        biases: Optional[Union[List[bool], bool]] = None,
        path_normalization: Union[str, float] = None,
        gradient_normalization: Union[str, float] = None,
    ):
        if path_normalization is None:
            path_normalization = config("path_normalization")
        if isinstance(path_normalization, str):
            path_normalization = {"element": 0.0, "path": 1.0}[path_normalization]

        if gradient_normalization is None:
            gradient_normalization = config("gradient_normalization")
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[
                gradient_normalization
            ]

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
                weight_std=1,
            )
            for i_in, i_out in instructions
        ]

        def alpha(this):
            x = irreps_in[this.i_in].mul ** path_normalization * sum(
                irreps_in[other.i_in].mul ** (1.0 - path_normalization)
                for other in instructions
                if other.i_out == this.i_out
            )
            return 1 / x if x > 0 else 1.0

        instructions = [
            Instruction(
                i_in=ins.i_in,
                i_out=ins.i_out,
                path_shape=ins.path_shape,
                path_weight=sqrt(alpha(ins)) ** gradient_normalization,
                weight_std=sqrt(alpha(ins)) ** (1.0 - gradient_normalization),
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
                path_weight=1.0,
                weight_std=0.0,
            )
            for i_out, (bias, mul_ir) in enumerate(zip(biases, irreps_out))
            if bias
        ]

        with jax.ensure_compile_time_eval():
            if irreps_out.dim > 0:
                output_mask = jnp.concatenate(
                    [
                        jnp.ones(mul_ir.dim, bool)
                        if any(
                            (ins.i_out == i_out) and (0 not in ins.path_shape)
                            for ins in instructions
                        )
                        else jnp.zeros(mul_ir.dim, bool)
                        for i_out, mul_ir in enumerate(irreps_out)
                    ]
                )
            else:
                output_mask = jnp.ones(0, bool)

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.instructions = instructions
        self.output_mask = output_mask

    @property
    def num_weights(self) -> int:
        return sum(np.prod(i.path_shape) for i in self.instructions)

    def aggregate_paths(self, paths, output_shape, output_dtype) -> IrrepsArray:
        output = [
            sum_tensors(
                [
                    out
                    for ins, out in zip(self.instructions, paths)
                    if ins.i_out == i_out
                ],
                shape=output_shape
                + (
                    mul_ir_out.mul,
                    mul_ir_out.ir.dim,
                ),
                empty_return_none=True,
            )
            for i_out, mul_ir_out in enumerate(self.irreps_out)
        ]
        return e3nn.from_chunks(self.irreps_out, output, output_shape, output_dtype)

    def split_weights(self, weights: jnp.ndarray) -> List[jnp.ndarray]:
        ws = []
        cursor = 0
        for i in self.instructions:
            ws += [
                weights[cursor : cursor + np.prod(i.path_shape)].reshape(i.path_shape)
            ]
            cursor += np.prod(i.path_shape)
        return ws

    def __call__(
        self, ws: Union[List[jnp.ndarray], jnp.ndarray], input: IrrepsArray
    ) -> IrrepsArray:
        input = input.rechunk(self.irreps_in)
        if input.ndim != 1:
            raise ValueError(
                f"FunctionalLinear does not support broadcasting, input shape is {input.shape}"
            )

        if not isinstance(ws, list):
            ws = self.split_weights(ws)

        paths = [
            ins.path_weight * w
            if ins.i_in == -1
            else (
                None
                if input.chunks[ins.i_in] is None
                else ins.path_weight
                * jnp.einsum("uw,ui->wi", w, input.chunks[ins.i_in])
            )
            for ins, w in zip(self.instructions, ws)
        ]
        return self.aggregate_paths(paths, input.shape[:-1], input.dtype)

    def matrix(self, ws: List[jnp.ndarray]) -> jnp.ndarray:
        r"""Compute the matrix representation of the linear operator.

        Args:
            ws: List of weights.

        Returns:
            The matrix representation of the linear operator. The matrix is shape ``(irreps_in.dim, irreps_out.dim)``.
        """
        dtype = get_pytree_dtype(ws)
        output = jnp.zeros((self.irreps_in.dim, self.irreps_out.dim), dtype)
        for ins, w in zip(self.instructions, ws):
            assert ins.i_in != -1
            mul_in, ir_in = self.irreps_in[ins.i_in]
            mul_out, ir_out = self.irreps_out[ins.i_out]
            output = output.at[
                self.irreps_in.slices()[ins.i_in], self.irreps_out.slices()[ins.i_out]
            ].add(
                ins.path_weight
                * jnp.einsum("uw,ij->uiwj", w, jnp.eye(ir_in.dim, dtype=dtype)).reshape(
                    (mul_in * ir_in.dim, mul_out * ir_out.dim)
                )
            )
        return output


def linear_vanilla(
    input: IrrepsArray,
    linear: FunctionalLinear,
    get_parameter: Callable[[str, Tuple[int, ...], float, Any], jnp.ndarray],
) -> IrrepsArray:
    w = [
        get_parameter(
            f"b[{ins.i_out}] {linear.irreps_out[ins.i_out]}"
            if ins.i_in == -1
            else f"w[{ins.i_in},{ins.i_out}] {linear.irreps_in[ins.i_in]},{linear.irreps_out[ins.i_out]}",
            ins.path_shape,
            ins.weight_std,
            input.dtype,
        )
        for ins in linear.instructions
    ]
    f = lambda x: linear(w, x)
    for _ in range(input.ndim - 1):
        f = e3nn.utils.vmap(f)

    return f(input)


def linear_indexed(
    input: IrrepsArray,
    lin: FunctionalLinear,
    get_parameter: Callable[[str, Tuple[int, ...], float, Any], jnp.ndarray],
    weights: jnp.ndarray,
    num_indexed_weights: int,
) -> IrrepsArray:
    shape = jnp.broadcast_shapes(input.shape[:-1], weights.shape)
    input = input.broadcast_to(shape + (-1,))
    weights = jnp.broadcast_to(weights, shape)

    w = [
        get_parameter(
            f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}"
            if ins.i_in == -1
            else f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}",
            (num_indexed_weights,) + ins.path_shape,
            ins.weight_std,
            input.dtype,
        )
        for ins in lin.instructions
    ]  # List of shape (num_weights, *path_shape)
    w = [wi[weights] for wi in w]  # List of shape (..., *path_shape)

    f = lin
    for _ in range(input.ndim - 1):
        f = e3nn.utils.vmap(f)
    return f(w, input)


def linear_mixed(
    input: IrrepsArray,
    lin: FunctionalLinear,
    get_parameter: Callable[[str, Tuple[int, ...], float, Any], jnp.ndarray],
    weights: jnp.ndarray,
    gradient_normalization: float,
) -> IrrepsArray:
    shape = jnp.broadcast_shapes(input.shape[:-1], weights.shape[:-1])
    input = input.broadcast_to(shape + (-1,))  # (..., irreps)
    weights = jnp.broadcast_to(weights, shape + weights.shape[-1:])  # (..., d)

    # Should be equivalent to the last layer of e3nn.MultiLayerPerceptron
    d = weights.shape[-1]
    alpha = 1 / d
    stddev = jnp.sqrt(alpha) ** (1.0 - gradient_normalization)

    w = [
        get_parameter(
            f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}"
            if ins.i_in == -1
            else f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}",
            (d,) + ins.path_shape,
            stddev * ins.weight_std,
            input.dtype,
        )
        for ins in lin.instructions
    ]  # List of shape (d, *path_shape)
    weights = weights.astype(input.array.dtype)
    w = [
        jnp.sqrt(alpha) ** gradient_normalization
        * jax.lax.dot_general(weights, wi, (((weights.ndim - 1,), (0,)), ((), ())))
        for wi in w
    ]  # List of shape (..., *path_shape)

    f = lin
    for _ in range(input.ndim - 1):
        f = e3nn.utils.vmap(f)
    return f(w, input)  # (..., irreps)


def linear_mixed_per_channel(
    input: IrrepsArray,
    lin: FunctionalLinear,
    get_parameter: Callable[[str, Tuple[int, ...], float, Any], jnp.ndarray],
    weights: jnp.ndarray,
    gradient_normalization: float,
) -> IrrepsArray:
    shape = jnp.broadcast_shapes(input.shape[:-2], weights.shape[:-1])
    input = input.broadcast_to(shape + input.shape[-2:])  # (..., num_channels, irreps)
    weights = jnp.broadcast_to(weights, shape + weights.shape[-1:])  # (..., d)
    nc = input.shape[-2]

    # Should be equivalent to the last layer of e3nn.MultiLayerPerceptron
    d = weights.shape[-1]
    alpha = 1 / d
    stddev = jnp.sqrt(alpha) ** (1.0 - gradient_normalization)

    w = [
        get_parameter(
            f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}"
            if ins.i_in == -1
            else f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}",
            (d, nc) + ins.path_shape,
            stddev * ins.weight_std,
            input.dtype,
        )
        for ins in lin.instructions
    ]  # List of shape (d, num_channels, *path_shape)
    weights = weights.astype(input.array.dtype)
    w = [
        jnp.sqrt(alpha) ** gradient_normalization
        * jax.lax.dot_general(weights, wi, (((weights.ndim - 1,), (0,)), ((), ())))
        for wi in w
    ]  # List of shape (..., num_channels, *path_shape)

    f = lin
    for _ in range(input.ndim - 1):
        f = e3nn.utils.vmap(f)
    return f(w, input)  # (..., num_channels, irreps)
