from typing import List, NamedTuple, Optional, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp

from e3nn_jax import Irreps, IrrepsArray, config
from math import sqrt

from e3nn_jax._src.core_tensor_product import _sum_tensors


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
            gradient_normalization = {"element": 0.0, "path": 1.0}[gradient_normalization]

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
                irreps_in[other.i_in].mul ** (1.0 - path_normalization) for other in instructions if other.i_out == this.i_out
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
            Instruction(i_in=-1, i_out=i_out, path_shape=(mul_ir.dim,), path_weight=1.0, weight_std=0.0)
            for i_out, (bias, mul_ir) in enumerate(zip(biases, irreps_out))
            if bias
        ]

        with jax.ensure_compile_time_eval():
            if irreps_out.dim > 0:
                output_mask = jnp.concatenate(
                    [
                        jnp.ones(mul_ir.dim)
                        if any((ins.i_out == i_out) and (0 not in ins.path_shape) for ins in instructions)
                        else jnp.zeros(mul_ir.dim)
                        for i_out, mul_ir in enumerate(irreps_out)
                    ]
                )
            else:
                output_mask = jnp.ones(0)

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.instructions = instructions
        self.output_mask = output_mask

    def aggregate_paths(self, paths, output_shape) -> IrrepsArray:
        output = [
            _sum_tensors(
                [out for ins, out in zip(self.instructions, paths) if ins.i_out == i_out],
                shape=output_shape
                + (
                    mul_ir_out.mul,
                    mul_ir_out.ir.dim,
                ),
                empty_return_none=True,
            )
            for i_out, mul_ir_out in enumerate(self.irreps_out)
        ]
        return IrrepsArray.from_list(self.irreps_out, output, output_shape)

    def __call__(self, ws: List[jnp.ndarray], input: IrrepsArray) -> IrrepsArray:
        input = input.convert(self.irreps_in)
        if input.ndim != 1:
            raise ValueError(f"FunctionalLinear does not support broadcasting, input shape is {input.shape}")

        paths = [
            ins.path_weight * w
            if ins.i_in == -1
            else (None if input.list[ins.i_in] is None else ins.path_weight * jnp.einsum("uw,ui->wi", w, input.list[ins.i_in]))
            for ins, w in zip(self.instructions, ws)
        ]
        return self.aggregate_paths(paths, input.shape[:-1])

    def matrix(self, ws: List[jnp.ndarray]) -> jnp.ndarray:
        r"""Compute the matrix representation of the linear operator.

        Args:
            ws: List of weights.

        Returns:
            The matrix representation of the linear operator. The matrix is shape ``(irreps_in.dim, irreps_out.dim)``.
        """
        output = jnp.zeros((self.irreps_in.dim, self.irreps_out.dim))
        for ins, w in zip(self.instructions, ws):
            assert ins.i_in != -1
            mul_in, ir_in = self.irreps_in[ins.i_in]
            mul_out, ir_out = self.irreps_out[ins.i_out]
            output = output.at[self.irreps_in.slices()[ins.i_in], self.irreps_out.slices()[ins.i_out]].add(
                ins.path_weight
                * jnp.einsum("uw,ij->uiwj", w, jnp.eye(ir_in.dim)).reshape((mul_in * ir_in.dim, mul_out * ir_out.dim))
            )
        return output


class Linear(hk.Module):
    r"""Equivariant Linear Haiku Module.

    Args:
        irreps_out (`e3nn_jax.Irreps`): output representations
        channel_out (optional int): if specified, the last axis is assumed to be the channel axis
            and is mixed with the irreps.

    Example:
        >>> import e3nn_jax as e3nn
        >>> @hk.without_apply_rng
        ... @hk.transform
        ... def linear(x):
        ...     return e3nn.Linear("0e + 1o")(x)
        >>> x = e3nn.IrrepsArray("1o + 2x0e", jnp.ones(5))
        >>> params = linear.init(jax.random.PRNGKey(0), x)
        >>> y = linear.apply(params, x)
    """

    def __init__(
        self,
        irreps_out: Irreps,
        channel_out: int = None,
        *,
        irreps_in: Optional[Irreps] = None,
        biases: bool = False,
        path_normalization: Union[str, float] = None,
        gradient_normalization: Union[str, float] = None,
    ):
        super().__init__()

        self.irreps_in = Irreps(irreps_in) if irreps_in is not None else None
        self.channel_out = channel_out
        self.irreps_out = Irreps(irreps_out)
        self.instructions = None
        self.biases = biases
        self.path_normalization = path_normalization
        self.gradient_normalization = gradient_normalization

    def __call__(self, input: IrrepsArray) -> IrrepsArray:
        if self.irreps_in is not None:
            input = input.convert(self.irreps_in)

        input = input.remove_nones().simplify()
        output_irreps = self.irreps_out.simplify()
        if self.channel_out is not None:
            input = input.repeat_mul_by_last_axis()
            output_irreps = Irreps([(self.channel_out * mul, ir) for mul, ir in output_irreps])

        lin = FunctionalLinear(
            input.irreps,
            output_irreps,
            self.instructions,
            biases=self.biases,
            path_normalization=self.path_normalization,
            gradient_normalization=self.gradient_normalization,
        )
        w = [
            hk.get_parameter(
                f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}",
                shape=ins.path_shape,
                init=hk.initializers.RandomNormal(stddev=ins.weight_std),
            )
            if ins.i_in == -1
            else hk.get_parameter(
                f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}",
                shape=ins.path_shape,
                init=hk.initializers.RandomNormal(stddev=ins.weight_std),
            )
            for ins in lin.instructions
        ]
        f = lambda x: lin(w, x)
        for _ in range(input.ndim - 1):
            f = jax.vmap(f)
        output = f(input)

        if self.channel_out is not None:
            output = output.factor_mul_to_last_axis(self.channel_out)
        return output.convert(self.irreps_out)
