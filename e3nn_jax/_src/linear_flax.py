from typing import Optional, Union

import flax
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn

from .linear import FunctionalLinear


class Linear(flax.linen.Module):
    r"""Equivariant Linear Flax module"""
    irreps_out: e3nn.Irreps
    channel_out: Optional[int] = None
    gradient_normalization: Optional[Union[float, str]] = None
    path_normalization: Optional[Union[float, str]] = None
    biases: bool = False
    num_indexed_weights: Optional[int] = None

    @flax.linen.compact
    def __call__(self, weights_or_input, input_or_none=None) -> e3nn.IrrepsArray:
        """Apply the linear operator.

        Args:
            weights (optional IrrepsArray or jnp.ndarray): scalar weights that are contracted with free parameters.
                An array of shape ``(..., contracted_axis)``. Broadcasting with `input` is supported.
            input (IrrepsArray): input irreps-array of shape ``(..., [channel_in,] irreps_in.dim)``.
                Broadcasting with `weights` is supported.

        Returns:
            IrrepsArray: output irreps-array of shape ``(..., [channel_out,] irreps_out.dim)``.
                Properly normalized assuming that the weights and input are properly normalized.
        """
        if input_or_none is None:
            weights = None
            input: e3nn.IrrepsArray = weights_or_input
        else:
            weights: jnp.ndarray = weights_or_input
            input: e3nn.IrrepsArray = input_or_none
        del weights_or_input, input_or_none

        input = input.remove_nones().regroup()
        output_irreps = e3nn.Irreps(self.irreps_out).simplify()

        if self.channel_out is not None:
            # assert not self.weights_per_channel
            input = input.axis_to_mul()
            output_irreps = self.channel_out * output_irreps

        lin = FunctionalLinear(
            input.irreps,
            output_irreps,
            biases=self.biases,
            path_normalization=self.path_normalization,
            gradient_normalization=self.gradient_normalization,
        )

        if weights is None:
            # assert not self.weights_per_channel  # Not implemented yet
            w = [
                self.param(
                    f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}",
                    flax.linen.initializers.normal(stddev=ins.weight_std),
                    ins.path_shape,
                    input.dtype,
                )
                if ins.i_in == -1
                else self.param(
                    f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}",
                    flax.linen.initializers.normal(stddev=ins.weight_std),
                    ins.path_shape,
                    input.dtype,
                )
                for ins in lin.instructions
            ]
            f = lambda x: lin(w, x)
            for _ in range(input.ndim - 1):
                f = jax.vmap(f)
            output = f(input)
        else:
            if isinstance(weights, e3nn.IrrepsArray):
                if not weights.irreps.is_scalar():
                    raise ValueError("weights must be scalar")
                weights = weights.array

            if weights.dtype.kind == "i" and self.num_indexed_weights is not None:
                # assert not self.weights_per_channel  # Not implemented yet

                shape = jnp.broadcast_shapes(input.shape[:-1], weights.shape)
                input = input.broadcast_to(shape + (-1,))
                weights = jnp.broadcast_to(weights, shape)

                w = [
                    self.param(
                        f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}",
                        flax.linen.initializers.normal(stddev=ins.weight_std),
                        (self.num_indexed_weights,) + ins.path_shape,
                        input.dtype,
                    )
                    if ins.i_in == -1
                    else self.param(
                        f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}",
                        flax.linen.initializers.normal(stddev=ins.weight_std),
                        (self.num_indexed_weights,) + ins.path_shape,
                        input.dtype,
                    )
                    for ins in lin.instructions
                ]  # List of shape (num_weights, *path_shape)
                w = [wi[weights] for wi in w]  # List of shape (..., *path_shape)

                f = lin
                for _ in range(input.ndim - 1):
                    f = jax.vmap(f)
                output = f(w, input)

            elif weights.dtype.kind in "fc" and self.num_indexed_weights is None:

                gradient_normalization = self.gradient_normalization
                if gradient_normalization is None:
                    gradient_normalization = e3nn.config("gradient_normalization")
                if isinstance(gradient_normalization, str):
                    gradient_normalization = {"element": 0.0, "path": 1.0}[gradient_normalization]

                if False:
                    pass
                # if self.weights_per_channel:
                #     shape = jnp.broadcast_shapes(input.shape[:-2], weights.shape[:-1])
                #     input = input.broadcast_to(shape + input.shape[-2:])
                #     weights = jnp.broadcast_to(weights, shape + weights.shape[-1:])
                #     nc = input.shape[-2]

                #     weights = weights.astype(input.array.dtype)

                #     # Should be equivalent to the last layer of e3nn.MultiLayerPerceptron
                #     d = weights.shape[-1]
                #     alpha = 1 / d
                #     stddev = jnp.sqrt(alpha) ** (1.0 - gradient_normalization)

                #     w = [
                #         self.param(
                #             f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}",
                #             flax.linen.initializers.normal(stddev=stddev * ins.weight_std),
                #             (d, nc) + ins.path_shape,
                #             input.dtype,
                #         )
                #         if ins.i_in == -1
                #         else self.param(
                #             f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}",
                #             flax.linen.initializers.normal(stddev=stddev * ins.weight_std),
                #             (d, nc) + ins.path_shape,
                #             input.dtype,
                #         )
                #         for ins in lin.instructions
                #     ]  # List of shape (d, *path_shape)
                #     w = [
                #         jnp.sqrt(alpha) ** gradient_normalization
                #         * jax.lax.dot_general(weights, wi.astype(input.array.dtype), (((weights.ndim - 1,), (0,)), ((), ())))
                #         for wi in w
                #     ]  # List of shape (..., num_channels, *path_shape)

                #     f = lin
                #     for _ in range(input.ndim - 1):
                #         f = jax.vmap(f)
                #     output = f(w, input)
                else:
                    shape = jnp.broadcast_shapes(input.shape[:-1], weights.shape[:-1])
                    input = input.broadcast_to(shape + (-1,))
                    weights = jnp.broadcast_to(weights, shape + weights.shape[-1:])

                    weights = weights.astype(input.array.dtype)

                    # Should be equivalent to the last layer of e3nn.MultiLayerPerceptron
                    d = weights.shape[-1]
                    alpha = 1 / d
                    stddev = jnp.sqrt(alpha) ** (1.0 - gradient_normalization)

                    w = [
                        self.param(
                            f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}",
                            flax.linen.initializers.normal(stddev=stddev * ins.weight_std),
                            (d,) + ins.path_shape,
                            input.dtype,
                        )
                        if ins.i_in == -1
                        else self.param(
                            f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}",
                            flax.linen.initializers.normal(stddev=stddev * ins.weight_std),
                            (d,) + ins.path_shape,
                            input.dtype,
                        )
                        for ins in lin.instructions
                    ]  # List of shape (d, *path_shape)
                    w = [
                        jnp.sqrt(alpha) ** gradient_normalization
                        * jax.lax.dot_general(weights, wi.astype(input.array.dtype), (((weights.ndim - 1,), (0,)), ((), ())))
                        for wi in w
                    ]  # List of shape (..., *path_shape)

                    f = lin
                    for _ in range(input.ndim - 1):
                        f = jax.vmap(f)
                    output = f(w, input)

            else:
                raise ValueError(
                    "If weights are provided, they must be either integers and num_weights must be provided "
                    "or floats and num_weights must not be provided."
                )

        if self.channel_out is not None:
            output = output.mul_to_axis(self.channel_out)
        return output._convert(self.irreps_out)
