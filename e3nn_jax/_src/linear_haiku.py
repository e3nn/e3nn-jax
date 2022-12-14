from typing import Callable, Optional, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn

from .linear import FunctionalLinear, Instruction


class Linear(hk.Module):
    r"""Equivariant Linear Haiku Module.

    Args:
        irreps_out (`Irreps`): output representations.
        channel_out (optional int): if specified, the last axis before the irreps
            is assumed to be the channel axis and is mixed with the irreps.
        irreps_in (optional `Irreps`): input representations. If not specified,
            the input is obtained when calling the module.
        biases (bool): whether to add a bias to the output.
        path_normalization (str or float): Normalization of the paths, ``element`` or ``path``.
            0/1 corresponds to a normalization where each element/path has an equal contribution to the forward.
        gradient_normalization (str or float): Normalization of the gradients, ``element`` or ``path``.
            0/1 corresponds to a normalization where each element/path has an equal contribution to the learning.
        get_parameter (optional Callable): function to get the parameters.
        num_indexed_weights (optional int): number of indexed weights. See example below.
        weights_per_channel (bool): whether to have one set of weights per channel.
        name (optional str): name of the module.

    Example:
        Basic usage::

        >>> import e3nn_jax as e3nn
        >>> @hk.without_apply_rng
        ... @hk.transform
        ... def linear(x):
        ...     return e3nn.Linear("0e + 1o")(x)
        >>> x = e3nn.IrrepsArray("1o + 2x0e", jnp.ones(5))
        >>> params = linear.init(jax.random.PRNGKey(0), x)
        >>> y = linear.apply(params, x)
        >>> y.shape
        (4,)

        External weights::

        >>> @hk.without_apply_rng
        ... @hk.transform
        ... def linear(w, x):
        ...     return e3nn.Linear("0e + 1o")(w, x)
        >>> x = e3nn.IrrepsArray("1o + 2x0e", jnp.ones(5))
        >>> w = jnp.array([1., 2., 3., 4.])
        >>> params = linear.init(jax.random.PRNGKey(0), w, x)
        >>> y = linear.apply(params, w, x)
        >>> y.shape
        (4,)

        External indices::

        >>> @hk.without_apply_rng
        ... @hk.transform
        ... def linear(i, x):
        ...     return e3nn.Linear("0e + 1o", num_indexed_weights=4)(i, x)
        >>> x = e3nn.IrrepsArray("1o + 2x0e", jnp.ones((2, 5)))
        >>> i = jnp.array([2, 3])
        >>> params = linear.init(jax.random.PRNGKey(0), i, x)
        >>> y = linear.apply(params, i, x)
        >>> y.shape
        (2, 4)
    """

    def __init__(
        self,
        irreps_out: e3nn.Irreps,
        channel_out: int = None,
        *,
        irreps_in: Optional[e3nn.Irreps] = None,
        biases: bool = False,
        path_normalization: Union[str, float] = None,
        gradient_normalization: Union[str, float] = None,
        get_parameter: Optional[Callable[[str, Instruction], jnp.ndarray]] = None,
        num_indexed_weights: Optional[int] = None,
        weights_per_channel: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.irreps_in = e3nn.Irreps(irreps_in) if irreps_in is not None else None
        self.channel_out = channel_out
        self.irreps_out = e3nn.Irreps(irreps_out)
        self.biases = biases
        self.path_normalization = path_normalization
        self.num_indexed_weights = num_indexed_weights
        self.weights_per_channel = weights_per_channel

        if gradient_normalization is None:
            gradient_normalization = e3nn.config("gradient_normalization")
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[gradient_normalization]
        self.gradient_normalization = gradient_normalization

        if get_parameter is None:

            def get_parameter(name: str, path_shape: Tuple[int, ...], weight_std: float, dtype: jnp.dtype = jnp.float32):
                return hk.get_parameter(
                    name,
                    shape=path_shape,
                    dtype=dtype,
                    init=hk.initializers.RandomNormal(stddev=weight_std),
                )

        self.get_parameter = get_parameter

    def __call__(
        self, weights: Optional[Union[e3nn.IrrepsArray, jnp.ndarray]], input: e3nn.IrrepsArray = None
    ) -> e3nn.IrrepsArray:
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
        if input is None:
            input = weights
            weights = None

        if self.irreps_in is not None:
            if self.irreps_in.regroup() != input.irreps.regroup():
                raise ValueError(
                    f"e3nn.Linear: The input irreps ({input.irreps}) do not match the expected irreps ({self.irreps_in})"
                )

        input = input.remove_nones().regroup()
        output_irreps = self.irreps_out.simplify()

        if self.channel_out is not None:
            assert not self.weights_per_channel
            input = input.axis_to_mul()
            output_irreps = e3nn.Irreps([(self.channel_out * mul, ir) for mul, ir in output_irreps])

        lin = FunctionalLinear(
            input.irreps,
            output_irreps,
            biases=self.biases,
            path_normalization=self.path_normalization,
            gradient_normalization=self.gradient_normalization,
        )

        if weights is None:
            assert not self.weights_per_channel  # Not implemented yet
            w = [
                self.get_parameter(f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}", ins.path_shape, ins.weight_std)
                if ins.i_in == -1
                else self.get_parameter(
                    f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}",
                    ins.path_shape,
                    ins.weight_std,
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
                assert not self.weights_per_channel  # Not implemented yet

                shape = jnp.broadcast_shapes(input.shape[:-1], weights.shape)
                input = input.broadcast_to(shape + (-1,))
                weights = jnp.broadcast_to(weights, shape)

                w = [
                    self.get_parameter(
                        f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}",
                        (self.num_indexed_weights,) + ins.path_shape,
                        ins.weight_std,
                        dtype=input.dtype,
                    )
                    if ins.i_in == -1
                    else self.get_parameter(
                        f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}",
                        (self.num_indexed_weights,) + ins.path_shape,
                        ins.weight_std,
                        dtype=input.dtype,
                    )
                    for ins in lin.instructions
                ]  # List of shape (num_weights, *path_shape)
                w = [wi[weights] for wi in w]  # List of shape (..., *path_shape)

                f = lin
                for _ in range(input.ndim - 1):
                    f = jax.vmap(f)
                output = f(w, input)

            elif weights.dtype.kind in "fc" and self.num_indexed_weights is None:

                if self.weights_per_channel:
                    shape = jnp.broadcast_shapes(input.shape[:-2], weights.shape[:-1])
                    input = input.broadcast_to(shape + input.shape[-2:])
                    weights = jnp.broadcast_to(weights, shape + weights.shape[-1:])
                    nc = input.shape[-2]

                    weights = weights.astype(input.array.dtype)

                    # Should be equivalent to the last layer of e3nn.MultiLayerPerceptron
                    d = weights.shape[-1]
                    alpha = 1 / d
                    stddev = jnp.sqrt(alpha) ** (1.0 - self.gradient_normalization)

                    w = [
                        self.get_parameter(
                            f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}",
                            (d, nc) + ins.path_shape,
                            stddev * ins.weight_std,
                            dtype=input.dtype,
                        )
                        if ins.i_in == -1
                        else self.get_parameter(
                            f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}",
                            (d, nc) + ins.path_shape,
                            stddev * ins.weight_std,
                            dtype=input.dtype,
                        )
                        for ins in lin.instructions
                    ]  # List of shape (d, *path_shape)
                    w = [
                        jnp.sqrt(alpha) ** self.gradient_normalization
                        * jax.lax.dot_general(weights, wi.astype(input.array.dtype), (((weights.ndim - 1,), (0,)), ((), ())))
                        for wi in w
                    ]  # List of shape (..., num_channels, *path_shape)

                    f = lin
                    for _ in range(input.ndim - 1):
                        f = jax.vmap(f)
                    output = f(w, input)
                else:
                    shape = jnp.broadcast_shapes(input.shape[:-1], weights.shape[:-1])
                    input = input.broadcast_to(shape + (-1,))
                    weights = jnp.broadcast_to(weights, shape + weights.shape[-1:])

                    weights = weights.astype(input.array.dtype)

                    # Should be equivalent to the last layer of e3nn.MultiLayerPerceptron
                    d = weights.shape[-1]
                    alpha = 1 / d
                    stddev = jnp.sqrt(alpha) ** (1.0 - self.gradient_normalization)

                    w = [
                        self.get_parameter(
                            f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}",
                            (d,) + ins.path_shape,
                            stddev * ins.weight_std,
                            dtype=input.dtype,
                        )
                        if ins.i_in == -1
                        else self.get_parameter(
                            f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}",
                            (d,) + ins.path_shape,
                            stddev * ins.weight_std,
                            dtype=input.dtype,
                        )
                        for ins in lin.instructions
                    ]  # List of shape (d, *path_shape)
                    w = [
                        jnp.sqrt(alpha) ** self.gradient_normalization
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
