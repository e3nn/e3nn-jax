from typing import Any, Callable, Optional, Tuple, Union

import haiku as hk
import jax.numpy as jnp

import e3nn_jax as e3nn
from e3nn_jax._src.utils.dtype import get_pytree_dtype

from .linear import (
    FunctionalLinear,
    linear_indexed,
    linear_mixed,
    linear_mixed_per_channel,
    linear_vanilla,
)


class Linear(hk.Module):
    r"""Equivariant Linear Haiku Module.

    Args:
        irreps_out (`Irreps`): output representations, if allowed bu Schur's lemma.
        channel_out (optional int): if specified, the last axis before the irreps
            is assumed to be the channel axis and is mixed with the irreps.
        irreps_in (optional `Irreps`): input representations. If not specified,
            the input representations is obtained when calling the module.
        biases (bool): whether to add a bias to the output.
        path_normalization (str or float): Normalization of the paths, ``element`` or ``path``.
            0/1 corresponds to a normalization where each element/path has an equal contribution to the forward.
        gradient_normalization (str or float): Normalization of the gradients, ``element`` or ``path``.
            0/1 corresponds to a normalization where each element/path has an equal contribution to the learning.
        get_parameter (optional Callable): function to get the parameters.
        num_indexed_weights (optional int): number of indexed weights. See example below.
        weights_per_channel (bool): whether to have one set of weights per channel.
        force_irreps_out (bool): whether to force the output irreps to be the one specified in ``irreps_out``.
        name (optional str): name of the module.

    Examples:
        Vanilla::

            >>> import e3nn_jax as e3nn
            >>> import jax
            >>>
            >>> @hk.without_apply_rng
            ... @hk.transform
            ... def linear(x):
            ...     return e3nn.haiku.Linear("0e + 1o + 2e")(x)
            >>> x = e3nn.IrrepsArray("1o + 2x0e", jnp.ones(5))
            >>> params = linear.init(jax.random.PRNGKey(0), x)
            >>> y = linear.apply(params, x)
            >>> y.irreps  # Note that the 2e is discarded
            1x0e+1x1o
            >>> y.shape
            (4,)

        External weights::

            >>> @hk.without_apply_rng
            ... @hk.transform
            ... def linear(w, x):
            ...     return e3nn.haiku.Linear("0e + 1o")(w, x)
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
            ...     return e3nn.haiku.Linear("0e + 1o", num_indexed_weights=4)(i, x)
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
        get_parameter: Optional[
            Callable[[str, Tuple[int, ...], float, Any], jnp.ndarray]
        ] = None,
        num_indexed_weights: Optional[int] = None,
        weights_per_channel: bool = False,
        force_irreps_out: bool = False,
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
        self.force_irreps_out = force_irreps_out

        if gradient_normalization is None:
            gradient_normalization = e3nn.config("gradient_normalization")
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[
                gradient_normalization
            ]
        self.gradient_normalization = gradient_normalization

        if get_parameter is None:

            def get_parameter(
                name: str,
                path_shape: Tuple[int, ...],
                weight_std: float,
                dtype: jnp.dtype = jnp.float32,
            ):
                return hk.get_parameter(
                    name,
                    shape=path_shape,
                    dtype=dtype,
                    init=hk.initializers.RandomNormal(stddev=weight_std),
                )

        self.get_parameter = get_parameter

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

        input = e3nn.as_irreps_array(input)

        dtype = get_pytree_dtype(weights, input)
        if dtype.kind == "i":
            dtype = jnp.float32
        input = input.astype(dtype)

        if self.irreps_in is not None:
            if self.irreps_in.regroup() != input.irreps.regroup():
                raise ValueError(
                    f"e3nn.haiku.Linear: The input irreps ({input.irreps}) do not match the expected irreps ({self.irreps_in})"
                )

        input = input.remove_zero_chunks().regroup()
        if self.force_irreps_out:
            output_irreps = self.irreps_out.simplify()
        else:
            output_irreps_unsimplified = self.irreps_out.filter(input.irreps)
            output_irreps = output_irreps_unsimplified.simplify()

        if self.channel_out is not None:
            assert not self.weights_per_channel
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
            assert not self.weights_per_channel  # Not implemented yet
            output = linear_vanilla(input, lin, self.get_parameter)
        else:
            if isinstance(weights, e3nn.IrrepsArray):
                if not weights.irreps.is_scalar():
                    raise ValueError("weights must be scalar")
                weights = weights.array

            if weights.dtype.kind == "i" and self.num_indexed_weights is not None:
                assert not self.weights_per_channel  # Not implemented yet
                output = linear_indexed(
                    input, lin, self.get_parameter, weights, self.num_indexed_weights
                )

            elif weights.dtype.kind in "fc" and self.num_indexed_weights is None:
                if self.weights_per_channel:
                    output = linear_mixed_per_channel(
                        input,
                        lin,
                        self.get_parameter,
                        weights,
                        self.gradient_normalization,
                    )
                else:
                    output = linear_mixed(
                        input,
                        lin,
                        self.get_parameter,
                        weights,
                        self.gradient_normalization,
                    )

            else:
                raise ValueError(
                    "If weights are provided, they must be either integers and num_indexed_weights must be provided "
                    "or floats and num_indexed_weights must not be provided. "
                    f"weights.dtype={weights.dtype}, num_indexed_weights={self.num_indexed_weights}"
                )

        if self.channel_out is not None:
            output = output.mul_to_axis(self.channel_out)

        if self.force_irreps_out:
            return output.rechunk(self.irreps_out)
        else:
            return output.rechunk(output_irreps_unsimplified)
