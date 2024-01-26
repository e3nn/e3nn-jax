from typing import Optional, Union, Tuple, Dict

import equinox as eqx
import jax
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


def _get_gradient_normalization(
    gradient_normalization: Optional[Union[float, str]]
) -> float:
    """Get the gradient normalization from the config or from the argument."""
    if gradient_normalization is None:
        gradient_normalization = e3nn.config("gradient_normalization")
    if isinstance(gradient_normalization, str):
        return {"element": 0.0, "path": 1.0}[gradient_normalization]
    return gradient_normalization


class Linear(eqx.Module):
    r"""Equivariant Linear Flax module

    Args:
        irreps_out (`Irreps`): output representations, if allowed bu Schur's lemma.
        channel_out (optional int): if specified, the last axis before the irreps
            is assumed to be the channel axis and is mixed with the irreps.
        irreps_in (`Irreps`): input representations. If not specified,
            the input representations is obtained when calling the module.
        channel_in (optional int): required when using 'mixed_per_channel' linear_type,
            indicating the size of the last axis before the irreps in the input.
        biases (bool): whether to add a bias to the output.
        path_normalization (str or float): Normalization of the paths, ``element`` or ``path``.
            0/1 corresponds to a normalization where each element/path has an equal contribution to the forward.
        gradient_normalization (str or float): Normalization of the gradients, ``element`` or ``path``.
            0/1 corresponds to a normalization where each element/path has an equal contribution to the learning.
        num_indexed_weights (optional int): number of indexed weights. See example below.
        weights_per_channel (bool): whether to have one set of weights per channel.
        force_irreps_out (bool): whether to force the output irreps to be the one specified in ``irreps_out``.

    Due to how Equinox is implemented, the random key, irreps_in and irreps_out must be supplied at initialization.
    The type of the linear layer must also be supplied at initialization:
    'vanilla', 'indexed', 'mixed', 'mixed_per_channel'
    Also, depending on what type of linear layer is used, additional options
    (eg. 'num_indexed_weights', 'weights_per_channel', 'weights_dim', 'channel_in')
    must be supplied.

    Examples:
        Vanilla::

            >>> import e3nn_jax as e3nn
            >>> import jax

            >>> x = e3nn.normal("0e + 1o")
            >>> linear = e3nn.equinox.Linear(
                    irreps_out="2x0e + 1o + 2e",
                    irreps_in=x.irreps,
                    key=jax.random.PRNGKey(0),
                )
            >>> linear(x).irreps  # Note that the 2e is discarded. Avoid this by setting force_irreps_out=True.
            2x0e+1x1o
            >>> linear(x).shape
            (5,)

        External weights::

            >>> linear = e3nn.equinox.Linear(
                    irreps_out="2x0e + 1o",
                    irreps_in=x.irreps,
                    linear_type="mixed",
                    weights_dim=4,
                    key=jax.random.PRNGKey(0),
                )
            >>> e = jnp.array([1., 2., 3., 4.])
            >>> linear(e, x).irreps
                2x0e+1x1o
            >>> linear(e, x).shape
            (5,)

        Indexed weights::

            >>> linear = e3nn.equinox.Linear(
                    irreps_out="2x0e + 1o + 2e",
                    irreps_in=x.irreps,
                    linear_type="indexed",
                    num_indexed_weights=3,
                    key=jax.random.PRNGKey(0),
                )
            >>> i = jnp.array(2)
            >>> linear(i, x).irreps
                2x0e+1x1o
            >>> linear(i, x).shape
            (5,)
    """

    irreps_out: e3nn.Irreps
    irreps_in: e3nn.Irreps
    channel_out: int
    channel_in: int
    gradient_normalization: Optional[Union[float, str]]
    path_normalization: Optional[Union[float, str]]
    biases: bool
    num_indexed_weights: Optional[int]
    weights_per_channel: bool
    force_irreps_out: bool
    weights_dim: Optional[int]
    linear_type: str

    # These are used internally.
    _linear: FunctionalLinear
    _weights: Dict[str, jax.Array]
    _input_dtype: jnp.dtype

    def __init__(
        self,
        *,
        irreps_out: e3nn.Irreps,
        irreps_in: e3nn.Irreps,
        channel_out: Optional[int] = None,
        channel_in: Optional[int] = None,
        biases: bool = False,
        path_normalization: Optional[Union[str, float]] = None,
        gradient_normalization: Optional[Union[str, float]] = None,
        num_indexed_weights: Optional[int] = None,
        weights_per_channel: bool = False,
        force_irreps_out: bool = False,
        weights_dim: Optional[int] = None,
        input_dtype: jnp.dtype = jnp.float32,
        linear_type: str = "vanilla",
        key: jax.Array,
    ):
        irreps_in_regrouped = e3nn.Irreps(irreps_in).regroup()
        irreps_out = e3nn.Irreps(irreps_out)

        self.irreps_in = irreps_in_regrouped
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.biases = biases
        self.path_normalization = path_normalization
        self.num_indexed_weights = num_indexed_weights
        self.weights_per_channel = weights_per_channel
        self.force_irreps_out = force_irreps_out
        self.linear_type = linear_type
        self.weights_dim = weights_dim
        self._input_dtype = input_dtype

        self.gradient_normalization = _get_gradient_normalization(
            gradient_normalization
        )

        channel_irrep_multiplier = 1
        if self.channel_out is not None:
            assert not self.weights_per_channel
            channel_irrep_multiplier = self.channel_out

        if not self.force_irreps_out:
            irreps_out = irreps_out.filter(keep=irreps_in_regrouped)
            irreps_out = irreps_out.simplify()
        self.irreps_out = irreps_out

        self._linear = FunctionalLinear(
            irreps_in_regrouped,
            channel_irrep_multiplier * irreps_out,
            biases=self.biases,
            path_normalization=self.path_normalization,
            gradient_normalization=self.gradient_normalization,
        )
        self._weights = self._get_weights(key)

    def _get_weights(self, key: jax.Array):
        """Constructs the weights for the linear module."""
        irreps_in = self._linear.irreps_in
        irreps_out = self._linear.irreps_out

        weights = {}
        for ins in self._linear.instructions:
            weight_key, key = jax.random.split(key)
            if ins.i_in == -1:
                name = f"b[{ins.i_out}] {irreps_out[ins.i_out]}"
            else:
                name = f"w[{ins.i_in},{ins.i_out}] {irreps_in[ins.i_in]},{irreps_out[ins.i_out]}"

            if self.linear_type == "vanilla":
                weight_shape = ins.path_shape
                weight_std = ins.weight_std

            if self.linear_type == "indexed":
                if self.num_indexed_weights is None:
                    raise ValueError(
                        "num_indexed_weights must be provided when 'linear_type' is 'indexed'"
                    )

                weight_shape = (self.num_indexed_weights,) + ins.path_shape
                weight_std = ins.weight_std

            if self.linear_type in ["mixed", "mixed_per_channel"]:
                if self.weights_dim is None:
                    raise ValueError(
                        "weights_dim must be provided when 'linear_type' is 'mixed'"
                    )

                d = self.weights_dim
                if self.linear_type == "mixed":
                    weight_shape = (d,) + ins.path_shape

                if self.linear_type == "mixed_per_channel":
                    if self.channel_in is None:
                        raise ValueError(
                            "channel_in must be provided when 'linear_type' is 'mixed_per_channel'"
                        )
                    weight_shape = (d, self.channel_in) + ins.path_shape

                alpha = 1 / d
                stddev = jnp.sqrt(alpha) ** (1.0 - self.gradient_normalization)
                weight_std = stddev * ins.weight_std

            weights[name] = weight_std * jax.random.normal(
                weight_key,
                weight_shape,
                self._input_dtype,
            )
        return weights

    def __call__(self, weights_or_input, input_or_none=None) -> e3nn.IrrepsArray:
        """Apply the linear operator.

        Args:
            weights (optional IrrepsArray or jax.Array): scalar weights that are contracted with free parameters.
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
            weights: jax.Array = weights_or_input
            input: e3nn.IrrepsArray = input_or_none
        del weights_or_input, input_or_none

        input = e3nn.as_irreps_array(input)

        dtype = get_pytree_dtype(weights, input)
        if dtype.kind == "i":
            dtype = jnp.float32
        input = input.astype(dtype)

        if self.irreps_in != input.irreps.regroup():
            raise ValueError(
                f"e3nn.equinox.Linear: The input irreps ({input.irreps}) "
                f"do not match the expected irreps ({self.irreps_in})."
            )

        if self.channel_in is not None:
            if self.channel_in != input.shape[-2]:
                raise ValueError(
                    f"e3nn.equinox.Linear: The input channel ({input.shape[-2]}) "
                    f"does not match the expected channel ({self.channel_in})."
                )

        input = input.remove_zero_chunks().regroup()

        def get_parameter(
            name: str,
            path_shape: Tuple[int, ...],
            weight_std: float,
            dtype: jnp.dtype = jnp.float32,
        ):
            del path_shape, weight_std, dtype
            return self._weights[name]

        assertion_message = (
            "Weights cannot be provided when 'linear_type' is 'vanilla'."
            "Otherwise, weights must be provided."
            "If weights are provided, they must be either: \n"
            "* integers and num_indexed_weights must be provided, or \n"
            "* floats and num_indexed_weights must not be provided.\n"
            f"weights.dtype={weights.dtype if weights is not None else None}, "
            f"num_indexed_weights={self.num_indexed_weights}"
        )

        if self.linear_type == "vanilla":
            assert weights is None, assertion_message
            output = linear_vanilla(input, self._linear, get_parameter)

        if self.linear_type in ["indexed", "mixed", "mixed_per_channel"]:
            assert weights is not None, assertion_message
            if isinstance(weights, e3nn.IrrepsArray):
                if not weights.irreps.is_scalar():
                    raise ValueError("weights must be scalar")
                weights = weights.array

        if self.linear_type == "indexed":
            assert weights.dtype.kind == "i", assertion_message
            if self.weights_per_channel:
                raise NotImplementedError(
                    "weights_per_channel not implemented for indexed weights"
                )

            output = linear_indexed(
                input, self._linear, get_parameter, weights, self.num_indexed_weights
            )

        if self.linear_type in ["mixed", "mixed_per_channel"]:
            assert weights.dtype.kind in "fc", assertion_message
            assert self.num_indexed_weights is None, assertion_message

        if self.linear_type == "mixed":
            output = linear_mixed(
                input,
                self._linear,
                get_parameter,
                weights,
                self.gradient_normalization,
            )

        if self.linear_type == "mixed_per_channel":
            output = linear_mixed_per_channel(
                input,
                self._linear,
                get_parameter,
                weights,
                self.gradient_normalization,
            )

        if self.channel_out is not None:
            output = output.mul_to_axis(self.channel_out)

        return output.rechunk(self.irreps_out)
