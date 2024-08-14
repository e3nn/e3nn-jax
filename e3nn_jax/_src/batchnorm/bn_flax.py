from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn

from .bn import batch_norm


def first_nonnone(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


class BatchNorm(nn.Module):
    """Equivariant Batch Normalization.

    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations are orthonormal.

    Args:
        use_running_average: if True, the statistics stored in batch_stats will be
            used instead of computing the batch statistics on the input.
        eps (float): epsilon for numerical stability, has to be between 0 and 1.
            the field norm is transformed to ``(1 - eps) * norm + eps``
            leading to a slower convergence toward norm 1.
        momentum: momentum for moving average
        affine: whether to include learnable weights and biases
        reduce: reduce mode, either 'mean' or 'max'
        instance: whether to use instance normalization instead of batch normalization
        normalization: normalization mode, either 'norm' or 'component'
    """

    use_running_average: Optional[bool] = None
    eps: float = 1e-4
    momentum: float = 0.1
    affine: bool = True
    reduce: str = "mean"
    instance: bool = False
    normalization: str = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps}, momentum={self.momentum})"

    @nn.compact
    def __call__(
        self,
        input: e3nn.IrrepsArray,
        use_running_average: Optional[bool] = None,
        mask: Optional[jax.Array] = None,
    ) -> e3nn.IrrepsArray:
        """Normalizes the input using batch statistics.

        NOTE:
        During initialization (when `self.is_initializing()` is `True`) the running
        average of the batch statistics will not be updated. Therefore, the inputs
        fed during initialization don't need to match that of the actual input
        distribution and the reduction axis (set with `axis_name`) does not have
        to exist.

        Args:
            input: the input to be normalized.
            use_running_average: if true, the statistics stored in batch_stats will be
            used instead of computing the batch statistics on the input.
            mask: a boolean mask of shape (batch,) to indicate which inputs are valid.

        Returns:
            Normalized inputs (the same shape and irreps as input).
        """
        input = e3nn.as_irreps_array(input)

        if mask is not None and mask.shape != (input.shape[0],):
            raise ValueError(
                f"mask must have shape (batch,) but got {mask.shape} instead."
            )

        use_running_average = first_nonnone(
            use_running_average, self.use_running_average, False
        )

        if use_running_average and self.instance:
            # If instance, we can't use running average because the mean and variance
            # are different for each instance (i.e. they have a batch dimension)
            raise ValueError("If instance is True, use_running_average must be False")

        dtype = input.dtype

        num_scalars = input.irreps.filter(keep="0e").num_irreps
        num_irreps = input.irreps.num_irreps

        if not self.instance:
            ra_mean = self.variable(
                "batch_stats",
                "mean",
                lambda s: jnp.zeros(s, dtype),
                (num_scalars,),
            )
            ra_var = self.variable(
                "batch_stats", "var", lambda s: jnp.ones(s, dtype), (num_irreps,)
            )
        else:
            ra_mean = None
            ra_var = None

        if self.affine:
            weights = self.param("weights", lambda _: jnp.ones((num_irreps,), dtype))
            biases = self.param("biases", lambda _: jnp.zeros((num_scalars,), dtype))
        else:
            weights = None
            biases = None

        output, new_means, new_vars = batch_norm(
            input,
            ra_mean.value if ra_mean else None,
            ra_var.value if ra_var else None,
            weights,
            biases,
            self.normalization or e3nn.config("irrep_normalization"),
            self.reduce,
            self.instance,
            use_running_average,
            self.affine,
            self.momentum,
            self.eps,
            mask,
        )

        if not self.is_initializing() and not self.instance:
            ra_mean.value = new_means
            ra_var.value = new_vars

        return output
