import haiku as hk
import jax.numpy as jnp

import e3nn_jax as e3nn

from .bn import batch_norm


class BatchNorm(hk.Module):
    """Equivariant Batch Normalization.

    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations are orthonormal.

    Args:
        irreps: Irreducible representations of the input and output (unchanged)
        eps (float): epsilon for numerical stability, has to be between 0 and 1.
            the field norm is transformed to ``(1 - eps) * norm + eps``
            leading to a slower convergence toward norm 1.
        momentum: momentum for moving average
        affine: whether to include learnable weights and biases
        reduce: reduce mode, either 'mean' or 'max'
        instance: whether to use instance normalization instead of batch normalization
        normalization: normalization mode, either 'norm' or 'component'
    """

    def __init__(
        self,
        *,
        irreps: e3nn.Irreps = None,
        eps: float = 1e-4,
        momentum: float = 0.1,
        affine: bool = True,
        reduce: str = "mean",
        instance: bool = False,
        normalization: str = None,
    ):
        super().__init__()

        # TODO test with and without irreps argument given

        self.irreps = e3nn.Irreps(irreps) if irreps is not None else irreps
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.instance = instance

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ["mean", "max"], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce

        if normalization is None:
            normalization = e3nn.config("irrep_normalization")
        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps}, momentum={self.momentum})"

    def __call__(
        self, input: e3nn.IrrepsArray, is_training: bool = True
    ) -> e3nn.IrrepsArray:
        r"""Evaluate the batch normalization.

        Args:
            input: input tensor of shape ``(batch, [spatial], irreps.dim)``
            is_training: whether to train or evaluate

        Returns:
            output: normalized tensor of shape ``(batch, [spatial], irreps.dim)``
        """
        if self.irreps is not None:
            input = input.rechunk(self.irreps)

        num_scalar = sum(mul for mul, ir in input.irreps if ir.is_scalar())
        num_features = input.irreps.num_irreps

        if not self.instance:
            running_mean = hk.get_state(
                "running_mean", shape=(num_scalar,), init=jnp.zeros
            )
            running_var = hk.get_state(
                "running_var", shape=(num_features,), init=jnp.ones
            )
        else:
            running_mean = None
            running_var = None

        if self.affine:
            weight = hk.get_parameter("weight", shape=(num_features,), init=jnp.ones)
            bias = hk.get_parameter("bias", shape=(num_scalar,), init=jnp.zeros)
        else:
            weight = None
            bias = None

        output, new_means, new_vars = batch_norm(
            input,
            ra_mean=running_mean,
            ra_var=running_var,
            weight=weight,
            bias=bias,
            normalization=self.normalization,
            reduce=self.reduce,
            is_instance=self.instance,
            use_running_average=not is_training and not self.instance,
            use_affine=self.affine,
            momentum=self.momentum,
            epsilon=self.eps,
        )

        if is_training and not self.instance:
            if len(new_means):
                hk.set_state("running_mean", new_means)
            if len(new_vars):
                hk.set_state("running_var", new_vars)

        return output
