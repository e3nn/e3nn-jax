from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp

from e3nn_jax import Irreps, IrrepsData
from e3nn_jax.util import prod


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9, 10, 11))
def _batch_norm(
    input,
    running_mean,
    running_var,
    weight,
    bias,
    normalization,
    reduce,
    is_training,
    is_instance,
    has_affine,
    momentum,
    epsilon,
):
    def _roll_avg(curr, update):
        return (1 - momentum) * curr + momentum * jax.lax.stop_gradient(update)

    batch, *size = input.shape
    # TODO add test case for when prod(size) == 0

    input = input.reshape((batch, prod(size)))

    new_means = []
    new_vars = []

    fields = []

    i_wei = 0  # index for running_var and weight
    i_rmu = 0  # index for running_mean
    i_bia = 0  # index for bias

    for (mul, ir), field in zip(input.irreps, input.list):
        if field is None:
            # [batch, sample, mul, repr]
            if ir.is_scalar():  # scalars
                if is_training or is_instance:
                    if not is_instance:
                        new_means.append(jnp.zeros((mul,)))
                i_rmu += mul

            if is_training or is_instance:
                if not is_instance:
                    new_vars.append(jnp.ones((mul,)))

            if has_affine and ir.is_scalar():  # scalars
                i_bia += mul

            fields.append(field)  # [batch, sample, mul, repr]
        else:
            # [batch, sample, mul, repr]
            if ir.is_scalar():  # scalars
                if is_training or is_instance:
                    if is_instance:
                        field_mean = field.mean(1).reshape(batch, mul)  # [batch, mul]
                    else:
                        field_mean = field.mean([0, 1]).reshape(mul)  # [mul]
                        new_means.append(_roll_avg(running_mean[i_rmu : i_rmu + mul], field_mean))
                else:
                    field_mean = running_mean[i_rmu : i_rmu + mul]
                i_rmu += mul

                # [batch, sample, mul, repr]
                field = field - field_mean.reshape(-1, 1, mul, 1)

            if is_training or is_instance:
                if normalization == "norm":
                    field_norm = jnp.square(field).sum(3)  # [batch, sample, mul]
                elif normalization == "component":
                    field_norm = jnp.square(field).mean(3)  # [batch, sample, mul]
                else:
                    raise ValueError("Invalid normalization option {}".format(normalization))

                if reduce == "mean":
                    field_norm = field_norm.mean(1)  # [batch, mul]
                elif reduce == "max":
                    field_norm = field_norm.max(1)  # [batch, mul]
                else:
                    raise ValueError("Invalid reduce option {}".format(reduce))

                if not is_instance:
                    field_norm = field_norm.mean(0)  # [mul]
                    new_vars.append(_roll_avg(running_var[i_wei : i_wei + mul], field_norm))
            else:
                field_norm = running_var[i_wei : i_wei + mul]

            field_norm = jax.lax.rsqrt(field_norm + epsilon)  # [(batch,) mul]

            if has_affine:
                sub_weight = weight[i_wei : i_wei + mul]  # [mul]
                field_norm = field_norm * sub_weight  # [(batch,) mul]

            # TODO add test case for when mul == 0
            field_norm = field_norm[..., None, :, None]  # [(batch,) 1, mul, 1]
            field = field * field_norm  # [batch, sample, mul, repr]

            if has_affine and ir.is_scalar():  # scalars
                sub_bias = bias[i_bia : i_bia + mul]  # [mul]
                field += sub_bias.reshape(mul, 1)  # [batch, sample, mul, repr]
                i_bia += mul

            fields.append(field)  # [batch, sample, mul, repr]
        i_wei += mul

    output = IrrepsData.from_list(input.irreps, fields, (batch, prod(size)))
    output = output.reshape((batch,) + tuple(size))
    return output, new_means, new_vars


class BatchNorm(hk.Module):
    """Batch normalization for orthonormal representations
    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.

    Args:
        irreps: Irreducible representations
        eps (float): small number to avoid division by zero
        momentum: momentum for moving average
        affine: whether to include learnable biases
        reduce: reduce mode, either 'mean' or 'max'
        instance: whether to use instance normalization
        normalization: normalization mode, either 'norm' or 'component'
    """

    def __init__(
        self, *, irreps=None, eps=1e-4, momentum=0.1, affine=True, reduce="mean", instance=False, normalization="component"
    ):
        super().__init__()

        # TODO test with and without irreps argument given

        self.irreps = Irreps(irreps) if irreps is not None else irreps
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.instance = instance

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ["mean", "max"], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce

        assert normalization in ["norm", "component"], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps}, momentum={self.momentum})"

    def __call__(self, input, is_training=True):
        r"""evaluate the batch normalization

        Args:
            input: input tensor of shape ``(batch, [spatial], irreps.dim)``
            is_training: whether to train or evaluate

        Returns:
            output: normalized tensor of shape ``(batch, [spatial], irreps.dim)``
        """
        if self.irreps is not None:
            input = IrrepsData.new(self.irreps, input)
        if not isinstance(input, IrrepsData):
            raise ValueError("input should be of type IrrepsData")

        num_scalar = sum(mul for mul, ir in input.irreps if ir.is_scalar())
        num_features = input.irreps.num_irreps

        if not self.instance:
            running_mean = hk.get_state("running_mean", shape=(num_scalar,), init=jnp.zeros)
            running_var = hk.get_state("running_var", shape=(num_features,), init=jnp.ones)
        else:
            running_mean = None
            running_var = None

        if self.affine:
            weight = hk.get_parameter("weight", shape=(num_features,), init=jnp.ones)
            bias = hk.get_parameter("bias", shape=(num_scalar,), init=jnp.zeros)
        else:
            weight = None
            bias = None

        output, new_means, new_vars = _batch_norm(
            input,
            running_mean=running_mean,
            running_var=running_var,
            weight=weight,
            bias=bias,
            normalization=self.normalization,
            reduce=self.reduce,
            is_training=is_training,
            is_instance=self.instance,
            has_affine=self.affine,
            momentum=self.momentum,
            epsilon=self.eps,
        )

        if is_training and not self.instance:
            if len(new_means):
                hk.set_state("running_mean", jnp.concatenate(new_means))
            if len(new_vars):
                hk.set_state("running_var", jnp.concatenate(new_vars))

        return output
