from math import prod

import jax
import jax.numpy as jnp

import e3nn_jax as e3nn


def batch_norm(
    input: e3nn.IrrepsArray,
    running_mean,
    running_var,
    weight,
    bias,
    normalization,
    reduce: str,
    is_training,
    is_instance,
    has_affine,
    momentum,
    epsilon,
):
    def _roll_avg(curr, update):
        return (1 - momentum) * curr + momentum * jax.lax.stop_gradient(update)

    batch, *size = input.shape[:-1]
    # TODO add test case for when prod(size) == 0

    input = input.reshape((batch, prod(size), -1))

    new_means = []
    new_vars = []

    fields = []

    i_wei = 0  # index for running_var and weight
    i_rmu = 0  # index for running_mean
    i_bia = 0  # index for bias

    for (mul, ir), field in zip(input.irreps, input.chunks):
        if field is None:
            # [batch, sample, mul, repr]
            if ir.is_scalar():  # scalars
                if is_training or is_instance:
                    if not is_instance:
                        new_means.append(jnp.zeros((mul,), dtype=input.dtype))
                i_rmu += mul

            if is_training or is_instance:
                if not is_instance:
                    new_vars.append(jnp.ones((mul,), input.dtype))

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
                        new_means.append(
                            _roll_avg(running_mean[i_rmu : i_rmu + mul], field_mean)
                        )
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
                    raise ValueError(
                        "Invalid normalization option {}".format(normalization)
                    )

                if reduce == "mean":
                    field_norm = field_norm.mean(1)  # [batch, mul]
                elif reduce == "max":
                    field_norm = field_norm.max(1)  # [batch, mul]
                else:
                    raise ValueError("Invalid reduce option {}".format(reduce))

                if not is_instance:
                    field_norm = field_norm.mean(0)  # [mul]
                    new_vars.append(
                        _roll_avg(running_var[i_wei : i_wei + mul], field_norm)
                    )
            else:
                field_norm = running_var[i_wei : i_wei + mul]

            field_norm = jax.lax.rsqrt(
                (1 - epsilon) * field_norm + epsilon
            )  # [(batch,) mul]

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

    output = e3nn.from_chunks(input.irreps, fields, (batch, prod(size)), input.dtype)
    output = output.reshape((batch,) + tuple(size) + (-1,))
    return output, new_means, new_vars
