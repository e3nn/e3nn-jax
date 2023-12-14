from math import prod

import jax
import jax.numpy as jnp

import e3nn_jax as e3nn


def batch_norm(
    input: e3nn.IrrepsArray,
    ra_mean: jnp.ndarray,
    ra_var: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray,
    normalization: str,
    reduce: str,
    is_instance: bool,
    use_running_average: bool,
    use_affine: bool,
    momentum: float,
    epsilon: float,
):
    def _roll_avg(curr, update):
        return (1 - momentum) * curr + momentum * jax.lax.stop_gradient(update)

    batch, *size = input.shape[:-1]
    input = input.reshape((batch, prod(size), -1))

    if not is_instance:
        new_ra_means = []
        new_ra_vars = []

    new_chunks = []

    i_wei = 0  # index for running_var and weight
    i_rmu = 0  # index for running_mean
    i_bia = 0  # index for bias

    for (mul, ir), chunk in zip(input.irreps, input.chunks):
        if chunk is None:
            # [batch, sample, mul, repr]
            if ir.is_scalar():  # scalars
                if not is_instance:
                    new_ra_means.append(jnp.zeros((mul,), dtype=input.dtype))
                i_rmu += mul

            if not is_instance:
                new_ra_vars.append(jnp.ones((mul,), input.dtype))

            if use_affine and ir.is_scalar():  # scalars
                i_bia += mul

            new_chunks.append(chunk)  # [batch, sample, mul, repr]
        else:
            # [batch, sample, mul, repr]
            if ir.is_scalar():  # scalars
                if is_instance:
                    field_mean = chunk.mean(1).reshape(batch, mul)  # [batch, mul]
                else:
                    field_mean = chunk.mean([0, 1]).reshape(mul)  # [mul]
                    new_ra_means.append(
                        _roll_avg(ra_mean[i_rmu : i_rmu + mul], field_mean)
                    )

                if use_running_average:
                    field_mean = ra_mean[i_rmu : i_rmu + mul]
                i_rmu += mul

                # [batch, sample, mul, repr]
                chunk = chunk - field_mean.reshape(-1, 1, mul, 1)

            if normalization == "norm":
                field_norm = jnp.square(chunk).sum(3)  # [batch, sample, mul]
            elif normalization == "component":
                field_norm = jnp.square(chunk).mean(3)  # [batch, sample, mul]
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
                new_ra_vars.append(_roll_avg(ra_var[i_wei : i_wei + mul], field_norm))

            if use_running_average:
                field_norm = ra_var[i_wei : i_wei + mul]

            field_norm = jax.lax.rsqrt(
                (1 - epsilon) * field_norm + epsilon
            )  # [(batch,) mul]

            if use_affine:
                sub_weight = weight[i_wei : i_wei + mul]  # [mul]
                field_norm = field_norm * sub_weight  # [(batch,) mul]

            # TODO add test case for when mul == 0
            field_norm = field_norm[..., None, :, None]  # [(batch,) 1, mul, 1]
            chunk = chunk * field_norm  # [batch, sample, mul, repr]

            if use_affine and ir.is_scalar():  # scalars
                sub_bias = bias[i_bia : i_bia + mul]  # [mul]
                chunk += sub_bias.reshape(mul, 1)  # [batch, sample, mul, repr]
                i_bia += mul

            new_chunks.append(chunk)  # [batch, sample, mul, repr]
        i_wei += mul

    output = e3nn.from_chunks(
        input.irreps, new_chunks, (batch, prod(size)), input.dtype
    )
    output = output.reshape((batch,) + tuple(size) + (-1,))

    if not is_instance:
        new_ra_means = (
            jnp.concatenate(new_ra_means)
            if new_ra_means
            else jnp.zeros_like(output, shape=(0,))
        )
        new_ra_vars = jnp.concatenate(new_ra_vars)
    else:
        new_ra_means = None
        new_ra_vars = None

    return output, new_ra_means, new_ra_vars
