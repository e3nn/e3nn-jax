from math import prod
from typing import Optional

import jax
import jax.numpy as jnp

import e3nn_jax as e3nn


def batch_norm(
    input: e3nn.IrrepsArray,
    ra_mean: Optional[jax.Array],
    ra_var: Optional[jax.Array],
    weight: Optional[jax.Array],
    bias: Optional[jax.Array],
    normalization: str,
    reduce: str,
    is_instance: bool,
    use_running_average: bool,
    use_affine: bool,
    momentum: float,
    epsilon: float,
    mask: Optional[jax.Array] = None,
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
                    mean = chunk.mean(1).reshape(batch, mul)  # [batch, mul]
                else:
                    if mask is None:
                        mean = chunk.mean([0, 1]).reshape(mul)  # [mul]
                    else:
                        mean = (chunk.mean(1).squeeze(2) * mask[:, None]).sum(
                            0
                        ) / mask.sum()  # [mul]
                    new_ra_means.append(_roll_avg(ra_mean[i_rmu : i_rmu + mul], mean))

                if use_running_average:
                    mean = ra_mean[i_rmu : i_rmu + mul]
                i_rmu += mul

                # [batch, sample, mul, repr]
                chunk = chunk - mean.reshape(-1, 1, mul, 1)

            if normalization == "norm":
                norm_squared = jnp.square(chunk).sum(3)  # [batch, sample, mul]
            elif normalization == "component":
                norm_squared = jnp.square(chunk).mean(3)  # [batch, sample, mul]
            else:
                raise ValueError(
                    "Invalid normalization option {}".format(normalization)
                )

            if reduce == "mean":
                norm_squared = norm_squared.mean(1)  # [batch, mul]
            elif reduce == "max":
                norm_squared = norm_squared.max(1)  # [batch, mul]
            else:
                raise ValueError("Invalid reduce option {}".format(reduce))

            if not is_instance:
                if mask is None:
                    norm_squared = norm_squared.mean(0)  # [mul]
                else:
                    norm_squared = (norm_squared * mask[:, None]).sum(0) / mask.sum()
                new_ra_vars.append(_roll_avg(ra_var[i_wei : i_wei + mul], norm_squared))

            if use_running_average:
                norm_squared = ra_var[i_wei : i_wei + mul]

            inverse = jax.lax.rsqrt(
                (1 - epsilon) * norm_squared + epsilon
            )  # [(batch,) mul]

            if use_affine:
                sub_weight = weight[i_wei : i_wei + mul]  # [mul]
                inverse = inverse * sub_weight  # [(batch,) mul]

            chunk = chunk * inverse[..., None, :, None]  # [batch, sample, mul, repr]

            if use_affine and ir.is_scalar():  # scalars
                sub_bias = bias[i_bia : i_bia + mul]  # [mul]
                chunk += sub_bias.reshape(mul, 1)  # [batch, sample, mul, repr]
                i_bia += mul

            new_chunks.append(chunk)  # [batch, sample, mul, repr]
        i_wei += mul

    assert weight is None or i_wei == len(weight)
    assert ra_var is None or i_wei == len(ra_var)
    assert bias is None or i_bia == len(bias)
    assert ra_mean is None or i_rmu == len(ra_mean)

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
