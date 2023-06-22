"""Implementation of the Linear Spherical Harmonics Tensor Product.

Motivated by the paper: https://arxiv.org/pdf/2302.03655.pdf

- Added the support of inversion symmetry. (Mario Geiger)
"""
from typing import Sequence

import flax
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn


class LinearSHTP(flax.linen.Module):
    r"""Linear Spherical Harmonics Tensor Product.

    Computes a linear combination linearly equivalent to the following.

    .. math::

        \sum_{l=0}^{\infty} w^l x \otimes Y^l(\vec d)

    where :math:`w^l` are some weights, :math:`x` is the input, :math:`Y^l` are the spherical harmonics
    of the direction :math:`\vec d`.

    Args:
        irreps_out: input irreps, acts as a filter if `mix` is False.
        mix: if True, the output is a linear combination of the input, otherwise each output
            is kept separate.
    """

    irreps_out: e3nn.Irreps
    mix: bool = True

    @flax.linen.compact
    def __call__(
        self, input: e3nn.IrrepsArray, direction: e3nn.IrrepsArray
    ) -> e3nn.IrrepsArray:
        assert input.shape == (input.irreps.dim,)
        assert direction.shape == (3,)

        assert len(direction.irreps) == 1
        assert direction.irreps.num_irreps == 1
        _, ird = direction.irreps[0]
        assert ird in ["1o", "1e"]
        direction = e3nn.IrrepsArray(direction.irreps, normalize(direction.array))

        # Avoid gimbal lock
        gimbal_lock = jnp.abs(direction.array[1]) > 0.99

        def fix_gimbal_lock(array, inverse):
            array_rot = array.transform_by_angles(
                0.0, jnp.pi / 2.0, 0.0, inverse=inverse
            )
            return jax.tree_util.tree_map(
                lambda x_rot, x: jnp.where(gimbal_lock, x_rot, x), array_rot, array
            )

        input = fix_gimbal_lock(input, inverse=True)
        direction = fix_gimbal_lock(direction, inverse=True)

        # Calculate the rotation and align the input with the vector axis
        alpha, beta = e3nn.xyz_to_angles(direction.array)
        input = input.transform_by_angles(alpha, beta, 0.0, inverse=True)

        irreps_out = e3nn.Irreps(self.irreps_out)
        if not self.mix:
            irreps_out = irreps_out.regroup()

        irreps_out_ = []
        outputs = []

        for iz, (mulz, irz) in enumerate(irreps_out):
            if self.mix:
                zs = []
                dim = 0  # for normalization
            else:
                mulz = None

            for (ix, (mulx, irx)), x in zip(enumerate(input.irreps), input.chunks):
                if x is None:
                    continue

                l = min(irx.l, irz.l)
                x = x[:, sl(irx.l, l)]

                py = irx.p * irz.p

                # symmetric part
                ly = (irx.l + irz.l) % 2
                if ird.p**ly == py:
                    w = self.param(
                        f"{ix}_{iz}_S{irx}{irz}",
                        flax.linen.initializers.normal(stddev=1.0),
                        (l + 1, mulx, mulz) if self.mix else (l + 1, mulx),
                        x.dtype,
                    )
                    w = jnp.concatenate([w[::-1], w[1:]])

                    if self.mix:
                        z = jnp.einsum("ui,iuv->vi", x, w)
                    else:
                        z = jnp.einsum("ui,iu->ui", x, w)

                    if l < irz.l:
                        zeros = jnp.zeros_like(z, shape=(z.shape[0], irz.dim))
                        z = zeros.at[:, sl(irz.l, l)].set(z)

                    if self.mix:
                        zs.append(z)
                        dim += mulx
                    else:
                        irreps_out_.append((z.shape[0], irz))
                        outputs.append(z)

                # antisymmetric part
                ly = (irx.l + irz.l + 1) % 2
                if ird.p**ly == py and l > 0:
                    w = self.param(
                        f"{ix}_{iz}_A{irx}{irz}",
                        flax.linen.initializers.normal(stddev=1.0),
                        (l, mulx, mulz) if self.mix else (l, mulx),
                        x.dtype,
                    )
                    zeros = jnp.zeros_like(w, shape=(1,) + w.shape[1:])
                    w = jnp.concatenate([-w[::-1], zeros, w])

                    if self.mix:
                        z = jnp.einsum("ui,iuv->vi", x[:, ::-1], w)
                    else:
                        z = jnp.einsum("ui,iu->ui", x[:, ::-1], w)

                    if l < irz.l:
                        zeros = jnp.zeros_like(z, shape=(z.shape[0], irz.dim))
                        z = zeros.at[:, sl(irz.l, l)].set(z)

                    if self.mix:
                        zs.append(z)
                        dim += mulx
                    else:
                        irreps_out_.append((z.shape[0], irz))
                        outputs.append(z)

            if self.mix:
                z = sum_tensors(
                    zs, (mulz, irz.dim), empty_return_none=True, dtype=x.dtype
                )
                if z is not None:
                    z = z / jnp.sqrt(dim)
                irreps_out_.append((z.shape[0], irz))
                outputs.append(z)

        out = e3nn.from_chunks(irreps_out_, outputs, (), input.dtype)
        out = out.regroup()

        # Rotate back
        out = out.transform_by_angles(alpha, beta, 0.0)

        # Avoid gimbal lock
        out = fix_gimbal_lock(out, inverse=False)

        return out


def shtp(
    input: e3nn.IrrepsArray,
    direction: e3nn.IrrepsArray,
    filter_irreps_out: Sequence[e3nn.Irrep],
) -> e3nn.IrrepsArray:
    assert input.shape == (input.irreps.dim,)
    assert direction.shape == (3,)

    assert len(direction.irreps) == 1
    assert direction.irreps.num_irreps == 1
    _, ird = direction.irreps[0]
    assert ird in ["1o", "1e"]
    direction = e3nn.IrrepsArray(direction.irreps, normalize(direction.array))

    # Avoid gimbal lock
    gimbal_lock = jnp.abs(direction.array[1]) > 0.99

    def fix_gimbal_lock(array, inverse):
        array_rot = array.transform_by_angles(0.0, jnp.pi / 2.0, 0.0, inverse=inverse)
        return jax.tree_util.tree_map(
            lambda x_rot, x: jnp.where(gimbal_lock, x_rot, x), array_rot, array
        )

    input = fix_gimbal_lock(input, inverse=True)
    direction = fix_gimbal_lock(direction, inverse=True)

    # Calculate the rotation and align the input with the vector axis
    alpha, beta = e3nn.xyz_to_angles(direction.array)
    input = input.transform_by_angles(alpha, beta, 0.0, inverse=True)

    if isinstance(filter_irreps_out, str):
        filter_irreps_out = e3nn.Irreps(filter_irreps_out)
    if isinstance(filter_irreps_out, e3nn.Irreps):
        filter_irreps_out = [ir for _, ir in filter_irreps_out]
    filter_irreps_out = list(filter_irreps_out)
    filter_irreps_out = sorted(set(filter_irreps_out))

    irreps_out = []
    outputs = []

    for irz in filter_irreps_out:
        for (mulx, irx), x in zip(input.irreps, input.chunks):
            if x is None:
                continue

            l = min(irx.l, irz.l)
            x = x[:, sl(irx.l, l)]

            py = irx.p * irz.p

            # symmetric part
            ly = (irx.l + irz.l) % 2
            if ird.p**ly == py:
                zeros = jnp.zeros_like(x, shape=(mulx, l + 1, irz.dim))
                z = zeros.at[
                    :, jnp.arange(l + 1), jnp.arange(irz.l, irz.l + l + 1)
                ].set(x[:, l:])
                z = z.at[:, jnp.arange(l, 0, -1), jnp.arange(irz.l - l, irz.l)].set(
                    x[:, :l]
                )
                z = jnp.reshape(z, (mulx * (l + 1), irz.dim))

                irreps_out.append((z.shape[0], irz))
                outputs.append(z)

            # antisymmetric part
            ly = (irx.l + irz.l + 1) % 2
            if ird.p**ly == py and l > 0:
                zeros = jnp.zeros_like(x, shape=(mulx, l, irz.dim))
                z = zeros.at[
                    :, jnp.arange(l), jnp.arange(irz.l - 1, irz.l - l - 1, -1)
                ].set(x[:, l + 1 :])
                z = z.at[:, jnp.arange(l), jnp.arange(irz.l + 1, irz.l + l + 1)].set(
                    -x[:, :l][:, ::-1]
                )
                z = jnp.reshape(z, (mulx * l, irz.dim))

                irreps_out.append((z.shape[0], irz))
                outputs.append(z)

    out = e3nn.from_chunks(irreps_out, outputs, (), x.dtype)
    out = out.regroup()

    # Rotate back
    out = out.transform_by_angles(alpha, beta, 0.0)

    # Avoid gimbal lock
    out = fix_gimbal_lock(out, inverse=False)

    return out


def sum_tensors(xs, shape, empty_return_none=False, dtype=None):
    xs = [x for x in xs if x is not None]
    if len(xs) > 0:
        out = xs[0].reshape(shape)
        for x in xs[1:]:
            out = out + x.reshape(shape)
        return out
    if empty_return_none:
        return None
    return jnp.zeros(shape, dtype=dtype)


def normalize(x):
    n2 = jnp.sum(x**2, axis=-1, keepdims=True)
    n2 = jnp.where(n2 > 0.0, n2, 1.0)
    return x / jnp.sqrt(n2)


def sl(l_data: int, l_extracted: int) -> slice:
    return slice(l_data - l_extracted, l_data + l_extracted + 1)
