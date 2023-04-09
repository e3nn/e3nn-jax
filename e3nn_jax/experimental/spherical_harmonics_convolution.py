"""Implementation of the spherical harmonics convolution in Flax.

Motivated by the paper: https://arxiv.org/pdf/2302.03655.pdf

- Added the support of inversion symmetry. (Mario Geiger)
"""
import flax
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn


class SHConvolutionFlax(flax.linen.Module):
    irreps_out: e3nn.Irreps

    @flax.linen.compact
    def __call__(self, input: e3nn.IrrepsArray, direction: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
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
            return jax.tree_util.tree_map(lambda x_rot, x: jnp.where(gimbal_lock, x_rot, x), array_rot, array)

        input = fix_gimbal_lock(input, inverse=True)
        direction = fix_gimbal_lock(direction, inverse=True)

        # Calculate the rotation and align the input with the vector axis
        alpha, beta = e3nn.xyz_to_angles(direction.array)
        input = input.transform_by_angles(alpha, beta, 0.0, inverse=True)

        irreps_out = e3nn.Irreps(self.irreps_out)
        outputs = []

        for mulz, irz in irreps_out:
            zs = []
            dim = 0  # for normalization

            for (mulx, irx), x in zip(input.irreps, input.list):
                if x is None:
                    continue

                l = min(irx.l, irz.l)
                x = x[:, sl(irx.l, l)]

                py = irx.p * irz.p

                # symmetric part
                ly = (irx.l + irz.l) % 2
                if ird.p**ly == py:
                    w = self.param(f"S{irx}{irz}", flax.linen.initializers.normal(stddev=1.0), (l + 1, mulx, mulz), x.dtype)
                    w = jnp.concatenate([w[::-1], w[1:]])
                    assert w.shape == (2 * l + 1, mulx, mulz)

                    z = jnp.einsum("ui,iuv->vi", x, w)
                    dim += mulx

                    if l < irz.l:
                        zeros = jnp.zeros_like(z, shape=(mulz, irz.dim))
                        z = zeros.at[:, sl(irz.l, l)].set(z)

                    zs.append(z)

                # antisymmetric part
                ly = (irx.l + irz.l + 1) % 2
                if ird.p**ly == py:
                    w = self.param(f"A{irx}{irz}", flax.linen.initializers.normal(stddev=1.0), (l, mulx, mulz), x.dtype)
                    zeros = jnp.zeros_like(w, shape=(1, mulx, mulz))
                    w = jnp.concatenate([-w[::-1], zeros, w])
                    assert w.shape == (2 * l + 1, mulx, mulz)

                    z = jnp.einsum("ui,iuv->vi", x[:, ::-1], w)
                    dim += mulx

                    if l < irz.l:
                        zeros = jnp.zeros_like(z, shape=(mulz, irz.dim))
                        z = zeros.at[:, sl(irz.l, l)].set(z)

                    zs.append(z)

            z = sum_tensors(zs, (mulz, irz.dim), empty_return_none=True, dtype=x.dtype)
            if z is not None:
                z = z / jnp.sqrt(dim)
            outputs.append(z)

        out = e3nn.IrrepsArray.from_list(irreps_out, outputs, (), x.dtype)

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
