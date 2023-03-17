import flax
import jax.numpy as jnp

import e3nn_jax as e3nn


class TensorProductWithSphericalHarmonicsLinear(flax.linen.Module):
    output_irreps: e3nn.Irreps

    @flax.linen.compact
    def __call__(self, input: e3nn.IrrepsArray, vector: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        output_irreps = e3nn.Irreps(self.output_irreps)
        assert input.shape == (input.irreps.dim,)
        assert vector.shape == (3,)

        # Calculate the rotation and align the input with the vector axis
        alpha, beta = e3nn.xyz_to_angles(vector.array)
        input = input.transform_by_angles(alpha, beta, 0.0, inverse=True)

        out = flax.linen.Dense(output_irreps.dim, use_bias=False, dtype=input.dtype)(input.array)
        out = e3nn.IrrepsArray(output_irreps, out)

        out = out.transform_by_angles(alpha, beta, 0.0)
        return out


def sl(lout: int, lin: int) -> slice:
    return slice(lout - lin, lout + lin + 1)


def is_diag(x: jnp.ndarray) -> bool:
    return jnp.allclose(jnp.diag(jnp.diag(x)), x)
