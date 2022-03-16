import math
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import (FunctionalFullyConnectedTensorProduct, Irreps, FunctionalLinear,
                      soft_one_hot_linspace, spherical_harmonics)
from jax import lax


class Convolution(hk.Module):
    def __init__(self, irreps_in, irreps_out, irreps_sh, diameter: float, num_radial_basis: int, steps: Tuple[float, float, float]):
        super().__init__()

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.irreps_sh = Irreps(irreps_sh)
        self.diameter = diameter
        self.num_radial_basis = num_radial_basis
        self.steps = steps

        with jax.ensure_compile_time_eval():
            r = self.diameter / 2

            s = math.floor(r / self.steps[0])
            x = jnp.arange(-s, s + 1.0) * self.steps[0]

            s = math.floor(r / self.steps[1])
            y = jnp.arange(-s, s + 1.0) * self.steps[1]

            s = math.floor(r / self.steps[2])
            z = jnp.arange(-s, s + 1.0) * self.steps[2]

            lattice = jnp.stack(jnp.meshgrid(x, y, z, indexing='ij'), axis=-1)  # [x, y, z, R^3]

            self.emb = soft_one_hot_linspace(
                x=jnp.linalg.norm(lattice, ord=2, axis=-1),
                start=0.0,
                end=self.diameter / 2,
                number=self.num_radial_basis,
                basis='smooth_finite',
                cutoff=True,
            )  # [x, y, z, num_radial_basis]

            self.sh = spherical_harmonics(
                irreps_out=self.irreps_sh,
                x=lattice,
                normalize=True,
                normalization='component'
            )  # [x, y, z, irreps_sh.dim]

    def __call__(self, x):
        """
        x: [batch, x, y, z, irreps_in.dim]
        """

        # self-connection
        lin = FunctionalLinear(self.irreps_in, self.irreps_out)
        f = jax.vmap(lin, (None, 0), 0)
        w = [
            hk.get_parameter(
                f'linear_weight {i.i_in} -> {i.i_out}',
                i.path_shape,
                x.dtype,
                hk.initializers.RandomNormal()
            )
            for i in lin.instructions
        ]
        sc = f(w, x.reshape(-1, x.shape[-1])).contiguous
        sc = sc.reshape(x.shape[:-1] + (-1,))

        # convolution
        tp = FunctionalFullyConnectedTensorProduct(self.irreps_in, self.irreps_sh, self.irreps_out)

        tp_right = tp.right
        for _ in range(3):
            tp_right = jax.vmap(tp_right, (0, 0), 0)

        w = [
            hk.get_parameter(
                f'weight {i.i_in1} x {i.i_in2} -> {i.i_out}',
                (self.num_radial_basis,) + i.path_shape,
                x.dtype,
                hk.initializers.RandomNormal()
            )
            for i in tp.instructions
        ]
        w = [
            jnp.einsum("xyzk,k...->xyz...", self.emb, x) / (self.sh.shape[0] * self.sh.shape[1] * self.sh.shape[2])  # [x,y,z, tp_w]
            for x in w
        ]
        k = tp_right(w, self.sh)  # [x,y,z, irreps_in.dim, irreps_out.dim]
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=k,
            window_strides=(1, 1, 1),
            padding='SAME',
            dimension_numbers=('NXYZC', 'XYZIO', 'NXYZC')
        )

        return sc + x
