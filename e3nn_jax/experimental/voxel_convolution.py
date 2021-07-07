import math
from typing import Callable, Tuple

import flax
import jax
import jax.numpy as jnp
from e3nn_jax import (Irreps, fully_connected_tensor_product, linear,
                      soft_one_hot_linspace, spherical_harmonics)
from jax import lax


class Convolution(flax.linen.Module):
    irreps_in: Irreps
    irreps_out: Irreps
    irreps_sh: Irreps
    diameter: float
    num_radial_basis: int
    steps: Tuple[float, float, float]
    weight_init: Callable = jax.random.normal

    @flax.linen.compact
    def __call__(self, x):
        """
        x: [batch, x, y, z, irreps_in.dim]
        """
        def gen_lattice():
            r = self.diameter / 2

            s = math.floor(r / self.steps[0])
            x = jnp.arange(-s, s + 1.0) * self.steps[0]

            s = math.floor(r / self.steps[1])
            y = jnp.arange(-s, s + 1.0) * self.steps[1]

            s = math.floor(r / self.steps[2])
            z = jnp.arange(-s, s + 1.0) * self.steps[2]

            return jnp.stack(jnp.meshgrid(x, y, z, indexing='ij'), axis=-1)  # [x, y, z, R^3]

        lattice = self.variable('consts', 'lattice', gen_lattice).value

        def gen_emb(lattice):
            return soft_one_hot_linspace(
                x=jnp.linalg.norm(lattice, ord=2, axis=-1),
                start=0.0,
                end=self.diameter / 2,
                number=self.num_radial_basis,
                basis='smooth_finite',
                cutoff=True,
            )  # [x, y, z, num_radial_basis]

        emb = self.variable('consts', 'emb', gen_emb, lattice).value

        def gen_sh(lattice):
            return spherical_harmonics(
                irreps_out=self.irreps_sh,
                x=lattice,
                normalize=True,
                normalization='component'
            )  # [x, y, z, irreps_sh.dim]

        sh = self.variable('consts', 'sh', gen_sh, lattice).value

        # self-connection
        _, nw, _, f = linear(self.irreps_in, self.irreps_out)
        f = jax.vmap(f, (None, None, 0), 0)
        w = self.param('linear_weight', self.weight_init, (nw,))
        sc = f(w, jnp.ones((0,)), x.reshape(-1, x.shape[-1]))
        sc = sc.reshape(x.shape[:-1] + (-1,))

        # convolution
        _, nw, _, tp_right = fully_connected_tensor_product(self.irreps_in, self.irreps_sh, self.irreps_out)
        tp_right = jax.vmap(tp_right, (0, 0), 0)
        w = self.param('weight', self.weight_init, (self.num_radial_basis, nw))
        w = emb @ w  # [x,y,z, tp_w]
        w = w / (sh.shape[0] * sh.shape[1] * sh.shape[2])
        k = tp_right(w.reshape(-1, w.shape[-1]), sh.reshape(-1, sh.shape[-1]))
        k = k.reshape(sh.shape[:3] + k.shape[1:])  # [x,y,z, irreps_in.dim, irreps_out.dim]
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=k,
            window_strides=(1, 1, 1),
            padding='SAME',
            dimension_numbers=('NXYZC', 'XYZIO', 'NXYZC')
        )

        return sc + x
