import math
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import (
    FunctionalFullyConnectedTensorProduct,
    Linear,
    Irreps,
    IrrepsData,
    soft_one_hot_linspace,
    spherical_harmonics,
)
from jax import lax


class Convolution(hk.Module):
    def __init__(
        self,
        irreps_out,
        irreps_sh,
        diameter: float,
        num_radial_basis: int,
        steps: Tuple[float, float, float],
        *,
        irreps_in=None,
    ):
        super().__init__()

        self.irreps_in = Irreps(irreps_in) if irreps_in is not None else None
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

            lattice = jnp.stack(jnp.meshgrid(x, y, z, indexing="ij"), axis=-1)  # [x, y, z, R^3]

            self.emb = soft_one_hot_linspace(
                jnp.linalg.norm(lattice, ord=2, axis=-1),
                start=0.0,
                end=self.diameter / 2,
                number=self.num_radial_basis,
                basis="smooth_finite",
                cutoff=True,
            )  # [x, y, z, num_radial_basis]

            self.sh = spherical_harmonics(
                irreps_out=self.irreps_sh, input=lattice, normalize=True, normalization="component"
            )  # [x, y, z, irreps_sh.dim]

    def __call__(self, x: IrrepsData) -> IrrepsData:
        """
        x: [batch, x, y, z, irreps_in.dim]
        """
        if self.irreps_in is not None:
            x = IrrepsData.new(self.irreps_in, x)
        if not isinstance(x, IrrepsData):
            raise ValueError("Convolution: input should be of type IrrepsData")

        x = x.remove_nones().simplify()

        # self-connection
        lin = Linear(self.irreps_out)
        for _ in range(1 + 3):
            lin = jax.vmap(lin)
        sc = lin(x)

        irreps_out = Irreps(
            [
                (mul, ir)
                for (mul, ir) in self.irreps_out
                if any(ir in ir_in * ir_sh for _, ir_in in x.irreps for _, ir_sh in self.irreps_sh)
            ]
        )

        # convolution
        tp = FunctionalFullyConnectedTensorProduct(x.irreps, self.irreps_sh, irreps_out)

        w = [
            hk.get_parameter(
                f"w[{i.i_in1},{i.i_in2},{i.i_out}] {tp.irreps_in1[i.i_in1]},{tp.irreps_in2[i.i_in2]},{tp.irreps_out[i.i_out]}",
                (self.num_radial_basis,) + i.path_shape,
                init=hk.initializers.RandomNormal(),
            )
            for i in tp.instructions
        ]
        w = [
            jnp.einsum("xyzk,k...->xyz...", self.emb, x)
            / (self.sh.shape[0] * self.sh.shape[1] * self.sh.shape[2])  # [x,y,z, tp_w]
            for x in w
        ]

        tp_right = tp.right
        for _ in range(3):
            tp_right = jax.vmap(tp_right, (0, 0), 0)
        k = tp_right(w, self.sh)  # [x,y,z, irreps_in.dim, irreps_out.dim]

        x = IrrepsData.from_contiguous(
            irreps_out,
            lax.conv_general_dilated(
                lhs=x.contiguous,
                rhs=k,
                window_strides=(1, 1, 1),
                padding="SAME",
                dimension_numbers=("NXYZC", "XYZIO", "NXYZC"),
            ),
        )

        if irreps_out != self.irreps_out:
            list = []
            i = 0
            for mul_ir in self.irreps_out:
                if i < len(irreps_out) and irreps_out[i] == mul_ir:
                    list.append(x.list[i])
                    i += 1
                else:
                    list.append(None)
            x = IrrepsData.from_list(self.irreps_out, list, x.shape)

        return sc + x
