import math
from typing import Tuple, Dict, Union

from collections import defaultdict
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import (
    FunctionalFullyConnectedTensorProduct,
    FunctionalLinear,
    Irrep,
    Irreps,
    IrrepsArray,
    MulIrrep,
    soft_one_hot_linspace,
    spherical_harmonics,
)
from jax import lax


class Convolution(hk.Module):
    def __init__(
        self,
        irreps_out: Irreps,
        irreps_sh: Irreps,
        diameter: float,
        num_radial_basis: Union[int, Dict[int, int]],
        steps: Union[Tuple[float, float, float], Tuple[Tuple[float, float, float], jnp.ndarray]],
        *,
        relative_starts: Union[float, Dict[int, float]] = 0.0,
        padding="SAME",
        irreps_in=None,
    ):
        r"""3D Voxel Convolution.

        Args:
            irreps_out: Irreps of the output.
            irreps_sh: Irreps of the spherical harmonics.
            diameter: Diameter of convolution kernel expressed in any physical unit, let say millimeter.
            num_radial_basis: Number of radial basis functions, optionally for each spherical harmonic order.
            steps: Steps of the pixel grid in millimeter. For instance 1mm by 1mm by 3mm.
            relative_starts: Relative start of the radial basis functions, optionally for each spherical harmonic order.
            padding: Padding mode, see `lax.conv_general_dilated`.
            irreps_in: Irreps of the input. If None, it is inferred from the input data.
        """
        super().__init__()

        if isinstance(num_radial_basis, int):
            self.num_radial_basis = defaultdict(lambda: num_radial_basis)
        else:
            self.num_radial_basis = num_radial_basis

        if isinstance(relative_starts, (float, int)):
            self.relative_starts = defaultdict(lambda: relative_starts)
        else:
            self.relative_starts = relative_starts

        self.irreps_in = Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_out = Irreps(irreps_out)
        self.irreps_sh = Irreps(irreps_sh)
        self.diameter = diameter
        self.padding = padding

        def f(steps1, steps2):
            r = self.diameter / 2

            s = math.floor(r / steps1[0])
            x = jnp.arange(-s, s + 1.0) * steps2[0]

            s = math.floor(r / steps1[1])
            y = jnp.arange(-s, s + 1.0) * steps2[1]

            s = math.floor(r / steps1[2])
            z = jnp.arange(-s, s + 1.0) * steps2[2]

            self.lattice = jnp.stack(jnp.meshgrid(x, y, z, indexing="ij"), axis=-1)  # [x, y, z, R^3]

            self.sh = spherical_harmonics(
                irreps_out=self.irreps_sh, input=self.lattice, normalize=True
            )  # [x, y, z, irreps_sh.dim]

        if isinstance(steps[0], tuple):
            f(steps[0], steps[1])
        else:
            with jax.ensure_compile_time_eval():
                f(steps, steps)

    def tp_weight(
        self,
        i_in: int,
        i_sh: int,
        i_out: int,
        mul_ir_in: MulIrrep,
        ir_sh: Irrep,
        mul_ir_out: MulIrrep,
        path_shape: Tuple[int, ...],
        weight_std: float,
    ) -> jnp.ndarray:
        number = self.num_radial_basis[ir_sh.l]
        start = self.relative_starts[ir_sh.l]

        with jax.ensure_compile_time_eval():
            embedding = soft_one_hot_linspace(
                jnp.linalg.norm(self.lattice, ord=2, axis=-1),
                start=start * self.diameter / 2,
                end=self.diameter / 2,
                number=number,
                basis="smooth_finite",
                start_zero=True,
                end_zero=True,
            )  # [x, y, z, number]

        w = hk.get_parameter(
            f"w[{i_in},{i_sh},{i_out}] {mul_ir_in},{ir_sh},{mul_ir_out}",
            (number,) + path_shape,
            init=hk.initializers.RandomNormal(weight_std),
        )
        return jnp.einsum("xyzk,k...->xyz...", embedding, w) / (
            self.lattice.shape[0] * self.lattice.shape[1] * self.lattice.shape[2]
        )  # [x, y, z, tp_w]

    def kernel(self, irreps_in: Irreps, irreps_out: Irreps) -> jnp.ndarray:
        # convolution
        tp = FunctionalFullyConnectedTensorProduct(irreps_in, self.irreps_sh, irreps_out)

        ws = [
            self.tp_weight(
                i.i_in1,
                i.i_in2,
                i.i_out,
                tp.irreps_in1[i.i_in1],
                tp.irreps_in2[i.i_in2].ir,
                tp.irreps_out[i.i_out],
                i.path_shape,
                i.weight_std,
            )
            for i in tp.instructions
        ]

        tp_right = tp.right
        for _ in range(3):
            tp_right = jax.vmap(tp_right, (0, 0), 0)
        k = tp_right(ws, self.sh)  # [x, y, z, irreps_in.dim, irreps_out.dim]

        # self-connection
        lin = FunctionalLinear(irreps_in, irreps_out)
        w = [
            hk.get_parameter(
                f"self-connection[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}",
                shape=ins.path_shape,
                init=hk.initializers.RandomNormal(ins.weight_std),
            )
            for ins in lin.instructions
        ]
        # note that lattice[center] is always displacement zero
        k = k.at[k.shape[0] // 2, k.shape[1] // 2, k.shape[2] // 2].set(lin.matrix(w))
        return k

    def __call__(self, input: IrrepsArray) -> IrrepsArray:
        r"""
        Args:
            input: Input data of shape ``[batch, x, y, z, irreps_in.dim]``

        Returns:
            Output data of shape ``[batch, x, y, z, irreps_out.dim]``
        """
        if self.irreps_in is not None:
            input = IrrepsArray.from_any(self.irreps_in, input)
        if not isinstance(input, IrrepsArray):
            raise ValueError("Convolution: input should be of type IrrepsArray")

        input = input.remove_nones().simplify()

        irreps_out = Irreps(
            [
                (mul, ir)
                for (mul, ir) in self.irreps_out
                if any(ir in ir_in * ir_sh for _, ir_in in input.irreps for _, ir_sh in self.irreps_sh)
            ]
        )

        output = IrrepsArray(
            irreps_out,
            lax.conv_general_dilated(
                lhs=input.array,
                rhs=self.kernel(input.irreps, irreps_out),
                window_strides=(1, 1, 1),
                padding=self.padding,
                dimension_numbers=("NXYZC", "XYZIO", "NXYZC"),
            ),
        )

        if irreps_out != self.irreps_out:
            list = []
            i = 0
            for mul_ir in self.irreps_out:
                if i < len(irreps_out) and irreps_out[i] == mul_ir:
                    list.append(output.list[i])
                    i += 1
                else:
                    list.append(None)
            output = IrrepsArray.from_list(self.irreps_out, list, output.shape)

        return output
