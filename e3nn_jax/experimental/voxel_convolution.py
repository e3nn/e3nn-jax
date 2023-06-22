import math
from typing import Dict, Optional, Tuple, Union

import flax
import haiku as hk
import jax.numpy as jnp
from jax import lax

import e3nn_jax as e3nn


class ConvolutionHaiku(hk.Module):
    r"""3D Voxel Convolution.

    Args:
        irreps_out: Irreps of the output.
        irreps_sh: Irreps of the spherical harmonics.
        diameter: Diameter of convolution kernel expressed in any physical unit, let say millimeter.
        num_radial_basis: Number of radial basis functions, optionally for each spherical harmonic order.
        steps: Steps of the pixel grid in millimeter. For instance 1mm by 1mm by 3mm.
        relative_starts: Relative start of the radial basis functions, optionally for each spherical harmonic order.
        padding: Padding mode, see `lax.conv_general_dilated`.
    """

    def __init__(
        self,
        irreps_out: e3nn.Irreps,
        irreps_sh: e3nn.Irreps,
        diameter: float,
        num_radial_basis: Union[int, Dict[int, int]],
        steps: Tuple[float, float, float],
        *,
        relative_starts: Union[float, Dict[int, float]] = 0.0,
        padding: str = "SAME",
    ):
        super().__init__()

        self.num_radial_basis = num_radial_basis
        self.relative_starts = relative_starts
        self.irreps_out = irreps_out
        self.irreps_sh = irreps_sh
        self.diameter = diameter
        self.padding = padding
        self.steps = steps

    def kernel(
        self,
        irreps_in: e3nn.Irreps,
        irreps_out: e3nn.Irreps,
        steps: Optional[jnp.ndarray] = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        r"""Compute the convolution kernel.

        Args:
            irreps_in: Irreps of the input.
            irreps_out: Irreps of the output.
            steps: dynamic steps, if None use the static steps
            dtype: Data type of the kernel.
        """

        def _get_params(name: str, shape: Tuple[int, ...], weight_std: float):
            return hk.get_parameter(
                name,
                shape=shape,
                init=hk.initializers.RandomNormal(weight_std),
                dtype=dtype,
            )

        return _kernel(self, irreps_in, irreps_out, steps, _get_params, dtype)

    def __call__(
        self, input: e3nn.IrrepsArray, steps: Optional[jnp.ndarray] = None
    ) -> e3nn.IrrepsArray:
        r"""Evaluate the convolution.

        Args:
            input: Input data of shape ``[batch, x, y, z, irreps_in.dim]``
            steps: dynamic steps, if None use the static steps

        Returns:
            Output data of shape ``[batch, x, y, z, irreps_out.dim]``
        """
        return _call(self, input, steps)


class ConvolutionFlax(flax.linen.Module):
    irreps_out: e3nn.Irreps
    irreps_sh: e3nn.Irreps
    diameter: float
    num_radial_basis: Union[int, Dict[int, int]]
    steps: Tuple[float, float, float]
    relative_starts: Union[float, Dict[int, float]] = 0.0
    padding: str = "SAME"

    def kernel(
        self,
        irreps_in: e3nn.Irreps,
        irreps_out: e3nn.Irreps,
        steps: Optional[jnp.ndarray] = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        def _get_params(name: str, shape: Tuple[int, ...], weight_std: float):
            return self.param(
                name, flax.linen.initializers.normal(stddev=weight_std), shape, dtype
            )

        return _kernel(self, irreps_in, irreps_out, steps, _get_params, dtype)

    @flax.linen.compact
    def __call__(
        self, input: e3nn.IrrepsArray, steps: Optional[jnp.ndarray] = None
    ) -> e3nn.IrrepsArray:
        return _call(self, input, steps)


ConvolutionFlax.__doc__ = ConvolutionHaiku.__doc__
ConvolutionFlax.kernel.__doc__ = ConvolutionHaiku.kernel.__doc__
ConvolutionFlax.__call__.__doc__ = ConvolutionHaiku.__call__.__doc__


def _tp_weight(
    self: Union[ConvolutionHaiku, ConvolutionFlax],
    lattice: jnp.ndarray,
    i_in: int,
    i_sh: int,
    i_out: int,
    mul_ir_in: e3nn.MulIrrep,
    ir_sh: e3nn.Irrep,
    mul_ir_out: e3nn.MulIrrep,
    path_shape: Tuple[int, ...],
    weight_std: float,
    get_parameter,
) -> jnp.ndarray:
    number = (
        self.num_radial_basis
        if isinstance(self.num_radial_basis, int)
        else self.num_radial_basis[ir_sh.l]
    )
    start = (
        self.relative_starts
        if isinstance(self.relative_starts, (float, int))
        else self.relative_starts[ir_sh.l]
    )

    embedding = e3nn.soft_one_hot_linspace(
        jnp.linalg.norm(lattice, ord=2, axis=-1),
        start=start * self.diameter / 2,
        end=self.diameter / 2,
        number=number,
        basis="smooth_finite",
        start_zero=True,
        end_zero=True,
    )  # [x, y, z, number]

    w = get_parameter(
        f"w[{i_in},{i_sh},{i_out}] {mul_ir_in},{ir_sh},{mul_ir_out}",
        (number,) + path_shape,
        weight_std,
    )

    return jnp.einsum("xyzk,k...->xyz...", embedding, w) / (
        lattice.shape[0] * lattice.shape[1] * lattice.shape[2]
    )  # [x, y, z, tp_w]


def _kernel(
    self: Union[ConvolutionHaiku, ConvolutionFlax],
    irreps_in: e3nn.Irreps,
    irreps_out: e3nn.Irreps,
    steps: Optional[jnp.ndarray],
    get_parameter,
    dtype,
) -> jnp.ndarray:
    if steps is None:
        steps = self.steps

    r = self.diameter / 2

    s = math.floor(r / self.steps[0])
    x = jnp.arange(-s, s + 1.0) * steps[0]

    s = math.floor(r / self.steps[1])
    y = jnp.arange(-s, s + 1.0) * steps[1]

    s = math.floor(r / self.steps[2])
    z = jnp.arange(-s, s + 1.0) * steps[2]

    lattice = jnp.stack(jnp.meshgrid(x, y, z, indexing="ij"), axis=-1)  # [x, y, z, R^3]
    lattice = lattice.astype(dtype)

    # convolution kernel
    tp = e3nn.FunctionalFullyConnectedTensorProduct(
        irreps_in, self.irreps_sh, irreps_out
    )

    ws = [
        _tp_weight(
            self,
            lattice,
            i.i_in1,
            i.i_in2,
            i.i_out,
            tp.irreps_in1[i.i_in1],
            tp.irreps_in2[i.i_in2].ir,
            tp.irreps_out[i.i_out],
            i.path_shape,
            i.weight_std,
            get_parameter,
        )
        for i in tp.instructions
    ]

    sh = e3nn.spherical_harmonics(
        irreps_out=self.irreps_sh, input=lattice, normalize=True
    )  # [x, y, z, irreps_sh.dim]

    tp_right = tp.right
    for _ in range(3):
        tp_right = e3nn.utils.vmap(tp_right, (0, 0), 0)
    k = tp_right(ws, sh)  # [x, y, z, irreps_in.dim, irreps_out.dim]

    # self-connection, center of the kernel
    lin = e3nn.FunctionalLinear(irreps_in, irreps_out)
    w = [
        get_parameter(
            f"self-connection[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},{lin.irreps_out[ins.i_out]}",
            ins.path_shape,
            ins.weight_std,
        )
        for ins in lin.instructions
    ]
    # note that lattice[center] is always displacement zero
    k = k.at[k.shape[0] // 2, k.shape[1] // 2, k.shape[2] // 2].set(lin.matrix(w))
    return k


def _call(
    self: Union[ConvolutionHaiku, ConvolutionFlax],
    input: e3nn.IrrepsArray,
    steps: Optional[jnp.ndarray] = None,
) -> e3nn.IrrepsArray:
    if not isinstance(input, e3nn.IrrepsArray):
        raise ValueError("Convolution: input should be of type IrrepsArray")

    input = input.remove_zero_chunks().simplify()

    irreps_out = e3nn.Irreps(
        [
            (mul, ir)
            for (mul, ir) in e3nn.Irreps(self.irreps_out)
            if any(
                ir in ir_in * ir_sh
                for _, ir_in in input.irreps
                for _, ir_sh in e3nn.Irreps(self.irreps_sh)
            )
        ]
    )

    output = e3nn.IrrepsArray(
        irreps_out,
        lax.conv_general_dilated(
            lhs=input.array,
            rhs=self.kernel(input.irreps, irreps_out, steps, input.dtype),
            window_strides=(1, 1, 1),
            padding=self.padding,
            dimension_numbers=("NXYZC", "XYZIO", "NXYZC"),
        ),
    )

    if irreps_out != e3nn.Irreps(self.irreps_out):
        list = []
        i = 0
        for mul_ir in e3nn.Irreps(self.irreps_out):
            if i < len(irreps_out) and irreps_out[i] == mul_ir:
                list.append(output.chunks[i])
                i += 1
            else:
                list.append(None)
        output = e3nn.from_chunks(
            self.irreps_out, list, output.shape[:-1], output.dtype
        )

    return output
