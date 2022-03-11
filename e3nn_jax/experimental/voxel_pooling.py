from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def lowpass_filter(input, scale, strides, transposed=False, steps=(1, 1, 1)):
    r"""Lowpass filter for 3D field.

    Args:
        input: [..., x, y, z]
        scale (float): typically 2.0
        strides (int): typically 1 or 2
        transposed (bool): if True, dilate the input instead of stride
        steps (tuple): physical dimensions of the voxel grid

    Known issues:
           stride=2    transposed=True
        64 -------> 32 --------------> 63
    """

    if isinstance(strides, int):
        strides = (strides,) * 3

    with jax.core.eval_context():
        sigma = 0.5 * (scale ** 2 - 1)**0.5

        size = int(1 + 2 * 2.5 * sigma)
        if size % 2 == 0:
            size += 1

        r = jnp.linspace(-1, 1, size)
        x = r * steps[0] / min(steps)
        x = x[jnp.abs(x) <= 1]
        y = r * steps[1] / min(steps)
        y = y[jnp.abs(y) <= 1]
        z = r * steps[2] / min(steps)
        z = z[jnp.abs(z) <= 1]
        lattice = jnp.stack(jnp.meshgrid(x, y, z, indexing='ij'), axis=-1)  # [x, y, z, R^3]
        lattice = (size // 2) * lattice

        kernel = jnp.exp(-jnp.sum(lattice**2, axis=-1) / (2 * sigma**2))
        kernel = kernel / jnp.sum(kernel)

        if transposed:
            kernel = kernel * strides[0] * strides[1] * strides[2]

        kernel = kernel[None, None]  # [1, 1, x, y, z]

    if scale <= 1:
        assert strides == (1,) * 3
        return input

    pad = (kernel.shape[-3] // 2, kernel.shape[-2] // 2, kernel.shape[-1] // 2)

    output = input
    output = output.reshape(-1, 1, *output.shape[-3:])
    output = lax.conv_general_dilated(
        lhs=output,
        rhs=kernel,
        lhs_dilation=strides if transposed else (1, 1, 1),
        window_strides=(1, 1, 1) if transposed else strides,
        padding=((pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2])),
        dimension_numbers=('NCXYZ', 'IOXYZ', 'NCXYZ')
    )

    output = output.reshape(*input.shape[:-3], *output.shape[-3:])
    return output


@jax.jit
def interpolate_bilinear(input, x, y, z):
    r"""interpolate voxels in coordinate (x, y, z).

    Args:
        input: [..., x, y, z]
        x: x coordinate
        y: y coordinate
        z: z coordinate
    """
    # based on http://stackoverflow.com/a/12729229
    x_lo = jnp.floor(x).astype(int)
    x_hi = x_lo + 1
    y_lo = jnp.floor(y).astype(int)
    y_hi = y_lo + 1
    z_lo = jnp.floor(z).astype(int)
    z_hi = z_lo + 1

    nx, ny, nz = input.shape[-3:]

    def xclip(x):
        return jnp.clip(x, 0, nx - 1)

    def yclip(y):
        return jnp.clip(y, 0, ny - 1)

    def zclip(z):
        return jnp.clip(z, 0, nz - 1)

    Ia = input[..., xclip(x_lo), yclip(y_lo), zclip(z_lo)]
    Ib = input[..., xclip(x_hi), yclip(y_lo), zclip(z_lo)]
    Ic = input[..., xclip(x_lo), yclip(y_hi), zclip(z_lo)]
    Id = input[..., xclip(x_hi), yclip(y_hi), zclip(z_lo)]
    Ie = input[..., xclip(x_lo), yclip(y_lo), zclip(z_hi)]
    If = input[..., xclip(x_hi), yclip(y_lo), zclip(z_hi)]
    Ig = input[..., xclip(x_lo), yclip(y_hi), zclip(z_hi)]
    Ih = input[..., xclip(x_hi), yclip(y_hi), zclip(z_hi)]

    wa = (x_hi - x) * (y_hi - y) * (z_hi - z)
    wb = (x - x_lo) * (y_hi - y) * (z_hi - z)
    wc = (x_hi - x) * (y - y_lo) * (z_hi - z)
    wd = (x - x_lo) * (y - y_lo) * (z_hi - z)
    we = (x_hi - x) * (y_hi - y) * (z - z_lo)
    wf = (x - x_lo) * (y_hi - y) * (z - z_lo)
    wg = (x_hi - x) * (y - y_lo) * (z - z_lo)
    wh = (x - x_lo) * (y - y_lo) * (z - z_lo)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id + we * Ie + wf * If + wg * Ig + wh * Ih


@partial(jax.jit, static_argnums=(1,))
def zoom(input, resize_rate):
    r"""Rescale the input by a factor of `resize_rate`.

    Args:
        input: [..., x, y, z]
        resize_rate (float): typically 2.0 or 0.5

    Returns:
        resize_rate times larger field
    """
    nx, ny, nz = input.shape[-3:]

    if isinstance(resize_rate, int):
        resize_rate = (resize_rate,) * 3

    def f(n_src, n_dst):
        a = n_src / n_dst * jnp.arange(n_dst)
        # delta = 0.5 * (n_src / n_dst - 1)
        # return delta + a
        return a  # the offset to the left make it compatible with the stride

    xi = f(nx, round(nx * resize_rate[0]))
    yi = f(ny, round(ny * resize_rate[1]))
    zi = f(nz, round(nz * resize_rate[2]))

    xg, yg, zg = jnp.meshgrid(xi, yi, zi, indexing='ij')

    output = jax.vmap(interpolate_bilinear, (None, 0, 0, 0), -1)(input, xg.flatten(), yg.flatten(), zg.flatten())
    output = output.reshape(*input.shape[:-3], len(xi), len(yi), len(zi))

    return output
