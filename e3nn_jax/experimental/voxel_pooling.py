from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def interpolate_trilinear(
    input: jnp.ndarray, x: float, y: float, z: float
) -> jnp.ndarray:
    r"""Interpolate voxels in coordinate (x, y, z).

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


def interpolate_nearest(
    input: jnp.ndarray, x: float, y: float, z: float
) -> jnp.ndarray:
    r"""Interpolate voxels in coordinate (x, y, z).

    Args:
        input: [..., x, y, z]
        x: x coordinate
        y: y coordinate
        z: z coordinate
    """
    x = jnp.round(x).astype(int)
    y = jnp.round(y).astype(int)
    z = jnp.round(z).astype(int)

    nx, ny, nz = input.shape[-3:]

    def xclip(x):
        return jnp.clip(x, 0, nx - 1)

    def yclip(y):
        return jnp.clip(y, 0, ny - 1)

    def zclip(z):
        return jnp.clip(z, 0, nz - 1)

    return input[..., xclip(x), yclip(y), zclip(z)]


@partial(jax.jit, static_argnums=(1, 2))
def _zoom(
    input: jnp.ndarray,
    output_size: Tuple[int, int, int],
    interpolation="linear",
) -> jnp.ndarray:
    nx, ny, nz = input.shape[-3:]

    def f(n_src, n_dst):
        # new pixel positions in source coordinate
        a = n_src / n_dst * jnp.arange(n_dst)
        delta = 0.5 * (n_src / n_dst - 1)
        return delta + a

    xi = f(nx, output_size[0])
    yi = f(ny, output_size[1])
    zi = f(nz, output_size[2])

    xg, yg, zg = jnp.meshgrid(xi, yi, zi, indexing="ij")

    if interpolation == "linear":
        interp = interpolate_trilinear
    if interpolation == "nearest":
        interp = interpolate_nearest

    output = jax.vmap(interp, (None, 0, 0, 0), -1)(
        input, xg.flatten(), yg.flatten(), zg.flatten()
    )
    output = output.reshape(*input.shape[:-3], *output_size)
    return output


def zoom(
    input: jnp.ndarray,
    *,
    resize_rate: Optional[Tuple[float, float, float]] = None,
    output_size: Optional[Tuple[int, int, int]] = None,
    interpolation: str = "linear",
) -> jnp.ndarray:
    r"""Rescale a 3D image by bilinear interpolation.

    Args:
        input: array of shape ``[..., x, y, z]``
        resize_rate: tuple of 3 floats
        output_size: tuple of 3 ints

    Returns:
        3D image of size output_size
    """
    nx, ny, nz = input.shape[-3:]

    if resize_rate is not None:
        assert output_size is None

        if isinstance(resize_rate, (float, int)):
            resize_rate = (resize_rate,) * 3

        output_size = (
            round(nx * resize_rate[0]),
            round(ny * resize_rate[1]),
            round(nz * resize_rate[2]),
        )

    assert isinstance(output_size, tuple)

    return _zoom(input, output_size, interpolation)
