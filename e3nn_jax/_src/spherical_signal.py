"""Implements spherical signal."""

from typing import Union, Tuple, List, Callable
from dataclasses import dataclass

import e3nn_jax as e3nn
import chex
import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal
import scipy.spatial


@dataclass(frozen=True)
class SphericalSignal:

    grid_values: chex.Array
    y: chex.Array
    alpha: chex.Array
    quadrature: str
    quadrature_weights: chex.Array
    p_val: int = 1
    p_arg: int = -1

    def __mul__(self, scalar: float) -> "SphericalSignal":
        """Multiply SphericalSignal by a scalar."""
        return SphericalSignal(
            self.grid_values * scalar, self.y, self.alpha, self.quadrature, self.quadrature_weights, self.p_val, self.p_arg
        )

    def __rmul__(self, scalar: float) -> "SphericalSignal":
        """Multiply SphericalSignal by a scalar."""
        return SphericalSignal(
            self.grid_values * scalar, self.y, self.alpha, self.quadrature, self.quadrature_weights, self.p_val, self.p_arg
        )

    def __add__(self, other: "SphericalSignal") -> "SphericalSignal":
        """Add to another SphericalSignal."""
        if self.grid_resolution() != other.grid_resolution():
            raise ValueError(
                "Grid resolutions for both signals must be identical. "
                "Use .resample() to change one of the grid resolutions."
            )
        if (self.p_val, self.p_arg) != (other.p_val, other.p_arg):
            raise ValueError("Parity for both signals must be identical.")

        return SphericalSignal(
            self.grid_values + other.grid_values,
            self.y,
            self.alpha,
            self.quadrature,
            self.quadrature_weights,
            self.p_val,
            self.p_arg,
        )

    def __sub__(self, other: "SphericalSignal") -> "SphericalSignal":
        """Subtract another SphericalSignal."""
        if self.grid_resolution() != other.grid_resolution():
            raise ValueError(
                "Grid resolutions for both signals must be identical. "
                "Use .resample() to change one of the grid resolutions."
            )
        if (self.p_val, self.p_arg) != (other.p_val, other.p_arg):
            raise ValueError("Parity for both signals must be identical.")

        return SphericalSignal(
            self.grid_values - other.grid_values,
            self.y,
            self.alpha,
            self.quadrature,
            self.quadrature_weights,
            self.p_val,
            self.p_arg,
        )

    def grid_resolution(self) -> Tuple[int, int]:
        """Grid resolution for (beta, alpha)."""
        return self.grid_values.shape[-2:]

    def resample(self, grid_resolution: Tuple[int, int], lmax: int) -> "SphericalSignal":
        """Resamples a signal via the spherical harmonic coefficients."""
        coeffs = self._coeffs(lmax)
        resampled_grid_values = e3nn.to_s2grid(coeffs, *grid_resolution, quadrature=self.quadrature)
        y, alpha, quadrature_weights = e3nn.s2grid(*grid_resolution, quadrature=self.quadrature)
        return SphericalSignal(resampled_grid_values, y, alpha, self.quadrature, quadrature_weights, self.p_val, self.p_arg)

    def _coeffs(self, lmax: int) -> e3nn.IrrepsArray:
        """Returns the coefficients for the spherical harmonics upto l = lmax."""
        irreps = SphericalSignal.irreps_for_spherical_signal(lmax, p_val=self.p_val, p_arg=self.p_arg)
        return e3nn.from_s2grid(self.grid_values, irreps, quadrature=self.quadrature)

    def _transform_by(self, transform_type: str, transform_kwargs: Tuple[Union[float, int], ...], lmax: int):
        """A wrapper for different transform_by functions."""
        coeffs = self._coeffs(lmax)
        transforms = {
            "angles": coeffs.transform_by_angles,
            "matrix": coeffs.transform_by_matrix,
            "axis_angle": coeffs.transform_by_axis_angle,
            "quaternion": coeffs.transform_by_quaternion,
        }
        transformed_coeffs = transforms[transform_type](**transform_kwargs)
        transformed_grid_values = e3nn.to_s2grid(transformed_coeffs, *self.grid_resolution(), quadrature=self.quadrature)
        return SphericalSignal(
            transformed_grid_values, self.y, self.alpha, self.quadrature, self.quadrature_weights, self.p_val, self.p_arg
        )

    def transform_by_angles(self, lmax: int, **kwargs) -> "SphericalSignal":
        """Rotate the signal by the given Euler angles."""
        return self._transform_by("angles", transform_kwargs=kwargs, lmax=lmax)

    def transform_by_matrix(self, R: chex.Array, lmax: int, **kwargs) -> "SphericalSignal":
        """Rotate the signal by the given rotation matrix."""
        return self._transform_by("matrix", transform_kwargs=kwargs, lmax=lmax)

    def transform_by_axis_angle(self, lmax: int, **kwargs) -> "SphericalSignal":
        """Rotate the signal by the given angle around an axis."""
        return self._transform_by("axis_angle", transform_kwargs=kwargs, lmax=lmax)

    def transform_by_quaternion(self, lmax: int, **kwargs) -> "SphericalSignal":
        """Rotate the signal by the given quaternion."""
        return self._transform_by("quaternion", transform_kwargs=kwargs, lmax=lmax)

    def apply(self, func: Callable[[chex.Array], chex.Array]):
        """Applies a function pointwise on the grid."""
        return SphericalSignal(func(self.grid_values), self.y, self.alpha, self.quadrature, self.quadrature_weights)

    @staticmethod
    def irreps_for_spherical_signal(lmax: int, p_val: int, p_arg: int) -> e3nn.Irreps:
        """Returns all Irreps upto l = lmax and of the required parity."""
        return e3nn.Irreps([(1, (l, p_val * p_arg**l)) for l in range(lmax + 1)])

    @classmethod
    def from_irreps_array(
        cls, coeffs: e3nn.IrrepsArray, res_beta: int = 100, res_alpha: int = 99, quadrature: str = "gausslegendre"
    ) -> "SphericalSignal":
        """Creates a SphericalSignal from an IrrepsArray and a given grid resolution."""
        grid_values = e3nn.to_s2grid(coeffs, res_beta=res_beta, res_alpha=res_alpha, quadrature=quadrature)
        s2grid = e3nn.s2grid(res_alpha=res_alpha, res_beta=res_beta, quadrature=quadrature)
        y, alpha, quadrature_weights = jax.tree_map(jnp.asarray, s2grid)
        return cls(grid_values, y, alpha, quadrature, quadrature_weights)

    @staticmethod
    def _find_peaks_2d(x: np.ndarray) -> List[Tuple[int, int]]:
        """Helper for finding peaks in a 2D signal."""
        iii = []
        for i in range(x.shape[0]):
            jj, _ = scipy.signal.find_peaks(x[i, :])
            iii += [(i, j) for j in jj]

        jjj = []
        for j in range(x.shape[1]):
            ii, _ = scipy.signal.find_peaks(x[:, j])
            jjj += [(i, j) for i in ii]

        return list(set(iii).intersection(set(jjj)))

    def find_peaks(self, lmax: int):
        r"""Locate peaks on the signal on the sphere."""
        grid_resolution = self.grid_resolution()
        x1, f1 = e3nn.s2grid_vectors(self.y, self.alpha), self.grid_values

        # Rotate signal.
        abc = np.array([jnp.pi / 2, jnp.pi / 2, jnp.pi / 2])
        R = e3nn.angles_to_matrix(*abc)
        r_signal = self.transform_by_angles(*abc, lmax=lmax)
        x2, f2 = np.einsum("ij,baj->bai", R.T, x1), r_signal.grid_values

        ij = self._find_peaks_2d(f1)
        x1p = np.stack([x1[i, j] for i, j in ij])
        f1p = np.stack([f1[i, j] for i, j in ij])

        ij = self._find_peaks_2d(f2)
        x2p = np.stack([x2[i, j] for i, j in ij])
        f2p = np.stack([f2[i, j] for i, j in ij])

        # Union of the results
        mask = scipy.spatial.distance.cdist(x1p, x2p) < 2 * jnp.pi / max(*grid_resolution)
        x = np.concatenate([x1p[mask.sum(axis=1) == 0], x2p])
        f = np.concatenate([f1p[mask.sum(axis=1) == 0], f2p])

        return x, f

    def plotly_surface(self, scale_radius_by_amplitude: bool = True):
        y, alpha, _ = e3nn.s2grid(*self.grid_resolution(), quadrature=self.quadrature)
        r, f = e3nn.pad_to_plot_on_s2grid(y, alpha, self.grid_values, scale_radius_by_amplitude=scale_radius_by_amplitude)
        return dict(
            x=r[:, :, 0],
            y=r[:, :, 1],
            z=r[:, :, 2],
            surfacecolor=f,
        )


def sum_of_diracs(positions: chex.Array, values: chex.Array, lmax: int, p_val: int, p_arg: int) -> e3nn.IrrepsArray:
    r"""Sum of (almost-)Dirac deltas

    .. math::

        f(x) = \sum_i v_i \delta^L(\vec r_i)

    where :math:`\delta^L` is the approximation of a Dirac delta.
    """
    values = values[..., None]
    positions, _ = jnp.broadcast_arrays(positions, values)
    irreps = SphericalSignal.irreps_for_spherical_signal(lmax, p_val, p_arg)
    y = e3nn.spherical_harmonics(irreps, positions, normalize=True, normalization="integral")  # [..., N, dim]
    return e3nn.sum(4 * jnp.pi / (lmax + 1) ** 2 * (y * values), axis=-2)
