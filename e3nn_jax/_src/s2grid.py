import math
from typing import Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal
import scipy.spatial

import e3nn_jax as e3nn

from .activation import parity_function
from .spherical_harmonics import _sh_alpha, _sh_beta


class SphericalSignal:
    r"""Representation of a signal on the sphere.

    Args:
        grid_values: values of the signal on a grid, shape ``(res_beta, res_alpha)``
        quadrature: quadrature used to create the grid, either ``"soft"`` or ``"gausslegendre"``
        p_val: parity of the signal, either ``+1`` or ``-1``
        p_arg: parity of the argument of the signal, either ``+1`` or ``-1``

    Examples:

        .. jupyter-execute::
            :hide-code:

            import e3nn_jax as e3nn
            import jax
            import jax.numpy as jnp
            jnp.set_printoptions(precision=3, suppress=True)

        Create a null signal:

        .. jupyter-execute::

            e3nn.SphericalSignal.zeros(50, 49, quadrature="soft")

        Create a signal from a spherical harmonic expansion:

        .. jupyter-execute::

            coeffs = e3nn.IrrepsArray("0e + 1o", jnp.array([1.0, 0.0, 2.0, 0.0]))
            signal = e3nn.to_s2grid(coeffs, 50, 49, quadrature="soft")
            signal

        Apply a function to the signal:

        .. jupyter-execute::

            signal = signal.apply(jnp.exp)
            signal

        Convert the signal back to a spherical harmonic expansion:

        .. jupyter-execute::

            irreps = e3nn.s2_irreps(4)
            coeffs = e3nn.from_s2grid(signal, irreps)
            coeffs["4e"]

        Resample the signal to a different grid resolution:

        .. jupyter-execute::

            signal = signal.resample(100, 99, lmax=5)
            signal

        Compute the integral of the signal:

        .. jupyter-execute::

            signal.integrate()

        Rotate the signal (we need to determine ``lmax`` because the rotation is done in the Fourier domain):

        .. jupyter-execute::

            signal = signal.transform_by_angles(jnp.pi / 2, jnp.pi / 3, 0.0, lmax=5)

        Sample a point on the sphere, using the signal as a density function:

        .. jupyter-execute::

            indices = signal.sample(jax.random.PRNGKey(0))
            signal.grid_vectors[indices], signal.grid_values[indices]

        Plot the signal:

        .. jupyter-execute::

            import plotly.graph_objects as go
            go.Figure([go.Surface(signal.plotly_surface())])
    """
    grid_values: jnp.ndarray
    quadrature: str
    p_val: int
    p_arg: int

    def __init__(
        self,
        grid_values: jnp.ndarray,
        quadrature: str,
        *,
        p_val: int = 1,
        p_arg: int = -1,
        _perform_checks: bool = True,
    ) -> None:
        if _perform_checks:
            if len(grid_values.shape) < 2:
                raise ValueError(
                    f"Grid values should have atleast 2 axes. Got grid_values of shape {grid_values.shape}."
                )

            if quadrature not in ["soft", "gausslegendre"]:
                raise ValueError(
                    f"Invalid quadrature for SphericalSignal: {quadrature}"
                )

            if p_val not in (-1, 1):
                raise ValueError(
                    f"Parity p_val must be either +1 or -1. Received: {p_val}"
                )

            if p_arg not in (-1, 1):
                raise ValueError(
                    f"Parity p_arg must be either +1 or -1. Received: {p_arg}"
                )

        self.grid_values = grid_values
        self.quadrature = quadrature
        self.p_val = p_val
        self.p_arg = p_arg

    @staticmethod
    def zeros(
        res_beta: int,
        res_alpha: int,
        quadrature: str,
        *,
        p_val: int = 1,
        p_arg: int = -1,
        dtype: jnp.dtype = jnp.float32,
    ) -> "SphericalSignal":
        """Create a null signal on a grid."""
        return SphericalSignal(
            jnp.zeros((res_beta, res_alpha), dtype),
            quadrature,
            p_val=p_val,
            p_arg=p_arg,
        )

    def __repr__(self) -> str:
        if self.ndim >= 2:
            return (
                "SphericalSignal("
                f"shape={self.shape}, "
                f"res_beta={self.res_beta}, res_alpha={self.res_alpha}, "
                f"quadrature={self.quadrature}, p_val={self.p_val}, p_arg={self.p_arg})\n"
                f"{self.grid_values}"
            )
        else:
            return f"SphericalSignal({self.grid_values})"

    def __mul__(self, scalar: Union[float, "SphericalSignal"]) -> "SphericalSignal":
        """Multiply SphericalSignal by a scalar."""
        if isinstance(scalar, SphericalSignal):
            other = scalar
            if self.quadrature != other.quadrature:
                raise ValueError(
                    "Multiplication of SphericalSignals with different quadrature is not supported."
                )
            if self.grid_resolution != other.grid_resolution:
                raise ValueError(
                    "Multiplication of SphericalSignals with different grid resolution is not supported."
                )
            if self.p_arg != other.p_arg:
                raise ValueError(
                    "Multiplication of SphericalSignals with different p_arg is not equivariant."
                )

            return SphericalSignal(
                self.grid_values * other.grid_values,
                self.quadrature,
                p_val=self.p_val * other.p_val,
                p_arg=self.p_arg,
            )

        if isinstance(scalar, e3nn.IrrepsArray):
            if scalar.irreps != e3nn.Irreps("0e"):
                raise ValueError("Scalar must be a 0e IrrepsArray.")
            scalar = scalar.array[..., 0]

        scalar = jnp.asarray(scalar)[..., None, None]
        return SphericalSignal(
            self.grid_values * scalar,
            self.quadrature,
            p_val=self.p_val,
            p_arg=self.p_arg,
        )

    def __rmul__(self, scalar: float) -> "SphericalSignal":
        """Multiply SphericalSignal by a scalar."""
        return self * scalar

    def __truediv__(self, scalar: float) -> "SphericalSignal":
        """Divide SphericalSignal by a scalar."""
        return self * (1 / scalar)

    def __add__(self, other: "SphericalSignal") -> "SphericalSignal":
        """Add to another SphericalSignal."""
        if self.grid_resolution != other.grid_resolution:
            raise ValueError(
                "Grid resolutions for both signals must be identical. "
                "Use .resample() to change one of the grid resolutions."
            )
        if (self.p_val, self.p_arg) != (other.p_val, other.p_arg):
            raise ValueError("Parity for both signals must be identical.")
        if self.quadrature != other.quadrature:
            raise ValueError("Quadrature for both signals must be identical.")

        return SphericalSignal(
            self.grid_values + other.grid_values,
            self.quadrature,
            p_val=self.p_val,
            p_arg=self.p_arg,
        )

    def __sub__(self, other: "SphericalSignal") -> "SphericalSignal":
        """Subtract another SphericalSignal."""
        return self + (-other)

    def __neg__(self) -> "SphericalSignal":
        """Negate SphericalSignal."""
        return SphericalSignal(
            -self.grid_values, self.quadrature, p_val=self.p_val, p_arg=self.p_arg
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the shape of this signal."""
        return self.grid_values.shape

    @property
    def dtype(self) -> jnp.dtype:
        """Returns the dtype of this signal."""
        return self.grid_values.dtype

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of this signal."""
        return self.grid_values.ndim

    @property
    def grid_y(self) -> jnp.ndarray:
        """Returns y-values on the grid for this signal."""
        y, _, _ = _s2grid(self.res_beta, self.res_alpha, self.quadrature)
        return y

    @property
    def grid_alpha(self) -> jnp.ndarray:
        """Returns alpha values on the grid for this signal."""
        _, alpha, _ = _s2grid(self.res_beta, self.res_alpha, self.quadrature)
        return alpha

    @property
    def grid_vectors(self) -> jnp.ndarray:
        """Returns the coordinates of the points on the sphere. Shape: ``(res_beta, res_alpha, 3)``."""
        y, alpha, _ = _s2grid(self.res_beta, self.res_alpha, self.quadrature)
        return _s2grid_vectors(y, alpha)

    @property
    def quadrature_weights(self) -> jnp.ndarray:
        """Returns quadrature weights along the y-coordinates."""
        _, _, qw = _s2grid(self.res_beta, self.res_alpha, self.quadrature)
        return qw

    @property
    def res_beta(self) -> int:
        """Grid resolution for beta."""
        return self.grid_values.shape[-2]

    @property
    def res_alpha(self) -> int:
        """Grid resolution for alpha."""
        return self.grid_values.shape[-1]

    @property
    def grid_resolution(self) -> Tuple[int, int]:
        """Grid resolution for (beta, alpha)."""
        return (self.res_beta, self.res_alpha)

    def resample(
        self, res_beta: int, res_alpha: int, lmax: int, quadrature: Optional[str] = None
    ) -> "SphericalSignal":
        """Resamples a signal via the spherical harmonic coefficients.

        Args:
            res_beta: New resolution for beta.
            res_alpha: New resolution for alpha.
            lmax: Maximum l for the spherical harmonics.
            quadrature: Quadrature to use. Defaults to reusing the current quadrature.

        Returns:
            A new SphericalSignal with the new resolution.
        """
        if quadrature is None:
            quadrature = self.quadrature
        coeffs = e3nn.from_s2grid(self, s2_irreps(lmax, self.p_val, self.p_arg))
        return e3nn.to_s2grid(
            coeffs,
            res_beta,
            res_alpha,
            quadrature=quadrature,
            p_val=self.p_val,
            p_arg=self.p_arg,
        )

    def _transform_by(
        self,
        transform_type: str,
        transform_kwargs: Tuple[Union[float, int], ...],
        lmax: int,
    ) -> "SphericalSignal":
        """A wrapper for different transform_by functions."""
        coeffs = e3nn.from_s2grid(self, s2_irreps(lmax, self.p_val, self.p_arg))
        transforms = {
            "angles": coeffs.transform_by_angles,
            "matrix": coeffs.transform_by_matrix,
            "axis_angle": coeffs.transform_by_axis_angle,
            "quaternion": coeffs.transform_by_quaternion,
        }
        transformed_coeffs = transforms[transform_type](**transform_kwargs)
        return e3nn.to_s2grid(
            transformed_coeffs,
            *self.grid_resolution,
            quadrature=self.quadrature,
            p_val=self.p_val,
            p_arg=self.p_arg,
        )

    def transform_by_angles(
        self, alpha: float, beta: float, gamma: float, lmax: int
    ) -> "SphericalSignal":
        """Rotate the signal by the given Euler angles."""
        return self._transform_by(
            "angles",
            transform_kwargs=dict(alpha=alpha, beta=beta, gamma=gamma),
            lmax=lmax,
        )

    def transform_by_matrix(self, R: jnp.ndarray, lmax: int) -> "SphericalSignal":
        """Rotate the signal by the given rotation matrix."""
        return self._transform_by("matrix", transform_kwargs=dict(R=R), lmax=lmax)

    def transform_by_axis_angle(
        self, axis: jnp.ndarray, angle: float, lmax: int
    ) -> "SphericalSignal":
        """Rotate the signal by the given angle around an axis."""
        return self._transform_by(
            "axis_angle", transform_kwargs=dict(axis=axis, angle=angle), lmax=lmax
        )

    def transform_by_quaternion(self, q: jnp.ndarray, lmax: int) -> "SphericalSignal":
        """Rotate the signal by the given quaternion."""
        return self._transform_by("quaternion", transform_kwargs=dict(q=q), lmax=lmax)

    def apply(self, func: Callable[[jnp.ndarray], jnp.ndarray]):
        """Applies a function pointwise on the grid."""
        new_p_val = parity_function(func) if self.p_val == -1 else self.p_val
        if new_p_val == 0:
            raise ValueError(
                "Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd."
            )
        return SphericalSignal(
            func(self.grid_values), self.quadrature, p_val=new_p_val, p_arg=self.p_arg
        )

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

    def find_peaks(self, lmax: int) -> Tuple[np.ndarray, np.ndarray]:
        r"""Locate peaks on the signal on the sphere.

        Currently cannot be wrapped with jax.jit().
        """
        # TODO: Still has the bug `ValueError: buffer source array is read-only`
        grid_resolution = self.grid_resolution
        x1, f1 = self.grid_vectors, self.grid_values
        x1, f1 = jax.tree_map(lambda arr: np.asarray(arr.copy()), (x1, f1))

        # Rotate signal.
        abc = (np.pi / 2, np.pi / 2, np.pi / 2)
        rotated_signal = self.transform_by_angles(*abc, lmax=lmax)
        rotated_vectors = e3nn.IrrepsArray("1o", x1).transform_by_angles(*abc).array
        x2, f2 = rotated_vectors, rotated_signal.grid_values
        x2, f2 = jax.tree_map(lambda arr: np.asarray(arr.copy()), (x2, f2))

        ij = self._find_peaks_2d(f1)
        x1p = np.stack([x1[i, j] for i, j in ij])
        f1p = np.stack([f1[i, j] for i, j in ij])

        ij = self._find_peaks_2d(f2)
        x2p = np.stack([x2[i, j] for i, j in ij])
        f2p = np.stack([f2[i, j] for i, j in ij])

        # Union of the results
        mask = scipy.spatial.distance.cdist(x1p, x2p) < 2 * np.pi / max(
            *grid_resolution
        )
        x = np.concatenate([x1p[mask.sum(axis=1) == 0], x2p])
        f = np.concatenate([f1p[mask.sum(axis=1) == 0], f2p])

        return x, f

    def pad_to_plot(
        self,
        *,
        translation: Optional[jnp.ndarray] = None,
        radius: float = 1.0,
        scale_radius_by_amplitude: bool = False,
        normalize_radius_by_max_amplitude: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        r"""Postprocess the borders of a given signal to allow to plot with plotly.

        Args:
            translation (optional): translation vector
            radius (float): radius of the sphere
            scale_radius_by_amplitude (bool): to rescale the output vectors with the amplitude of the signal
            normalize_radius_by_max_amplitude (bool): when scale_radius_by_amplitude is True,
                rescales the surface so that the maximum amplitude is equal to the radius

        Returns:
            r (`jax.numpy.ndarray`): vectors on the sphere, shape ``(res_beta + 2, res_alpha + 1, 3)``
            f (`jax.numpy.ndarray`): padded signal, shape ``(res_beta + 2, res_alpha + 1)``
        """
        f, y, alpha = self.grid_values, self.grid_y, self.grid_alpha
        assert f.ndim == 2 and f.shape == (
            len(y),
            len(alpha),
        ), f"Invalid shape: grid_values.shape={f.shape}, expected ({len(y)}, {len(alpha)})"

        # y: [-1, 1]
        one = jnp.ones_like(y, shape=(1,))
        ones = jnp.ones_like(f, shape=(1, len(alpha)))
        y = jnp.concatenate([-one, y, one])  # [res_beta + 2]
        f = jnp.concatenate(
            [jnp.mean(f[0]) * ones, f, jnp.mean(f[-1]) * ones], axis=0
        )  # [res_beta + 2, res_alpha]

        # alpha: [0, 2pi]
        alpha = jnp.concatenate([alpha, alpha[:1]])  # [res_alpha + 1]
        f = jnp.concatenate([f, f[:, :1]], axis=1)  # [res_beta + 2, res_alpha + 1]

        # Coordinate vectors of the grid.
        r = _s2grid_vectors(y, alpha)  # [res_beta + 2, res_alpha + 1, 3]

        if scale_radius_by_amplitude:
            nr = jnp.abs(f)[:, :, None]

            if normalize_radius_by_max_amplitude:
                nr = nr / jnp.max(nr)

            r = r * nr

        r = r * radius

        if translation is not None:
            r = r + translation

        return r, f

    def plotly_surface(
        self,
        translation: Optional[jnp.ndarray] = None,
        radius: float = 1.0,
        scale_radius_by_amplitude: bool = False,
        normalize_radius_by_max_amplitude: bool = False,
    ):
        """Returns a dictionary that can be plotted with plotly.

        Args:
            translation (optional): translation vector
            radius (float): radius of the sphere
            scale_radius_by_amplitude (bool): to rescale the output vectors with the amplitude of the signal
            normalize_radius_by_max_amplitude (bool): when scale_radius_by_amplitude is True,
                rescales the surface so that the maximum amplitude is equal to the radius

        Returns:
            dict: dictionary that can be plotted with plotly

        Examples:

        .. jupyter-execute::

            import jax.numpy as jnp
            import e3nn_jax as e3nn
            coeffs = e3nn.normal(e3nn.s2_irreps(5), jax.random.PRNGKey(0))
            signal = e3nn.to_s2grid(coeffs, 70, 141, quadrature="gausslegendre")

            import plotly.graph_objects as go
            go.Figure([go.Surface(signal.plotly_surface())])

        One can also scale the radius of the sphere by the amplitude of the signal:

        .. jupyter-execute::

            go.Figure([go.Surface(signal.plotly_surface(scale_radius_by_amplitude=True))])

        """
        r, f = self.pad_to_plot(
            translation=translation,
            radius=radius,
            scale_radius_by_amplitude=scale_radius_by_amplitude,
            normalize_radius_by_max_amplitude=normalize_radius_by_max_amplitude,
        )
        return dict(
            x=r[:, :, 0],
            y=r[:, :, 1],
            z=r[:, :, 2],
            surfacecolor=f,
        )

    def integrate(self) -> e3nn.IrrepsArray:
        """Integrate the signal on the sphere.

        The integral of a constant signal of value 1 is 4pi.

        Returns:
            `IrrepsArray`: integral of the signal
        """
        values = self.quadrature_weights[..., None] * self.grid_values
        values = jnp.sum(values, axis=-2)
        values = jnp.mean(values, axis=-1, keepdims=True) * 4 * jnp.pi
        # Handle parity of integral.
        integral_irreps = {1: "0e", -1: "0o"}[self.p_val]
        return e3nn.IrrepsArray(integral_irreps, values)

    def sample(self, key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample a point on the sphere using the signal as a probability distribution.

        The probability distribution does not need to be normalized.

        Args:
            key (`jax.numpy.ndarray`): random key

        Returns:
            (tuple): tuple containing:

                beta_index (`jax.numpy.ndarray`): index of the sampled beta
                alpha_index (`jax.numpy.ndarray`): index of the sampled alpha

        Examples:

        .. jupyter-execute::
            :hide-code:

            import jax
            import jax.numpy as jnp
            import e3nn_jax as e3nn


        .. jupyter-execute::

            coeffs = e3nn.IrrepsArray("0e + 1o", jnp.array([1.0, 2.0, 0.0, 0.0]))
            signal = e3nn.to_s2grid(coeffs, 50, 69, quadrature="gausslegendre")
            signal = signal.apply(jnp.exp)

            beta_index, alpha_index = signal.sample(jax.random.PRNGKey(0))
            print(beta_index, alpha_index)
            print(signal.grid_vectors[beta_index, alpha_index])
        """

        def f(k, p_ya):  # single signal only
            assert k.shape == (2,)
            assert p_ya.shape == (self.res_beta, self.res_alpha)
            k1, k2 = jax.random.split(k)
            p_y = self.quadrature_weights * jnp.sum(p_ya, axis=1)  # [y]
            y_index = jax.random.choice(k1, jnp.arange(self.res_beta), p=p_y)  # []
            alpha_index = jax.random.choice(
                k2, jnp.arange(self.res_alpha), p=p_ya[y_index]
            )  # []
            return y_index, alpha_index

        vf = f
        for _ in range(self.ndim - 2):
            vf = jax.vmap(vf)

        keys = jax.random.split(key, math.prod(self.shape[:-2])).reshape(
            self.shape[:-2] + key.shape
        )
        return vf(keys, self.grid_values)

    def __getitem__(self, index) -> "SphericalSignal":
        grid_values = self.grid_values[index]

        if grid_values.ndim < 2:
            raise ValueError(
                "This indexing does not produce something that can be interpreted as a signal on the sphere. "
                "Consider using `SphericalSignal.grid_values` instead."
            )

        return SphericalSignal(
            grid_values=grid_values,
            quadrature=self.quadrature,
            p_val=self.p_val,
            p_arg=self.p_arg,
            _perform_checks=False,
        )


jax.tree_util.register_pytree_node(
    SphericalSignal,
    lambda x: ((x.grid_values,), (x.quadrature, x.p_val, x.p_arg)),
    lambda aux, grid_values: SphericalSignal(
        grid_values=grid_values[0],
        quadrature=aux[0],
        p_val=aux[1],
        p_arg=aux[2],
        _perform_checks=False,
    ),
)


def s2_dirac(
    position: Union[jnp.ndarray, e3nn.IrrepsArray], lmax: int, *, p_val: int, p_arg: int
) -> e3nn.IrrepsArray:
    r"""Spherical harmonics expansion of a Dirac delta on the sphere.

    The integral of the Dirac delta is 1.

    Args:
        position (`jax.numpy.ndarray` or `IrrepsArray`): position of the delta, shape ``(3,)``.
            It will be normalized to have a norm of 1.

        lmax (int): maximum degree of the spherical harmonics expansion
        p_val (int): parity of the value of the signal on the sphere (1 or -1)
        p_arg (int): parity of the argument of the signal on the sphere (1 or -1)

    Returns:
        `IrrepsArray`: Spherical harmonics coefficients

    Examples:

    .. jupyter-execute::
        :hide-code:

        import jax.numpy as jnp
        import e3nn_jax as e3nn
        import plotly.graph_objects as go

    .. jupyter-execute::

        position = jnp.array([0.0, 0.0, 1.0])

        coeffs_3 = e3nn.s2_dirac(position, 3, p_val=1, p_arg=-1)
        coeffs_6 = e3nn.s2_dirac(position, 6, p_val=1, p_arg=-1)
        coeffs_9 = e3nn.s2_dirac(position, 9, p_val=1, p_arg=-1)

    .. jupyter-execute::
        :hide-code:

        signal_3 = e3nn.to_s2grid(coeffs_3, 50, 69, quadrature="gausslegendre")
        signal_6 = e3nn.to_s2grid(coeffs_6, 50, 69, quadrature="gausslegendre")
        signal_9 = e3nn.to_s2grid(coeffs_9, 50, 69, quadrature="gausslegendre")

        axis = dict(
            # showbackground=False,
            # showgrid=False,
            # showline=False,
            showticklabels=False,
            # ticks="",
            title="",
        )
        go.Figure(
            [
                go.Surface(dict(**signal_3.plotly_surface(jnp.array([-2.1, 0, 0])), showscale=False)),
                go.Surface(dict(**signal_6.plotly_surface(), showscale=False)),
                go.Surface(dict(**signal_9.plotly_surface(jnp.array([2.1, 0, 0])), showscale=False)),
            ],
            layout=go.Layout(
                scene=dict(
                    xaxis=dict(range=[-3.1, 3.1], **axis),
                    yaxis=dict(range=[-1, 1], **axis),
                    zaxis=dict(range=[-1, 1], **axis),
                    camera=dict(
                        eye=dict(x=0.0, y=1.0, z=3.0),
                        up=dict(x=0.0, y=1.0, z=0.0),
                    ),
                    aspectratio=dict(x=3.1, y=1, z=1),
                ),
            ),
        )

    Note:

        To compute a sum of weighted Dirac deltas, use:

        .. jupyter-execute::

            positions = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
            weights = jnp.array([1, 1, -1, -1.0])

            e3nn.sum(e3nn.s2_dirac(positions, 4, p_val=1, p_arg=-1) * weights[:, None], axis=0)
    """
    irreps = s2_irreps(lmax, p_val, p_arg)
    coeffs = e3nn.spherical_harmonics(
        irreps, position, normalize=True, normalization="integral"
    )  # [dim]
    return coeffs / jnp.sqrt(4 * jnp.pi)


def s2_irreps(lmax: int, p_val: int = 1, p_arg: int = -1) -> e3nn.Irreps:
    r"""The Irreps of coefficients of a spherical harmonics expansion.

    .. math::

        f(\vec x) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_l^m Y_{l,m}(\vec x)

    When the inversion operator is applied to the signal, the new function :math:`I f` is given by

    .. math::

        [I f](\vec x) = p_{\text{val}} f(p_{\text{arg}} \vec x)

    Args:
        lmax (int): maximum degree of the expansion
        p_val (int): parity of the value of the signal on the sphere (1 or -1)
        p_arg (int): parity of the argument of the signal on the sphere (1 or -1)
    """
    return e3nn.Irreps([(1, (l, p_val * p_arg**l)) for l in range(lmax + 1)])


def from_s2grid(
    x: SphericalSignal,
    irreps: e3nn.Irreps,
    *,
    normalization: str = "integral",
    lmax_in: Optional[int] = None,
    fft: bool = True,
) -> e3nn.IrrepsArray:
    r"""Transform signal on the sphere into spherical harmonics coefficients.

    The output has degree :math:`l` between 0 and lmax, and parity :math:`p = p_{val}p_{arg}^l`

    The inverse transformation of :func:`to_s2grid`

    Args:
        x (`SphericalSignal`): signal on the sphere of shape ``(..., y/beta, alpha)``
        irreps (`Irreps`): irreps of the coefficients
        normalization ({'norm', 'component', 'integral'}): normalization of the spherical harmonics basis
        lmax_in (int, optional): maximum degree of the input signal, only used for normalization purposes
        fft (bool): True if we use FFT, False if we use the naive implementation

    Returns:
        `IrrepsArray`: coefficient array of shape ``(..., (lmax+1)^2)``
    """
    res_beta, res_alpha = x.grid_resolution

    irreps = e3nn.Irreps(irreps)

    if not all(mul == 1 for mul, _ in irreps.regroup()):
        raise ValueError("multiplicities should be ones")

    _check_parities(irreps, x.p_val, x.p_arg)

    lmax = max(irreps.ls)

    if lmax_in is None:
        lmax_in = lmax

    _, _, sh_y, sha, qw = _spherical_harmonics_s2grid(
        lmax, res_beta, res_alpha, quadrature=x.quadrature, dtype=x.dtype
    )
    # sh_y: (res_beta, (l+1)(l+2)/2)

    n = _normalization(lmax, normalization, x.dtype, "from_s2", lmax_in)

    # prepare beta integrand
    m_in = jnp.asarray(_expand_matrix(range(lmax + 1)), x.dtype)  # [l, m, j]
    m_out = jnp.asarray(_expand_matrix(irreps.ls), x.dtype)  # [l, m, i]
    sh_y = _rollout_sh(sh_y, lmax)
    sh_y = jnp.einsum("lmj,bj,lmi,l,b->mbi", m_in, sh_y, m_out, n, qw)  # [m, b, i]

    # integrate over alpha
    if fft:
        int_a = _rfft(x.grid_values, lmax) / res_alpha  # [..., res_beta, 2*l+1]
    else:
        int_a = (
            jnp.einsum("...ba,am->...bm", x.grid_values, sha) / res_alpha
        )  # [..., res_beta, 2*l+1]

    # integrate over beta
    int_b = jnp.einsum("mbi,...bm->...i", sh_y, int_a)  # [..., irreps]

    # convert to IrrepsArray
    return e3nn.IrrepsArray(irreps, int_b)


def to_s2grid(
    coeffs: e3nn.IrrepsArray,
    res_beta: int,
    res_alpha: int,
    *,
    quadrature: str,
    normalization: str = "integral",
    fft: bool = True,
    p_val: Optional[int] = None,
    p_arg: Optional[int] = None,
) -> SphericalSignal:
    r"""Sample a signal on the sphere given by the coefficient in the spherical harmonics basis.

    The inverse transformation of :func:`from_s2grid`

    Args:
        coeffs (`IrrepsArray`): coefficient array
        res_beta (int): number of points on the sphere in the :math:`\theta` direction
        res_alpha (int): number of points on the sphere in the :math:`\phi` direction
        normalization ({'norm', 'component', 'integral'}): normalization of the basis
        quadrature (str): "soft" or "gausslegendre"
        fft (bool): True if we use FFT, False if we use the naive implementation
        p_val (int, optional): parity of the value of the signal
        p_arg (int, optional): parity of the argument of the signal

    Returns:
        `SphericalSignal`: signal on the sphere of shape ``(..., y/beta, alpha)``

    Note:

        We use a rectangular grid for the :math:`\beta` and :math:`\alpha` angles.
        The grid is uniform in the :math:`\alpha` angle while for :math:`\beta`, two different quadratures are available:

        * The `soft <https://link.springer.com/article/10.1007/s00041-008-9013-5>`_
          quadrature is a uniform sampling of the beta angle.
        * The `gauss-legendre <https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature>`_
          quadrature is a quadrature rule that is exact for polynomials of degree ``2 res_beta - 1``.
          On the sphere it is exact only for polynomials of :math:`y`.

        .. jupyter-execute::
            :hide-code:

            import jax.numpy as jnp
            import e3nn_jax as e3nn
            import plotly.graph_objects as go

            soft = e3nn.SphericalSignal.zeros(10, 19, "soft")
            gauss = e3nn.SphericalSignal.zeros(10, 19, "gausslegendre")

            axis = dict(showticklabels=False, title="", range=[-1.1, 1.1])

            go.Figure(
                data=[
                    go.Scatter3d(
                        x=s.grid_vectors[:, :, 0].reshape(-1),
                        y=s.grid_vectors[:, :, 1].reshape(-1),
                        z=s.grid_vectors[:, :, 2].reshape(-1),
                        mode="markers",
                        marker=dict(
                            color=c,
                            size=100 * jnp.broadcast_to(s.quadrature_weights[:, None], s.grid_resolution).reshape(-1)
                        ),
                        name=s.quadrature,
                    )
                    for s, c in zip([soft, gauss], ["blue", "red"])
                ],
                layout=go.Layout(
                    scene=dict(
                        xaxis=axis,
                        yaxis=axis,
                        zaxis=axis,
                        camera=dict(
                            eye=dict(x=0.4, y=0.0, z=1.3),
                            up=dict(x=0.0, y=1.0, z=0.0),
                        ),
                    ),
                ),
            )
    """
    coeffs = coeffs.regroup()
    lmax = coeffs.irreps.ls[-1]

    if not all(mul == 1 for mul, _ in coeffs.irreps):
        raise ValueError(f"Multiplicities should be ones. Got {coeffs.irreps}.")

    if (p_val is not None) != (p_arg is not None):
        raise ValueError("p_val and p_arg should be both None or both not None.")

    p_val, p_arg = _check_parities(coeffs.irreps, p_val, p_arg)

    if p_val is None or p_arg is None:
        raise ValueError(
            f"p_val and p_arg cannot be determined from the irreps {coeffs.irreps}, please specify them."
        )

    _, _, sh_y, sha, _ = _spherical_harmonics_s2grid(
        lmax, res_beta, res_alpha, quadrature=quadrature, dtype=coeffs.dtype
    )

    n = _normalization(lmax, normalization, coeffs.dtype, "to_s2")

    m_in = jnp.asarray(_expand_matrix(range(lmax + 1)), coeffs.dtype)  # [l, m, j]
    m_out = jnp.asarray(_expand_matrix(coeffs.irreps.ls), coeffs.dtype)  # [l, m, i]
    # put beta component in summable form
    sh_y = _rollout_sh(sh_y, lmax)
    sh_y = jnp.einsum("lmj,bj,lmi,l->mbi", m_in, sh_y, m_out, n)  # [m, b, i]

    # multiply spherical harmonics by their coefficients
    signal_b = jnp.einsum("mbi,...i->...bm", sh_y, coeffs.array)  # [batch, beta, m]

    if fft:
        if res_alpha % 2 == 0:
            raise ValueError("res_alpha must be odd for fft")

        signal = _irfft(signal_b, res_alpha) * res_alpha  # [..., res_beta, res_alpha]
    else:
        signal = jnp.einsum(
            "...bm,am->...ba", signal_b, sha
        )  # [..., res_beta, res_alpha]

    return SphericalSignal(signal, quadrature=quadrature, p_val=p_val, p_arg=p_arg)


def to_s2point(
    coeffs: e3nn.IrrepsArray,
    point: e3nn.IrrepsArray,
    *,
    normalization: str = "integral",
) -> e3nn.IrrepsArray:
    """Evaluate a signal on the sphere given by the coefficient in the spherical harmonics basis.

    It computes the same thing as :func:`to_s2grid` but at a single point.

    Args:
        coeffs (`IrrepsArray`): coefficient array of shape ``(*shape1, irreps)``
        point (`jax.numpy.ndarray`): point on the sphere of shape ``(*shape2, 3)``
        normalization ({'norm', 'component', 'integral'}): normalization of the basis

    Returns:
        `IrrepsArray`: signal on the sphere of shape ``(*shape1, *shape2, irreps)``
    """
    coeffs = coeffs.regroup()

    if not all(mul == 1 for mul, _ in coeffs.irreps):
        raise ValueError(f"Multiplicities should be ones. Got {coeffs.irreps}.")

    if not isinstance(point, e3nn.IrrepsArray):
        raise TypeError(f"point should be an e3nn.IrrepsArray, got {type(point)}.")

    if point.irreps not in ["1e", "1o"]:
        raise ValueError(f"point should be of irreps '1e' or '1o', got {point.irreps}.")

    p_arg = point.irreps[0].ir.p
    p_val, _ = _check_parities(coeffs.irreps, None, p_arg)

    sh = e3nn.spherical_harmonics(
        coeffs.irreps.ls, point, True, "integral"
    )  # [*shape2, irreps]
    n = _normalization(sh.irreps.lmax, normalization, coeffs.dtype, "to_s2")[
        jnp.array(sh.irreps.ls)
    ]  # [num_irreps]
    sh = sh * n

    shape1 = coeffs.shape[:-1]
    coeffs = coeffs.reshape((-1, coeffs.shape[-1]))
    shape2 = point.shape[:-1]
    sh = sh.reshape((-1, sh.shape[-1]))

    irreps = {1: "0e", -1: "0o"}[p_val]
    return e3nn.IrrepsArray(
        irreps,
        jnp.einsum("ai,bi->ab", sh.array, coeffs.array).reshape(shape1 + shape2 + (1,)),
    )


def _s2grid_vectors(y: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    r"""Calculate the coordinates of the points on the sphere.

    Args:
        y: array with y values, shape ``(res_beta)``
        alpha: array with alpha values, shape ``(res_alpha)``

    Returns:
        r: array of vectors, shape ``(res_beta, res_alpha, 3)``
    """

    return jnp.stack(
        [
            jnp.sqrt(1.0 - y[:, None] ** 2) * jnp.sin(alpha),
            y[:, None] * jnp.ones_like(alpha),
            jnp.sqrt(1.0 - y[:, None] ** 2) * jnp.cos(alpha),
        ],
        axis=2,
    )


def _quadrature_weights_soft(b: int) -> np.ndarray:
    r"""function copied from ``lie_learn.spaces.S3``
    Compute quadrature weights for the grid used by Kostelec & Rockmore [1, 2].
    """
    assert (
        b % 2 == 0
    ), "res_beta needs to be even for soft quadrature weights to be computed properly"
    k = np.arange(b // 2)
    return np.array(
        [
            (
                (4.0 / b)
                * np.sin(np.pi * (2.0 * j + 1.0) / (2.0 * b))
                * (
                    (1.0 / (2 * k + 1))
                    * np.sin((2 * j + 1) * (2 * k + 1) * np.pi / (2.0 * b))
                ).sum()
            )
            for j in np.arange(b)
        ],
    )


def _s2grid(
    res_beta: int, res_alpha: int, quadrature: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Returns arrays describing the grid on the sphere.

    Args:
        res_beta (int): :math:`N`
        res_alpha (int): :math:`M`
        quadrature (str): "soft" or "gausslegendre"

    Returns:
        (tuple): tuple containing:
            y (`numpy.ndarray`): array of shape ``(res_beta)``
            alpha (`numpy.ndarray`): array of shape ``(res_alpha)``
            qw (`numpy.ndarray`): array of shape ``(res_beta)``, ``sum(qw) = 1``
    """

    if quadrature == "soft":
        i = np.arange(res_beta)
        betas = (i + 0.5) / res_beta * np.pi
        y = -np.cos(betas)  # minus sign is here to go from -1 to 1 in both quadratures

        qw = _quadrature_weights_soft(res_beta)
    elif quadrature == "gausslegendre":
        y, qw = np.polynomial.legendre.leggauss(res_beta)
    else:
        raise Exception("quadrature needs to be 'soft' or 'gausslegendre'")

    qw /= 2.0
    i = np.arange(res_alpha)
    alpha = i / res_alpha * 2 * np.pi
    return y, alpha, qw


def _spherical_harmonics_s2grid(
    lmax: int,
    res_beta: int,
    res_alpha: int,
    *,
    quadrature: str,
    dtype: np.dtype = np.float32,
):
    r"""spherical harmonics evaluated on the grid on the sphere
    .. math::
        f(x) = \sum_{l=0}^{l_{\mathit{max}}} F^l \cdot Y^l(x)
        f(\beta, \alpha) = \sum_{l=0}^{l_{\mathit{max}}} F^l \cdot S^l(\alpha) P^l(\cos(\beta))
    Args:
        lmax (int): :math:`l_{\mathit{max}}`
        res_beta (int): :math:`N`
        res_alpha (int): :math:`M`
        quadrature (str): "soft" or "gausslegendre"

    Returns:
        (tuple): tuple containing:
            y (`jax.numpy.ndarray`): array of shape ``(res_beta)``
            alphas (`jax.numpy.ndarray`): array of shape ``(res_alpha)``
            sh_y (`jax.numpy.ndarray`): array of shape ``(res_beta, (lmax + 1)(lmax + 2)/2)``
            sh_alpha (`jax.numpy.ndarray`): array of shape ``(res_alpha, 2 * lmax + 1)``
            qw (`jax.numpy.ndarray`): array of shape ``(res_beta)``
    """
    y, alphas, qw = _s2grid(res_beta, res_alpha, quadrature)
    y, alphas, qw = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype), (y, alphas, qw)
    )
    sh_alpha = _sh_alpha(lmax, alphas)  # [..., 2 * l + 1]
    sh_y = _sh_beta(lmax, y)  # [..., (lmax + 1) * (lmax + 2) // 2]
    return y, alphas, sh_y, sh_alpha, qw


def _check_parities(
    irreps: e3nn.Irreps, p_val: Optional[int] = None, p_arg: Optional[int] = None
) -> Tuple[int, int]:
    p_even = {ir.p for mul, ir in irreps if ir.l % 2 == 0}
    p_odd = {ir.p for mul, ir in irreps if ir.l % 2 == 1}
    if not (p_even in [{1}, {-1}, set()] and p_odd in [{1}, {-1}, set()]):
        raise ValueError(
            "irrep parities should be of the form (p_val * p_arg**l) for all l, where p_val and p_arg are ±1"
        )

    p_even = p_even.pop() if p_even else None
    p_odd = p_odd.pop() if p_odd else None

    if p_val is not None and p_arg is not None:
        if not (p_even in [p_val, None] and p_odd in [p_val * p_arg, None]):
            raise ValueError(
                f"irrep ({irreps}) parities are not compatible with the given p_val ({p_val}) and p_arg ({p_arg})."
            )
        return p_val, p_arg

    if p_val is not None:
        if p_even is None:
            p_even = p_val
        if p_even != p_val:
            raise ValueError(
                f"irrep ({irreps}) parities are not compatible with the given p_val ({p_val})."
            )

    if p_arg is not None:
        if p_odd is None and p_even is not None:
            p_odd = p_even * p_arg
        elif p_odd is not None and p_even is None:
            p_even = p_odd * p_arg
        elif p_odd is not None and p_even is not None:
            if p_odd != p_even * p_arg:
                raise ValueError(
                    f"irrep ({irreps}) parities are not compatible with the given p_arg ({p_arg})."
                )

    if p_even is not None and p_odd is not None:
        return p_even, p_even * p_odd

    return p_even, None


def _normalization(
    lmax: int, normalization: str, dtype, direction: str, lmax_in: Optional[int] = None
) -> jnp.ndarray:
    """Handles normalization of different components of IrrepsArrays."""
    assert direction in ["to_s2", "from_s2"]

    if normalization == "component":
        # normalize such that all l has the same variance on the sphere
        # given that all component has mean 0 and variance 1
        if direction == "to_s2":
            return (
                jnp.sqrt(4 * jnp.pi)
                * jnp.asarray([1 / jnp.sqrt(2 * l + 1) for l in range(lmax + 1)], dtype)
                / jnp.sqrt(lmax + 1)
            )
        else:
            return (
                jnp.sqrt(4 * jnp.pi)
                * jnp.asarray([jnp.sqrt(2 * l + 1) for l in range(lmax + 1)], dtype)
                * jnp.sqrt(lmax_in + 1)
            )
    if normalization == "norm":
        # normalize such that all l has the same variance on the sphere
        # given that all component has mean 0 and variance 1/(2L+1)
        if direction == "to_s2":
            return jnp.sqrt(4 * jnp.pi) * jnp.ones(lmax + 1, dtype) / jnp.sqrt(lmax + 1)
        else:
            return (
                jnp.sqrt(4 * jnp.pi) * jnp.ones(lmax + 1, dtype) * jnp.sqrt(lmax_in + 1)
            )
    if normalization == "integral":
        # normalize such that the coefficient L=0 is equal to 4 pi the integral of the function
        # for "integral" normalization, the direction does not matter.
        return jnp.ones(lmax + 1, dtype) * jnp.sqrt(4 * jnp.pi)

    raise Exception("normalization needs to be 'norm', 'component' or 'integral'")


def _rfft(x: jnp.ndarray, l: int) -> jnp.ndarray:
    r"""Real fourier transform
    Args:
        x (`jax.numpy.ndarray`): input array of shape ``(..., res_beta, res_alpha)``
        l (int): value of `l` for which the transform is being run
    Returns:
        `jax.numpy.ndarray`: transformed values - array of shape ``(..., res_beta, 2*l+1)``
    """
    x_reshaped = x.reshape((-1, x.shape[-1]))
    x_transformed_c = jnp.fft.rfft(x_reshaped)  # (..., 2*l+1)
    x_transformed = jnp.concatenate(
        [
            jnp.flip(jnp.imag(x_transformed_c[..., 1 : l + 1]), -1) * -jnp.sqrt(2),
            jnp.real(x_transformed_c[..., :1]),
            jnp.real(x_transformed_c[..., 1 : l + 1]) * jnp.sqrt(2),
        ],
        axis=-1,
    )
    return x_transformed.reshape((*x.shape[:-1], 2 * l + 1))


def _irfft(x: jnp.ndarray, res: int) -> jnp.ndarray:
    r"""Inverse of the real fourier transform
    Args:
        x (`jax.numpy.ndarray`): array of shape ``(..., 2*l + 1)``
        res (int): output resolution, has to be an odd number
    Returns:
        `jax.numpy.ndarray`: positions on the sphere, array of shape ``(..., res)``
    """
    assert res % 2 == 1

    l = (x.shape[-1] - 1) // 2
    x_reshaped = jnp.concatenate(
        [
            x[..., l : l + 1],
            (x[..., l + 1 :] + jnp.flip(x[..., :l], -1) * -1j) / jnp.sqrt(2),
            jnp.zeros((*x.shape[:-1], l), x.dtype),
        ],
        axis=-1,
    ).reshape((-1, x.shape[-1]))
    x_transformed = jnp.fft.irfft(x_reshaped, res)
    return x_transformed.reshape((*x.shape[:-1], x_transformed.shape[-1]))


def _expand_matrix(ls: List[int]) -> np.ndarray:
    """
    conversion matrix between a flatten vector (L, m) like that
    (0, 0) (1, -1) (1, 0) (1, 1) (2, -2) (2, -1) (2, 0) (2, 1) (2, 2)
    and a bidimensional matrix representation like that
                    (0, 0)
            (1, -1) (1, 0) (1, 1)
    (2, -2) (2, -1) (2, 0) (2, 1) (2, 2)

    Args:
        ls: list of l values
    Returns:
        array of shape ``[l, m, l * m]``
    """
    lmax = max(ls)
    m = np.zeros((lmax + 1, 2 * lmax + 1, sum(2 * l + 1 for l in ls)), np.float64)
    i = 0
    for l in ls:
        m[l, lmax - l : lmax + l + 1, i : i + 2 * l + 1] = np.eye(
            2 * l + 1, dtype=np.float64
        )
        i += 2 * l + 1
    return m


def _rollout_sh(m: jnp.ndarray, lmax: int) -> jnp.ndarray:
    """
    Expand spherical harmonic representation.
    Args:
        m (`jax.numpy.ndarray`): of shape (..., (lmax+1)*(lmax+2)/2)
    Returns:
        `jax.numpy.ndarray`: of shape (..., (lmax+1)**2)
    """
    assert m.shape[-1] == (lmax + 1) * (lmax + 2) // 2
    m_full = jnp.zeros((*m.shape[:-1], (lmax + 1) ** 2), dtype=m.dtype)
    for l in range(lmax + 1):
        i_mid = l**2 + l
        for i in range(l + 1):
            m_full = m_full.at[..., i_mid + i].set(m[..., l * (l + 1) // 2 + i])
            m_full = m_full.at[..., i_mid - i].set(m[..., l * (l + 1) // 2 + i])
    return m_full
