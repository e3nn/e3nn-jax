from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp

import e3nn_jax as e3nn

from .s2grid import SphericalSignal


class SO3Signal:
    r"""Representation of a signal on SO(3) via a grid of signals on S2."""

    def __init__(
        self,
        s2_signals: SphericalSignal,
        *,
        _perform_checks: bool = True,
    ) -> None:
        if _perform_checks:
            if len(s2_signals.shape) < 3:
                raise ValueError(
                    f"s2_signals should have atleast 3 axes. Got {s2_signals.shape}."
                )

        self.s2_signals = s2_signals

    @property
    def batch_dims(self) -> Tuple[int, ...]:
        return self.s2_signals.shape[:-3]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.s2_signals.shape

    @property
    def res_beta(self) -> int:
        return self.s2_signals.res_beta

    @property
    def res_alpha(self) -> int:
        return self.s2_signals.res_alpha

    @property
    def grid_values(self) -> jnp.ndarray:
        return self.s2_signals.grid_values

    @property
    def res_theta(self) -> int:
        return self.s2_signals.shape[-3]

    @property
    def grid_theta(self) -> jnp.ndarray:
        return jnp.linspace(0, 2 * jnp.pi, self.res_theta)

    @property
    def grid_resolution(self) -> str:
        return (self.res_theta, self.res_beta, self.res_alpha)

    @staticmethod
    def from_function(
        func: Callable[[jax.Array], float],
        res_beta: int,
        res_alpha: int,
        res_theta: int,
        quadrature: str,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> "SO3Signal":
        """Create a signal on the sphere from a function of rotations.

        Args:
            func (`Callable`): function on the sphere that maps a 3x3 rotation matrix to a scalar or vector.
            res_theta: resolution for theta (for the angle in the axis-angle parametrization)
            res_beta: resolution for beta (for the axis in the axis-angle parametrization)
            res_alpha: resolution for alpha (for the axis in the axis-angle parametrization)
            quadrature: quadrature to use
            dtype: dtype of the signal

        Returns:
            `SO3Signal`: signal on SO(3)
        """
        # Construct the grid over S2 for the axis in the axis-angle parametrization.
        # The shape of s2_signals will be different at the end,
        # but we just extract the axes (via grid_vectors) from it now.
        s2_signals = e3nn.SphericalSignal.zeros(
            res_beta, res_alpha, quadrature=quadrature, dtype=dtype
        )
        axes = s2_signals.grid_vectors
        angles = jnp.linspace(0, 2 * jnp.pi, res_theta)

        assert axes.shape == (res_beta, res_alpha, 3)
        assert angles.shape == (res_theta,)

        # Construct the rotation matrices for each (axis, angle) pair.
        Rs_fn = e3nn.axis_angle_to_matrix
        Rs_fn = jax.vmap(Rs_fn, in_axes=(0, None))
        Rs_fn = jax.vmap(Rs_fn, in_axes=(None, 0))
        Rs = Rs_fn(axes, angles)
        assert Rs.shape == (res_theta, res_beta, res_alpha, 3, 3)

        # Evaluate the function at each rotation matrix.
        func_fn = jax.vmap(jax.vmap(jax.vmap(func)))
        fs = func_fn(Rs)

        # Move the function output axes to the front.
        if fs.ndim > 3:
            for _ in range(fs.ndim - 3):
                fs = jnp.moveaxis(fs, -1, 0)

        batch_dims = fs.shape[:-3]
        assert fs.shape == (*batch_dims, res_theta, res_beta, res_alpha)

        s2_signals = s2_signals.replace_values(fs)
        assert s2_signals.shape == (*batch_dims, res_theta, res_beta, res_alpha)
        return SO3Signal(s2_signals)

    def __mul__(self, other: Union[float, "SO3Signal"]) -> "SO3Signal":
        if isinstance(other, SO3Signal):
            if self.shape != other.shape:
                raise ValueError(
                    f"Shapes of the two signals do not match: {self.shape} != {other.shape}"
                )
            return SO3Signal(self.s2_signals * other.s2_signals)

        return SO3Signal(self.s2_signals * other)

    def __truediv__(self, other: float) -> "SO3Signal":
        return self * (1 / other)

    def vmap_over_batch_dims(
        self, func: Callable[..., jnp.ndarray]
    ) -> Callable[..., jnp.ndarray]:
        """Apply a function to the signal while preserving the batch dimensions."""
        for _ in range(len(self.batch_dims)):
            func = jax.vmap(func)
        return func

    def argmax(
        self,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Find the rotation (and corresponding grid indices) with the maximum value of the signal."""
        # Get flattened argmax
        flat_index = jnp.argmax(self.grid_values.reshape(*self.shape[:-3], -1), axis=-1)

        # Convert flat index back to indices for theta, beta, alpha
        theta_idx, beta_idx, alpha_idx = jnp.unravel_index(flat_index, self.shape[-3:])

        # Compute axis.
        axis = self.s2_signals.grid_vectors[..., beta_idx, alpha_idx, :]
        assert axis.shape == (*self.batch_dims, 3)

        # Compute angle.
        angle = self.grid_theta[theta_idx]
        assert angle.shape == (*self.batch_dims,)

        Rs = self.vmap_over_batch_dims(e3nn.axis_angle_to_matrix)(axis, angle)
        assert Rs.shape == (*self.batch_dims, 3, 3)

        return Rs, (theta_idx, beta_idx, alpha_idx)

    def replace_values(self, grid_values: jnp.ndarray) -> "SO3Signal":
        """Replace the values of the signal with the given grid_values."""
        return SO3Signal(self.s2_signals.replace_values(grid_values))

    def integrate_over_angles(self) -> SphericalSignal:
        """Integrate the signal over the angles in the axis-angle parametrization."""
        # Account for angle-dependency in Haar measure.
        grid_values = self.grid_values * (1 - jnp.cos(self.grid_theta))[..., None, None]

        # Trapezoidal rule for integration.
        delta_theta = self.grid_theta[1] - self.grid_theta[0]
        return self.s2_signals.replace_values(
            grid_values=jnp.sum(grid_values, axis=-3) * delta_theta
        )

    def integrate(self) -> float:
        """Numerically integrate the signal over SO(3)."""
        # Integrate over angles.
        sig_integrated = self.integrate_over_angles()
        assert sig_integrated.shape == (
            *self.batch_dims,
            self.res_beta,
            self.res_alpha,
        )

        # Integrate over axes using S2 quadrature.
        integral = sig_integrated.integrate().array.squeeze(-1)
        assert integral.shape == self.batch_dims

        # Factor of 8pi^2 from the Haar measure.
        integral = integral / (8 * jnp.pi**2)
        return integral

    def sample(self, rng: jax.random.PRNGKey) -> jnp.ndarray:
        """Sample a random rotation from SO(3) using the given probability distribution."""
        # Integrate over angles.
        sig_integrated = self.integrate_over_angles()
        assert sig_integrated.shape == (
            *self.batch_dims,
            self.res_beta,
            self.res_alpha,
        )

        # Sample the axis from the S2 signal (integrated over angles).
        axis_rng, rng = jax.random.split(rng)
        beta_idx, alpha_idx = sig_integrated.sample(axis_rng)
        axis = sig_integrated.grid_vectors[..., beta_idx, alpha_idx, :]
        assert axis.shape == (*self.batch_dims, 3)

        # Choose the angle from the distribution conditioned on the axis.
        angle_rng, rng = jax.random.split(rng)
        theta_probs = self.grid_values[..., beta_idx, alpha_idx]
        assert theta_probs.shape == (*self.batch_dims, self.res_theta)

        # Avoid log(0) by replacing 0 with a small value.
        theta_logits = jnp.where(theta_probs == 0, 1e-20, theta_probs)
        theta_logits = jnp.log(theta_logits)

        theta_idx = jax.random.categorical(angle_rng, theta_logits)
        angle = jnp.linspace(0, 2 * jnp.pi, self.res_theta)[theta_idx]
        assert angle.shape == (*self.batch_dims,)

        Rs = self.vmap_over_batch_dims(e3nn.axis_angle_to_matrix)(axis, angle)
        assert Rs.shape == (*self.batch_dims, 3, 3)
        return Rs
