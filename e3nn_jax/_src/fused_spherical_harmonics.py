"""
This module provides functionality to calculate the associated Legendre polynomials and
spherical harmonics for a given set of points and degree, using JAX for automatic differentiation
and JIT (Just-In-Time) compilation to GPU/TPU for increased performance.

Functions:
    get_klm(degree: int) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:
        Calculate and cache the k, l, m values, along with powers and a sign mask for spherical harmonics.

    get_factorial(N: int, nan: bool = False) -> jnp.array:
        Calculate and cache factorial values up to N, with an option to extend the array with NaN values.

    get_A(degree: int) -> jnp.array:
        Calculate and cache the 'A' coefficients used in the associated Legendre polynomials.

    calc_Plm_jit(x: jnp.array, A: jnp.array, m: jnp.array, power: jnp.array) -> jnp.array:
        JIT-compiled function to calculate the associated Legendre polynomials using pre-computed 'A' coefficients.

    calc_Plm(degree: int, x: jnp.array) -> jnp.array:
        Calculate the associated Legendre polynomials for a given degree and 'x' values.

    get_B(degree: int) -> jnp.array:
        Calculate and cache the 'B' coefficients used in the spherical harmonics.

    calc_Ylm_jit(x: jnp.array, B: jnp.array, m: jnp.array, power: jnp.array, mask: jnp.array) -> jnp.array:
        JIT-compiled function to calculate spherical harmonics using pre-computed 'B' coefficients.

    calc_Ylm(degree: int, x: jnp.array) -> jnp.array:
        Calculate the spherical harmonics for a given degree and set of 3D points.

Typical usage example:
    # Calculate associated Legendre polynomials P(l,m) for degree=3 and x values
    plm_values = calc_Plm(3, x_array)

    # Calculate spherical harmonics Y(l,m) for degree=3 and 3D points
    ylm_values = calc_Ylm(3, xyz_array)
"""
from functools import lru_cache
import math
from typing import Tuple

import jax
import jax.numpy as jnp

@lru_cache(maxsize=None)
def get_klm(degree: int) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:
    """
    Compute and cache the spherical harmonics parameters 'k', 'l', 'm', 'power', and 'mask'.

    Args:
        degree (int): The maximum degree -1 for the spherical harmonics.

    Returns:
        Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]: A tuple containing the
        'k', 'l', 'm' indices, the 'power' values for each term, and the 'mask' array that
        determines the sign based on 'm' values.
    """
    k = jnp.arange(degree).reshape(1, -1)
    lm_list = [[d, m] for d in range(degree) for m in range(-d, d + 1)]
    l, m = jnp.array(lm_list).T.reshape(2, -1, 1)
    mask = m >= 0
    m = jnp.abs(m)
    power = 2 * k - l - m
    power = jnp.where(power < 0, 0, power)
    return k, l, m, power, mask

@lru_cache(maxsize=None)
def get_factorial(N, nan=False):
    """
    Compute and cache the factorial values up to N, with optional extension of NaN values.

    Args:
        N (int): The maximum value for which to compute the factorial.
        nan (bool, optional): If True, extend the array with NaN values. Default is False.

    Returns:
        jnp.array: An array of factorial values from 0 to N, optionally extended with NaN values.
    """
    ints = jnp.arange(N, dtype=jnp.float32)
    ints = ints.at[0].set(1)  # factorial(0) = 1
    factorial = jnp.cumprod(ints, axis=0)
    if not nan:
        return factorial
    nan_tensor = jnp.full_like(factorial, jnp.nan)
    return jnp.concatenate((factorial, nan_tensor))

@lru_cache(maxsize=None)
def get_A(degree: int) -> jnp.array:
    """
    Compute and cache the 'A' coefficients used in the associated Legendre polynomials calculation.

    Args:
        degree (int): The maximum degree -1 of the associated Legendre polynomial.

    Returns:
        jnp.array: An array representing the 'A' coefficients.
    """
    k, l, m, _, _ = get_klm(degree)  # Use abs_m
    factorial = get_factorial(2 * degree, nan=True)  # Use float for nan=True

    A = (-1) ** (m + l - k) / (2 ** l) * factorial[2 * k] / factorial[k] / factorial[l - k] / factorial[2 * k - l - m]
    A = jnp.where(jnp.isnan(A), 0.0, A)
    return A

@jax.jit
def calc_Plm_jit(x: jnp.array, A: jnp.array, m: jnp.array, power: jnp.array) -> jnp.array:
    """
    JIT-compiled function to calculate the associated Legendre polynomials P(l,m) using
    pre-computed 'A' coefficients.

    Args:
        x (jnp.array): The input values for cos(theta), where theta is the polar angle.
        A (jnp.array): The pre-computed 'A' coefficients.
        m (jnp.array): The 'm' indices for the associated Legendre polynomials.
        power (jnp.array): The powers to raise 'x' to for each term in the polynomial.

    Returns:
        jnp.array: An array of the associated Legendre polynomial values P(l,m).
    """
    x = x[:, None]
    temp = jnp.power(x[None, :], power[:, None])
    pre_Plm = (temp @ A[:, None]).squeeze(2).T
    Plm = pre_Plm * (1 - x**2) ** (m.T / 2)
    return Plm

def calc_Plm(degree: int, x: jnp.array) -> jnp.array:
    """
    Calculate the associated Legendre polynomials P(l,m) for a given degree and set of 'x' values.

    Args:
        degree (int): The maximum degree -1 of the associated Legendre polynomial.
        x (jnp.array): The input values for cos(theta), where theta is the polar angle.

    Returns:
        jnp.array: An array of the associated Legendre polynomial values P(l,m).
    """
    B = get_A(degree)
    _, _, m, power, _ = get_klm(degree)
    Plm = calc_Plm_jit(x, B, m, power)
    return Plm

@lru_cache(maxsize=None)
def get_B(degree):
    """
    Compute and cache the 'B' coefficients used in the spherical harmonics calculation.

    Args:
        degree (int): The maximum degree -1 of the spherical harmonics.

    Returns:
        jnp.array: An array representing the 'B' coefficients.
    """
    _, l, m, _, _ = get_klm(degree)
    factorial = get_factorial(2 * degree)
    A = get_A(degree)

    B = A * (-1) ** m * 2 ** (m != 0) / 2 * jnp.sqrt((2 * l + 1) / (4 * math.pi) * factorial[l - m] / factorial[l + m])
    return B

@jax.jit
def calc_Ylm_jit(x: jnp.array, B: jnp.array, m: jnp.array, power: jnp.array, mask: jnp.array) -> jnp.array:
    """
    JIT-compiled function to calculate the spherical harmonics Y(l,m) using pre-computed 'B' coefficients.

    Args:
        x (jnp.array): The input 3D points on which to evaluate the spherical harmonics.
        B (jnp.array): The pre-computed 'B' coefficients.
        m (jnp.array): The 'm' indices for the spherical harmonics.
        power (jnp.array): The powers associated with the 'l' and 'm' indices.
        mask (jnp.array): The mask determining the sign based on 'm' values.

    Returns:
        jnp.array: An array of the spherical harmonics Y(l,m) values.
    """
    r = jnp.linalg.norm(x, axis=1, keepdims=True)
    cos_theta = x[..., 2:3] / r
    phi = jnp.arctan2(x[..., 1:2], x[..., 0:1])

    cos_pows = cos_theta ** power[:, None]
    pre_Plm = (cos_pows @ B[:, :, None]).squeeze(2).T
    sin_pows = jnp.sqrt(1 - cos_theta**2) ** m.T
    Plm = pre_Plm * sin_pows

    m_phi = m.T * phi
    cos_values = jnp.cos(m_phi)
    sin_values = jnp.sin(m_phi)
    cases = jnp.where(mask.T, cos_values, sin_values)
    return Plm * cases

def calc_Ylm(degree: int, x: jnp.array) -> jnp.array:
    """
    Calculate the spherical harmonics Y(l,m) for a given degree and set of 3D points.

    Args:
        degree (int): The maximum degree -1 of the spherical harmonics.
        x (jnp.array): The input 3D points on which to evaluate the spherical harmonics.

    Returns:
        jnp.array: An array of the spherical harmonics Y(l,m) values for the given points.
    """
    B = get_B(degree)
    _, _, m, power, mask = get_klm(degree)

    Ylm = calc_Ylm_jit(x, B, m, power, mask)
    return Ylm
