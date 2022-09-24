r"""Transformation between two representations of a signal on the sphere.

.. math:: f: S^2 \longrightarrow \mathbb{R}

is a signal on the sphere.

One representation that we like to call "spherical tensor" is

.. math:: f(x) = \sum_{l=0}^{l_{\mathit{max}}} F^l \cdot Y^l(x)

it is made of :math:`(l_{\mathit{max}} + 1)^2` real numbers represented in the above formula by the familly of vectors
:math:`F^l \in \mathbb{R}^{2l+1}`.

Another representation is the discretization around the sphere. For this representation we chose a particular grid of size
:math:`(N, M)`

.. math::

    x_{ij} &= (\sin(\beta_i) \sin(\alpha_j), \cos(\beta_i), \sin(\beta_i) \cos(\alpha_j))

    \beta_i &= \pi (i + 0.5) / N

    \alpha_j &= 2 \pi j / M

In the code, :math:`N` is called ``res_beta`` and :math:`M` is ``res_alpha``.

The discrete representation is therefore

.. math:: \{ h_{ij} = f(x_{ij}) \}_{ij}
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Sequence, Tuple, Union

import so3

# what do the dtypes have to be? I just removed them for now but idk if that was a good idea

def s2_grid(res_beta: int, res_alpha: int):
    r"""grid on the sphere
    Parameters
    ----------
    res_beta : int
        :math:`N`
    res_alpha : int
        :math:`M`
    Returns
    -------
    betas : `torch.Tensor`
        tensor of shape ``(res_beta)``
    alphas : `torch.Tensor`
        tensor of shape ``(res_alpha)``
    """

    i = jnp.arange(res_beta)
    betas = (i + 0.5) / res_beta * jnp.pi

    i = jnp.arange(res_alpha)
    alphas = i / res_alpha * 2 * jnp.pi
    return betas, alphas

