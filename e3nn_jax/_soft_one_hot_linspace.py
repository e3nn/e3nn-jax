import jax.numpy as jnp


def sus(x):
    r"""Soft Unit Step function. ``-inf->0, 0->0, 2->0.6, +inf->1``

    .. math::
        \text{sus}(x) = \begin{cases}
            0, & \text{if } x < 0 \\
            exp(-1/x), & \text{if } x \geq 0 \\
        \end{cases}

    """
    return jnp.where(x > 0.0, jnp.exp(-1.0 / jnp.where(x > 0.0, x, 1.0)), 0.0)


def soft_one_hot_linspace(
    input: jnp.ndarray,
    *,
    start: float,
    end: float,
    number: int,
    basis: str = None,
    cutoff: bool = None,
    start_zero: bool = None,
    end_zero: bool = None,
):
    r"""Projection on a basis of functions

    Returns a set of :math:`\{y_i(x)\}_{i=1}^N`,

    .. math::

        y_i(x) = \frac{1}{Z} f_i(x)

    where :math:`x` is the input and :math:`f_i` is the ith basis function.
    :math:`Z` is a constant defined (if possible) such that,

    .. math::

        \langle \sum_{i=1}^N y_i(x)^2 \rangle_x \approx 1

    See the last plot below.
    Note that ``bessel`` basis cannot be normalized.

    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape :math:`(...)`

    start : float
        minimum value span by the basis

    end : float
        maximum value span by the basis

    number : int
        number of basis functions :math:`N`

    basis : {'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}
        choice of basis family; note that due to the :math:`1/x` term, ``bessel`` basis does not satisfy the normalization of other basis choices

    start_zero : bool
        if ``True``, the first basis function is forced to be zero (or close) at ``start``

    end_zero : bool
        if ``True``, the last basis function is forced to be zero (or close) at ``end``

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., N)`

    Examples
    --------

    .. jupyter-execute::
        :hide-code:

        import torch
        from e3nn.math import soft_one_hot_linspace
        import matplotlib.pyplot as plt

    .. jupyter-execute::

        bases = ['gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel']
        x = torch.linspace(-1.0, 2.0, 100)

    .. jupyter-execute::

        fig, axss = plt.subplots(len(bases), 2, figsize=(9, 6), sharex=True, sharey=True)

        for axs, b in zip(axss, bases):
            for ax, c in zip(axs, [True, False]):
                plt.sca(ax)
                try:
                    plt.plot(x, soft_one_hot_linspace(x, start=-0.5, end=1.5, number=4, basis=b, cutoff=c))
                except NotImplementedError:
                    pass
                plt.plot([-0.5]*2, [-2, 2], 'k-.')
                plt.plot([1.5]*2, [-2, 2], 'k-.')
                plt.title(f"{b}" + (" with cutoff" if c else ""))

        plt.ylim(-1, 1.5)
        plt.tight_layout()

    .. jupyter-execute::

        fig, axss = plt.subplots(len(bases), 2, figsize=(9, 6), sharex=True, sharey=True)

        for axs, b in zip(axss, bases):
            for ax, c in zip(axs, [True, False]):
                plt.sca(ax)
                try:
                    plt.plot(x, soft_one_hot_linspace(x, start=-0.5, end=1.5, number=4, basis=b, cutoff=c).pow(2).sum(1))
                except NotImplementedError:
                    pass
                plt.plot([-0.5]*2, [-2, 2], 'k-.')
                plt.plot([1.5]*2, [-2, 2], 'k-.')
                plt.title(f"{b}" + (" with cutoff" if c else ""))

        plt.ylim(0, 2)
        plt.tight_layout()
    """
    if cutoff is not None:
        assert start_zero is None
        assert end_zero is None
        start_zero = cutoff
        end_zero = cutoff

    del cutoff

    if start_zero not in [True, False]:
        raise ValueError("start_zero must be specified")

    if end_zero not in [True, False]:
        raise ValueError("end_zero must be specified")

    if start_zero and end_zero:
        values = jnp.linspace(start, end, number + 2)
        step = values[1] - values[0]
        values = values[1:-1]

    if start_zero and not end_zero:
        values = jnp.linspace(start, end, number + 1)
        step = values[1] - values[0]
        values = values[1:]

    if not start_zero and end_zero:
        values = jnp.linspace(start, end, number + 1)
        step = values[1] - values[0]
        values = values[:-1]

    if not start_zero and not end_zero:
        values = jnp.linspace(start, end, number)
        step = values[1] - values[0]

    diff = (input[..., None] - values) / step

    if basis == "gaussian":
        return jnp.exp(-(diff ** 2)) / 1.12

    if basis == "cosine":
        return jnp.where((-1.0 < diff) & (diff < 1.0), jnp.cos(jnp.pi / 2 * diff), 0.0)

    if basis == "smooth_finite":
        return 1.14136 * jnp.exp(2.0) * sus(diff + 1.0) * sus(1.0 - diff)

    if basis == "fourier":
        x = (input[..., None] - start) / (end - start)
        if start_zero and end_zero:
            i = jnp.arange(1, number + 1)
            return jnp.where((0.0 < x) & (x < 1.0), jnp.sin(jnp.pi * i * x) / jnp.sqrt(0.25 + number / 2), 0.0)
        elif not start_zero and not end_zero:
            i = jnp.arange(0, number)
            return jnp.cos(jnp.pi * i * x) / jnp.sqrt(0.25 + number / 2)
        else:
            raise NotImplementedError

    if basis == "bessel":
        x = input[..., None] - start
        c = end - start
        bessel_roots = jnp.arange(1, number + 1) * jnp.pi
        out = jnp.sqrt(2 / c) * jnp.sin(bessel_roots * x / c) / x

        if not start_zero and not end_zero:
            return out
        else:
            raise NotImplementedError

    raise ValueError(f'basis="{basis}" is not a valid entry')
