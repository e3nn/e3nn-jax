import jax.numpy as jnp


def sus(x):
    r"""Soft Unit Step function.

    ``-inf->0, 0->0, 2->0.6, +inf->1``

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
    r"""Projection on a basis of functions.

    Returns a set of :math:`\{y_i(x)\}_{i=1}^N`,

    .. math::

        y_i(x) = \frac{1}{Z} f_i(x)

    where :math:`x` is the input and :math:`f_i` is the ith basis function.
    :math:`Z` is a constant defined (if possible) such that,

    .. math::

        \langle \sum_{i=1}^N y_i(x)^2 \rangle_x \approx 1

    See the last plot below.
    Note that ``bessel`` basis cannot be normalized.

    Args:
        input (jnp.ndarray): input of shape ``[...]``
        start (float): minimum value span by the basis
        end (float): maximum value span by the basis
        number (int): number of basis functions :math:`N`
        basis (str): type of basis functions, one of ``gaussian``, ``cosine``, ``smooth_finite``, ``fourier``
        cutoff (bool): if ``True``, the basis functions are cutoff at the start and end of the interval
        start_zero (bool): if ``True``, the first basis function is zero at the start of the interval
        end_zero (bool): if ``True``, the last basis function is zero at the end of the interval

    Returns:
        jnp.ndarray: basis functions of shape ``[..., number]``

    Examples:

        .. jupyter-execute::
            :hide-code:

            import jax.numpy as jnp
            import numpy as np
            import e3nn_jax as e3nn
            import matplotlib.pyplot as plt

        .. jupyter-execute::

            bases = ["gaussian", "cosine", "smooth_finite", "fourier"]
            x = np.linspace(-1.0, 2.0, 200)

        .. jupyter-execute::

            fig, axss = plt.subplots(len(bases), 2, figsize=(9, 6), sharex=True, sharey=True)

            for axs, b in zip(axss, bases):
                for ax, c in zip(axs, [True, False]):
                    y = e3nn.soft_one_hot_linspace(x, start=-0.5, end=1.5, number=4, basis=b, cutoff=c)

                    plt.sca(ax)
                    plt.plot(x, y)
                    plt.plot([-0.5]*2, [-2, 2], "k-.")
                    plt.plot([1.5]*2, [-2, 2], "k-.")
                    plt.title(f"{b}" + (" with cutoff" if c else ""))

            plt.ylim(-1, 1.5)
            plt.tight_layout()

        .. jupyter-execute::

            fig, axss = plt.subplots(len(bases), 2, figsize=(9, 6), sharex=True, sharey=True)

            for axs, b in zip(axss, bases):
                for ax, c in zip(axs, [True, False]):
                    y = e3nn.soft_one_hot_linspace(x, start=-0.5, end=1.5, number=4, basis=b, cutoff=c)

                    plt.sca(ax)
                    plt.plot(x, np.sum(y**2, axis=-1))
                    plt.plot([-0.5]*2, [-2, 2], "k-.")
                    plt.plot([1.5]*2, [-2, 2], "k-.")
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
        return jnp.exp(-(diff**2)) / 1.12

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
            raise ValueError("when using fourier basis, start_zero and end_zero must be the same")

    raise ValueError(f'basis="{basis}" is not a valid entry')


def bessel(x: jnp.ndarray, n: int, x_max: float = 1.0) -> jnp.ndarray:
    r"""Bessel basis functions.

    Args:
        x (jnp.ndarray): input of shape ``[...]``
        n (int): number of basis functions
        x_max (float): maximum value of the input

    Returns:
        jnp.ndarray: basis functions of shape ``[..., n]``

    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (7)
    """
    x = x[..., None]
    n = jnp.arange(1, n + 1)
    x_nonzero = jnp.where(x == 0.0, 1.0, x)
    return jnp.sqrt(2.0 / x_max) * jnp.where(
        x == 0,
        n * jnp.pi / x_max,
        jnp.sin(n * jnp.pi / x_max * x_nonzero) / x_nonzero,
    )
