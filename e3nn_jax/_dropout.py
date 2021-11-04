import jax
import jax.numpy as jnp
import haiku as hk

from e3nn_jax import Irreps


class Dropout(hk.Module):
    """Equivariant Dropout
    :math:`A_{zai}` is the input and :math:`B_{zai}` is the output where
    - ``z`` is the batch index
    - ``a`` any non-batch and non-irrep index
    - ``i`` is the irrep index, for instance if ``irreps="0e + 2x1e"`` then ``i=2`` select the *second vector*
    .. math::
        B_{zai} = \frac{x_{zi}}{1-p} A_{zai}
    where :math:`p` is the dropout probability and :math:`x` is a Bernoulli random variable with parameter :math:`1-p`.

    Args:
        irreps (`Irreps`): the irrep string
        p (float): dropout probability

    Returns:
        `Dropout`: the dropout module
    """
    def __init__(self, irreps, p):
        super().__init__()
        self.irreps = Irreps(irreps)
        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, p={self.p})"

    def __call__(self, rng, x, is_training=True):
        """equivariant dropout

        Args:
            rng (`jax.random.PRNGKey`): the random number generator
            x (`jnp.ndarray`): the input
            is_training (bool): whether to perform dropout

        Returns:
            `jnp.ndarray`: the output
        """
        if not is_training:
            return x

        batch = x.shape[0]

        noises = []
        for mul, ir in self.irreps:
            dim = ir.dim

            if self.p >= 1:
                noise = jnp.zeros((batch, mul, 1), dtype=x.dtype)
            elif self.p <= 0:
                noise = jnp.ones((batch, mul, 1), dtype=x.dtype)
            else:
                noise = jax.random.bernoulli(rng, p=1 - self.p, shape=(batch, mul, 1)) / (1 - self.p)

            noise = jnp.tile(noise, (1, 1, dim)).reshape(batch, mul * dim)
            noises.append(noise)

        noise = jnp.concatenate(noises, axis=-1)
        while len(noise.shape) < len(x.shape):
            noise = noise[:, None]
        return x * noise
