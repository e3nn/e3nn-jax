import jax
import jax.numpy as jnp
import haiku as hk

from e3nn_jax import Irreps, IrrepsData


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

    def __init__(self, p, *, irreps=None):
        super().__init__()
        self.irreps = Irreps(irreps) if irreps is not None else None
        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__} (p={self.p})"

    def __call__(self, rng, x: IrrepsData, is_training=True) -> IrrepsData:
        """equivariant dropout

        Args:
            rng (`jax.random.PRNGKey`): the random number generator
            x (IrrepsData): the input
            is_training (bool): whether to perform dropout

        Returns:
            IrrepsData: the output
        """
        if not is_training:
            return x

        if self.irreps is not None:
            x = IrrepsData.new(self.irreps, x)
        if not isinstance(x, IrrepsData):
            raise TypeError(f"{self.__class__.__name__} only supports IrrepsData")

        noises = []
        out_list = []
        for (mul, ir), a in zip(x.irreps, x.list):
            if self.p >= 1:
                out_list.append(None)
                noises.append(jnp.zeros((mul * ir.dim,)))
            elif self.p <= 0:
                out_list.append(a)
                noises.append(jnp.ones((mul * ir.dim,)))
            else:
                noise = jax.random.bernoulli(rng, p=1 - self.p, shape=(mul, 1)) / (1 - self.p)
                out_list.append(noise * a)
                noises.append(jnp.repeat(noise, ir.dim, axis=1).flatten())

        noises = jnp.concatenate(noises)
        return IrrepsData(irreps=x.irreps, contiguous=x.contiguous * noises, list=out_list)
