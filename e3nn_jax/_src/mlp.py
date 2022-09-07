from typing import Callable, Sequence, Union

import haiku as hk
import jax.numpy as jnp

from e3nn_jax import config, normalize_function


class MultiLayerPerceptron(hk.Module):
    def __init__(
        self,
        list_neurons: Sequence[int],
        act: Callable,
        *,
        gradient_normalization: Union[str, float] = None,
    ):
        super().__init__()

        self.list_neurons = list_neurons
        self.act = act

        if gradient_normalization is None:
            gradient_normalization = config("gradient_normalization")
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[gradient_normalization]
        self.gradient_normalization = gradient_normalization

    def __call__(self, x):
        act = None if self.act is None else normalize_function(self.act)

        for h in self.list_neurons:
            alpha = 1 / x.shape[-1]
            d = hk.Linear(
                h,
                with_bias=False,
                w_init=hk.initializers.RandomNormal(stddev=jnp.sqrt(alpha) ** (1.0 - self.gradient_normalization)),
            )
            x = jnp.sqrt(alpha) ** self.gradient_normalization * d(x)
            if act is not None:
                x = act(x)

        return x
