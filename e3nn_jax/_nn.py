from typing import Callable, Sequence

import haiku as hk
from e3nn_jax import normalize_function


class MultiLayerPerceptron(hk.Module):
    def __init__(self, list_neurons: Sequence[int], act: Callable):
        super().__init__()

        self.list_neurons = list_neurons
        self.act = act

    def __call__(self, x):
        act = normalize_function(self.act)

        for h in self.list_neurons:
            d = hk.Linear(h, with_bias=False, w_init=hk.initializers.RandomNormal())
            x = act(d(x) / x.shape[-1] ** 0.5)

        return x
