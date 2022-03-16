from typing import Callable, Sequence

import haiku as hk
import jax.numpy as jnp
from e3nn_jax import Irreps, FunctionalTensorProduct, normalize_function


class MultiLayerPerceptron(hk.Module):
    def __init__(self, list_neurons: Sequence[int], phi: Callable):
        super().__init__()

        self.list_neurons = list_neurons
        self.phi = phi

    def __call__(self, x):
        phi = normalize_function(self.phi)

        for h in self.list_neurons:
            d = hk.Linear(h, with_bias=False, w_init=hk.initializers.RandomNormal())
            x = phi(d(x) / x.shape[-1]**0.5)

        return x


class TensorProductMultiLayerPerceptron(hk.Module):
    irreps_in1: Irreps
    irreps_in2: Irreps
    irreps_out: Irreps

    def __init__(self, tp: FunctionalTensorProduct, list_neurons: Sequence[int], phi: Callable):
        super().__init__()

        self.tp = tp
        self.mlp = MultiLayerPerceptron(list_neurons, phi)

        self.irreps_in1 = tp.irreps_in1.simplify()
        self.irreps_in2 = tp.irreps_in2.simplify()
        self.irreps_out = tp.irreps_out.simplify()

        assert all(i.has_weight for i in self.tp.instructions)

    def __call__(self, emb, x1, x2):
        w = self.mlp(emb)

        w = [
            jnp.einsum(
                "x...,x->...",
                hk.get_parameter(
                    f'{i.i_in1} x {i.i_in2} -> {i.i_out}',
                    shape=(w.shape[0],) + i.path_shape,
                    init=hk.initializers.RandomNormal()
                ) / w.shape[0]**0.5,
                w
            )
            for i in self.tp.instructions
        ]
        return self.tp.left_right(w, x1, x2)
