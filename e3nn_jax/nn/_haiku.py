# TODO remove all these classes and remove e3nn_jax.nn
from typing import Callable, Sequence

import haiku as hk
import jax.numpy as jnp
from e3nn_jax import Irreps, FunctionalTensorProduct, TensorSquare, normalize_function


class HTensorSquare(hk.Module):
    def __init__(self, irreps_in, irreps_out, init=None):
        super().__init__()

        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)

        if init is None:
            init = hk.initializers.RandomNormal()
        self.init = init

    def __call__(self, x):
        tp = TensorSquare(self.irreps_in, self.irreps_out)
        ws = [
            hk.get_parameter(f'weight {i}', shape=ins.path_shape, init=self.init)
            for i, ins in enumerate(tp.instructions)
        ]
        return tp.left_right(ws, x, x)


class HMLP(hk.Module):
    def __init__(self, features: Sequence[int], phi: Callable):
        super().__init__()

        self.features = features
        self.phi = phi

    def __call__(self, x):
        phi = normalize_function(self.phi)

        for h in self.features:
            d = hk.Linear(h, with_bias=False, w_init=hk.initializers.RandomNormal())
            x = phi(d(x) / x.shape[-1]**0.5)

        return x


class HTensorProductMLP(hk.Module):
    irreps_in1: Irreps
    irreps_in2: Irreps
    irreps_out: Irreps

    def __init__(self, tp: FunctionalTensorProduct, features: Sequence[int], phi: Callable):
        super().__init__()

        self.tp = tp
        self.mlp = HMLP(features, phi)

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
