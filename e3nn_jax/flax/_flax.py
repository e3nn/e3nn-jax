from typing import Callable, List, Optional, Sequence, Tuple, Union

import flax
import jax

from e3nn_jax import FullyConnectedTensorProduct, Irreps, Linear, normalize_act


class FlaxLinear(flax.linen.Module):
    irreps_in: Irreps
    irreps_out: Irreps
    instructions: Optional[Tuple[int, int]] = None
    biases: Union[bool, List[bool]] = False
    weight_init: Callable = jax.random.normal
    bias_init: Callable = flax.linen.initializers.zeros

    @flax.linen.compact
    def __call__(self, x, output_list=False):
        lin = Linear(self.irreps_in, self.irreps_out, self.instructions, biases=self.biases)
        w = [
            self.param(f'bias {ins.i_out}', self.bias_init, ins.path_shape)
            if ins.i_in == -1 else
            self.param(f'weight {ins.i_in} -> {ins.i_out}', self.weight_init, ins.path_shape)
            for ins in lin.instructions
        ]
        return lin(w, x, output_list=output_list)


class FlaxFullyConnectedTensorProduct(flax.linen.Module):
    irreps_in1: Irreps
    irreps_in2: Irreps
    irreps_out: Irreps
    weight_init: Callable = jax.random.normal

    @flax.linen.compact
    def __call__(self, x1, x2):
        tp = FullyConnectedTensorProduct(self.irreps_in1, self.irreps_in2, self.irreps_out)
        ws = [self.param(f'weight {ins.i_in1} x {ins.i_in2} -> {ins.i_out}', self.weight_init, ins.path_shape) for ins in tp.instructions]
        return tp.left_right(ws, x1, x2)


class MLP(flax.linen.Module):
    features: Sequence[int]
    phi: Callable

    @flax.linen.compact
    def __call__(self, x):
        phi = normalize_act(self.phi)

        for feat in self.features[:-1]:
            d = flax.linen.Dense(feat, kernel_init=jax.random.normal, use_bias=False)
            x = phi(d(x) / x.shape[-1]**0.5)

        h = self.features[-1]
        d = flax.linen.Dense(h, kernel_init=jax.random.normal, use_bias=False)
        x = d(x) / x.shape[-1]
        return x
