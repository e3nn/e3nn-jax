import haiku as hk
import jax.numpy as jnp

import e3nn_jax as e3nn
from e3nn_jax.legacy import FunctionalFullyConnectedTensorProduct


class FullyConnectedTensorProduct(hk.Module):
    def __init__(self, irreps_out, *, irreps_in1=None, irreps_in2=None):
        super().__init__()

        self.irreps_out = e3nn.Irreps(irreps_out)
        self.irreps_in1 = e3nn.Irreps(irreps_in1) if irreps_in1 is not None else None
        self.irreps_in2 = e3nn.Irreps(irreps_in2) if irreps_in2 is not None else None

    def __call__(
        self, x1: e3nn.IrrepsArray, x2: e3nn.IrrepsArray, **kwargs
    ) -> e3nn.IrrepsArray:
        x1 = e3nn.as_irreps_array(x1)
        x2 = e3nn.as_irreps_array(x2)

        leading_shape = jnp.broadcast_shapes(x1.shape[:-1], x2.shape[:-1])
        x1 = x1.broadcast_to(leading_shape + (-1,))
        x2 = x2.broadcast_to(leading_shape + (-1,))

        if self.irreps_in1 is not None:
            x1 = x1.rechunk(self.irreps_in1)
        if self.irreps_in2 is not None:
            x2 = x2.rechunk(self.irreps_in2)

        x1 = x1.remove_zero_chunks().simplify()
        x2 = x2.remove_zero_chunks().simplify()

        tp = FunctionalFullyConnectedTensorProduct(
            x1.irreps, x2.irreps, self.irreps_out.simplify()
        )
        ws = [
            hk.get_parameter(
                (
                    f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] "
                    f"{tp.irreps_in1[ins.i_in1]},{tp.irreps_in2[ins.i_in2]},{tp.irreps_out[ins.i_out]}"
                ),
                shape=ins.path_shape,
                init=hk.initializers.RandomNormal(stddev=ins.weight_std),
            )
            for ins in tp.instructions
        ]
        f = lambda x1, x2: tp.left_right(ws, x1, x2, **kwargs)

        for _ in range(len(leading_shape)):
            f = e3nn.utils.vmap(f)

        output = f(x1, x2)
        return output.rechunk(self.irreps_out)
