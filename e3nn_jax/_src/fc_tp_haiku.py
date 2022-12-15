import haiku as hk

import e3nn_jax as e3nn
from e3nn_jax._src.tensor_products import naive_broadcast_decorator


class FullyConnectedTensorProduct(hk.Module):
    def __init__(self, irreps_out, *, irreps_in1=None, irreps_in2=None):
        super().__init__()

        self.irreps_out = e3nn.Irreps(irreps_out)
        self.irreps_in1 = e3nn.Irreps(irreps_in1) if irreps_in1 is not None else None
        self.irreps_in2 = e3nn.Irreps(irreps_in2) if irreps_in2 is not None else None

    def __call__(self, x1: e3nn.IrrepsArray, x2: e3nn.IrrepsArray, **kwargs) -> e3nn.IrrepsArray:
        if self.irreps_in1 is not None:
            x1 = x1._convert(self.irreps_in1)
        if self.irreps_in2 is not None:
            x2 = x2._convert(self.irreps_in2)

        x1 = x1.remove_nones().simplify()
        x2 = x2.remove_nones().simplify()

        tp = e3nn.FunctionalFullyConnectedTensorProduct(x1.irreps, x2.irreps, self.irreps_out.simplify())
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
        f = naive_broadcast_decorator(lambda x1, x2: tp.left_right(ws, x1, x2, **kwargs))
        output = f(x1, x2)
        return output._convert(self.irreps_out)
