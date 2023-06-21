"""
Implementation from MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields
Ilyes Batatia, Dávid Péter Kovács, Gregor N. C. Simm, Christoph Ortner and Gábor Csányi
"""
from typing import Any, Callable, Optional, Set, Tuple

import haiku as hk
import jax.numpy as jnp

import e3nn_jax as e3nn


class SymmetricTensorProduct(hk.Module):
    r"""Symmetric tensor product contraction with parameters

    Equivalent to the following code executed in parallel on the channel dimension::

        e3nn.haiku.Linear(irreps_out)(
            e3nn.concatenate([
                x,
                tensor_product(x, x),  # additionally keeping only the symmetric terms
                tensor_product(tensor_product(x, x), x),
                ...
            ])
        )

    Each channel has its own parameters.

    Args:
        orders (tuple of int): orders of the tensor product
        keep_irrep_out (optional, set of Irrep): irreps to keep in the output
        get_parameter (optional, callable): function to get the parameters, by default it uses ``hk.get_parameter``
            it should have the signature ``get_parameter(name, shape) -> ndarray`` and return a normal distribution
            with variance 1
    """

    def __init__(
        self,
        orders: Tuple[int, ...],
        keep_irrep_out: Optional[Set[e3nn.Irrep]] = None,
        get_parameter: Optional[
            Callable[[str, Tuple[int, ...], Any], jnp.ndarray]
        ] = None,
    ):
        super().__init__()

        orders = tuple(orders)
        assert all(isinstance(order, int) for order in orders)
        assert all(order > 0 for order in orders)
        self.orders = orders

        if isinstance(keep_irrep_out, str):
            keep_irrep_out = e3nn.Irreps(keep_irrep_out)
            assert all(mul == 1 for mul, _ in keep_irrep_out)

        if keep_irrep_out is not None:
            keep_irrep_out = {e3nn.Irrep(ir) for ir in keep_irrep_out}

        self.keep_irrep_out = keep_irrep_out

        if get_parameter is None:
            get_parameter = lambda name, shape, dtype: hk.get_parameter(
                name, shape, dtype, hk.initializers.RandomNormal()
            )

        self.get_parameter = get_parameter

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        r"""Evaluate the symmetric tensor product

        Args:
            x (IrrepsArray): input of shape ``(..., num_channel, irreps)``

        Returns:
            IrrepsArray: output of shape ``(..., num_channel, irreps_out)``
        """

        # TODO: normalize by taking into account the correlation, like in TensorSquare
        def fn(x: e3nn.IrrepsArray):
            # TODO: what do we do with num_channel?
            # - This operation is parallel on the feature dimension (but each feature has its own parameters)
            assert x.ndim == 2  # [num_channel, irreps_x.dim]

            out = dict()

            for order in range(max(self.orders), 0, -1):  # max(orders), ..., 1
                U = e3nn.reduced_symmetric_tensor_product_basis(
                    x.irreps, order, keep_ir=self.keep_irrep_out
                )

                # ((w3 x + w2) x + w1) x
                #  \-----------/
                #       out

                if order in self.orders:
                    for (mul, ir_out), u in zip(U.irreps, U.chunks):
                        # u: ndarray [(irreps_x.dim)^order, multiplicity, ir_out.dim]
                        u = (
                            u / u.shape[-2]
                        )  # normalize both U and the contraction with w

                        w = self.get_parameter(  # parameters initialized with a normal distribution (variance 1)
                            f"w{order}_{ir_out}",
                            (mul, x.shape[0]),
                            x.dtype,
                        )  # [multiplicity, num_channel]

                        if ir_out not in out:
                            out[ir_out] = (
                                "special",
                                jnp.einsum("...jki,kc,cj->c...i", u, w, x.array),
                            )  # [num_channel, (irreps_x.dim)^(oder-1), ir_out.dim]
                        else:
                            out[ir_out] += jnp.einsum(
                                "...ki,kc->c...i", u, w
                            )  # [num_channel, (irreps_x.dim)^order, ir_out.dim]

                # ((w3 x + w2) x + w1) x
                #  \----------------/
                #         out (in the normal case)

                for ir_out in out:
                    if isinstance(out[ir_out], tuple):
                        out[ir_out] = out[ir_out][1]
                        continue  # already done (special case optimization above)

                    out[ir_out] = jnp.einsum(
                        "c...ji,cj->c...i", out[ir_out], x.array
                    )  # [num_channel, (irreps_x.dim)^(oder-1), ir_out.dim]

                # ((w3 x + w2) x + w1) x
                #  \-------------------/
                #           out

            # out[irrep_out] : [num_channel, ir_out.dim]
            irreps_out = e3nn.Irreps(sorted(out.keys()))
            return e3nn.from_chunks(
                irreps_out,
                [out[ir][:, None, :] for (_, ir) in irreps_out],
                (x.shape[0],),
                x.dtype,
            )

        # Treat batch indices using vmap
        fn_mapped = fn
        for _ in range(x.ndim - 2):
            fn_mapped = hk.vmap(fn_mapped, split_rng=False)

        return fn_mapped(x)
