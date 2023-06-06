from typing import Callable, List

import haiku as hk
import jax
import jax.numpy as jnp

import e3nn_jax as e3nn


def _index_max(i: jnp.ndarray, x: jnp.ndarray, out_dim: int) -> jnp.ndarray:
    return jnp.zeros((out_dim,) + x.shape[1:], x.dtype).at[i].max(x)


class Transformer(hk.Module):
    def __init__(
        self,
        irreps_node_output: e3nn.Irreps,
        list_neurons: List[int],
        act: Callable[[jnp.ndarray], jnp.ndarray],
        num_heads: int = 1,
    ):
        super().__init__()

        self.irreps_node_output = e3nn.Irreps(irreps_node_output)
        self.list_neurons = list_neurons
        self.act = act
        self.num_heads = num_heads

    def __call__(
        self,
        edge_src: jnp.ndarray,  # [E] dtype=int32
        edge_dst: jnp.ndarray,  # [E] dtype=int32
        edge_weight_cutoff: jnp.ndarray,  # [E] dtype=float
        edge_attr: e3nn.IrrepsArray,  # [E, D] dtype=float
        node_feat: e3nn.IrrepsArray,  # [N, D] dtype=float
    ) -> e3nn.IrrepsArray:
        r"""Equivariant Transformer.

        Args:
            edge_src (array of int32): source index of the edges
            edge_dst (array of int32): destination index of the edges
            edge_weight_cutoff (array of float): cutoff weight for the edges (typically given by ``soft_envelope``)
            edge_attr (e3nn.IrrepsArray): attributes of the edges (typically given by ``spherical_harmonics``)
            node_f (e3nn.IrrepsArray): features of the nodes

        Returns:
            e3nn.IrrepsArray: output features of the nodes
        """

        def f(x, y, filter_ir_out=None, name=None):
            out1 = (
                e3nn.concatenate([x, e3nn.tensor_product(x, y.filter(drop="0e"))])
                .regroup()
                .filter(keep=filter_ir_out)
            )
            out2 = e3nn.haiku.MultiLayerPerceptron(
                self.list_neurons + [out1.irreps.num_irreps],
                self.act,
                output_activation=False,
                name=name,
            )(y.filter(keep="0e"))
            return out1 * out2

        edge_key = f(node_feat[edge_src], edge_attr, node_feat.irreps, name="mlp_key")
        edge_logit = e3nn.haiku.Linear(f"{self.num_heads}x0e", name="linear_logit")(
            e3nn.tensor_product(node_feat[edge_dst], edge_key, filter_ir_out="0e")
        ).array  # [E, H]
        node_logit_max = _index_max(edge_dst, edge_logit, node_feat.shape[0])  # [N, H]
        exp = edge_weight_cutoff[:, None] * jnp.exp(
            edge_logit - node_logit_max[edge_dst]
        )  # [E, H]
        z = e3nn.scatter_sum(
            exp, dst=edge_dst, output_size=node_feat.shape[0]
        )  # [N, H]
        z = jnp.where(z == 0.0, 1.0, z)
        alpha = exp / z[edge_dst]  # [E, H]

        edge_v = f(
            node_feat[edge_src], edge_attr, self.irreps_node_output, "mlp_val"
        )  # [E, D]
        edge_v = edge_v.mul_to_axis(self.num_heads)  # [E, H, D]
        edge_v = edge_v * jnp.sqrt(jax.nn.relu(alpha))[:, :, None]  # [E, H, D]
        edge_v = edge_v.axis_to_mul()  # [E, D]

        node_out = e3nn.scatter_sum(
            edge_v, dst=edge_dst, output_size=node_feat.shape[0]
        )  # [N, D]
        return e3nn.haiku.Linear(self.irreps_node_output, name="linear_out")(
            node_out
        )  # [N, D]
