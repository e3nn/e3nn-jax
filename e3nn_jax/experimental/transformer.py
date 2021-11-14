from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps, TensorProduct, index_add
from e3nn_jax.nn import (HFullyConnectedTensorProduct, HLinear,
                         HTensorProductMLP)


def _instructions_uvu(irreps_in1, irreps_in2, ir_out_list):
    irreps_out = []
    instructions = []
    for i1, (mul, ir_in1) in enumerate(irreps_in1):
        for i2, (_, ir_in2) in enumerate(irreps_in2):
            for ir_out in ir_in1 * ir_in2:
                if ir_out in ir_out_list:
                    k = len(irreps_out)
                    irreps_out.append((mul, ir_out))
                    instructions.append((i1, i2, k, 'uvu', True))
    irreps_out = Irreps(irreps_out)

    assert irreps_out.dim > 0, (
        f"irreps_in1={irreps_in1} "
        f"time irreps_in2={irreps_in2} "
        f"produces nothing in irreps_out={ir_out_list}"
    )

    irreps_out, p, _ = irreps_out.sort()
    instructions = [
        (i_1, i_2, p[i_out], mode, has_weight)
        for i_1, i_2, i_out, mode, has_weight in instructions
    ]

    return irreps_out, instructions


def _tensor_product_mlp_uvu(irreps_in1, irreps_in2, ir_out_list, features, phi):
    irreps_out, instructions = _instructions_uvu(irreps_in1, irreps_in2, ir_out_list)
    tp = TensorProduct(irreps_in1, irreps_in2, irreps_out, instructions)
    return HTensorProductMLP(tp, features, phi)


class Transformer(hk.Module):
    def __init__(self, irreps_node_input, irreps_node_output, irreps_edge_attr, features, phi):
        super().__init__()

        self.irreps_node_input = Irreps(irreps_node_input)
        self.irreps_edge_attr = Irreps(irreps_edge_attr)
        self.irreps_node_output = Irreps(irreps_node_output)

        self.features = features
        self.phi = phi

    def __call__(self, edge_src, edge_dst, edge_scalar_attr, edge_attr, edge_weight_cutoff, node_f):
        r"""
        Args:
            edge_src (array of int32): source index of the edges
            edge_dst (array of int32): destination index of the edges
            edge_scalar_attr (array of float): scalar attributes of the edges (typically given by ``soft_one_hot_linspace``)
            edge_attr (array of float): attributes of the edges (typically given by ``spherical_harmonics``)
            edge_weight_cutoff (float): cutoff weight for the edges (typically given by ``sus``)
            node_f (array of float): features of the nodes

        Returns:
            array of float: output features of the nodes
        """
        tp_k = _tensor_product_mlp_uvu(self.irreps_node_input, self.irreps_edge_attr, self.irreps_node_input, self.features, self.phi)
        edge_k = jax.vmap(partial(tp_k, output_list=True))(edge_scalar_attr, node_f[edge_src], edge_attr)

        dot = HFullyConnectedTensorProduct(self.irreps_node_input, tp_k.irreps_out, "0e")
        exp = edge_weight_cutoff[:, None] * jnp.exp(jax.vmap(dot)(node_f[edge_dst], edge_k))
        z = index_add(edge_dst, exp, len(node_f))
        z = jnp.where(z == 0.0, 1.0, z)
        alpha = exp / z[edge_dst]

        tp_v = _tensor_product_mlp_uvu(self.irreps_node_input, self.irreps_edge_attr, self.irreps_node_output, self.features, self.phi)
        edge_v = jax.vmap(tp_v)(edge_scalar_attr, node_f[edge_src], edge_attr)

        node_out = index_add(edge_dst, jnp.sqrt(jax.nn.relu(alpha)) * edge_v, len(node_f))
        return jax.vmap(HLinear(tp_v.irreps_out, self.irreps_node_output))(node_out)
