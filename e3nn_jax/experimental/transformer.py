# TODO rewrite with better integration of IrrepsData
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps, IrrepsData, FunctionalTensorProduct, index_add, Linear, FullyConnectedTensorProduct, MultiLayerPerceptron


class TensorProductMultiLayerPerceptron(hk.Module):
    irreps_in1: Irreps
    irreps_in2: Irreps
    irreps_out: Irreps

    def __init__(self, tp: FunctionalTensorProduct, list_neurons, phi):
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
                    f'w[{i.i_in1},{i.i_in2},{i.i_out}] {self.tp.irreps_in1[i.i_in1]},{self.tp.irreps_in2[i.i_in2]},{self.tp.irreps_out[i.i_out]}',
                    shape=(w.shape[0],) + i.path_shape,
                    init=hk.initializers.RandomNormal()
                ) / w.shape[0]**0.5,
                w
            )
            for i in self.tp.instructions
        ]
        return self.tp.left_right(w, x1, x2)


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


def _tensor_product_mlp_uvu(irreps_in1, irreps_in2, ir_out_list, list_neurons, phi):
    irreps_out, instructions = _instructions_uvu(irreps_in1, irreps_in2, ir_out_list)
    tp = FunctionalTensorProduct(irreps_in1, irreps_in2, irreps_out, instructions)
    return TensorProductMultiLayerPerceptron(tp, list_neurons, phi)


class Transformer(hk.Module):
    def __init__(self, irreps_node_input, irreps_node_output, irreps_edge_attr, list_neurons, phi, num_heads=1):
        super().__init__()

        self.irreps_node_input = Irreps(irreps_node_input)
        self.irreps_edge_attr = Irreps(irreps_edge_attr)
        self.irreps_node_output = Irreps(irreps_node_output)

        self.list_neurons = list_neurons
        self.phi = phi
        self.num_heads = num_heads

        assert all(mul % num_heads == 0 for mul, _ in self.irreps_node_input), "num_heads must divide all irreps_node_input multiplicities"

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
        node_f = IrrepsData.new(self.irreps_node_input, node_f).contiguous

        tp_k = _tensor_product_mlp_uvu(self.irreps_node_input, self.irreps_edge_attr, self.irreps_node_input, self.list_neurons, self.phi)
        edge_k = jax.vmap(tp_k)(edge_scalar_attr, node_f[edge_src], edge_attr)

        dot = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_node_input,
            irreps_in2=tp_k.irreps_out,
            irreps_out=f"{self.num_heads}x 0e"
        )
        exp = edge_weight_cutoff[:, None] * jnp.exp(jax.vmap(dot)(node_f[edge_dst], edge_k).contiguous)  # array[edge, head]
        z = index_add(edge_dst, exp, len(node_f))  # array[node, head]
        z = jnp.where(z == 0.0, 1.0, z)
        alpha = exp / z[edge_dst]  # array[edge, head]

        tp_v = _tensor_product_mlp_uvu(self.irreps_node_input, self.irreps_edge_attr, self.irreps_node_output, self.list_neurons, self.phi)
        edge_v = jax.vmap(tp_v)(edge_scalar_attr, node_f[edge_src], edge_attr)  # list of array[edge, mul, ir]
        edge_v = [jnp.sqrt(jax.nn.relu(alpha))[:, :, None, None] * v.reshape(v.shape[0], self.num_heads, v.shape[1] // self.num_heads, v.shape[2]) for v in edge_v.list]
        edge_v = jnp.concatenate([v.reshape(v.shape[0], -1) for v in edge_v], axis=-1)  # array[edge, irreps]

        node_out = index_add(edge_dst, edge_v, len(node_f))
        lin = Linear(irreps_in=tp_v.irreps_out, irreps_out=self.irreps_node_output)
        return jax.vmap(lin)(node_out)
