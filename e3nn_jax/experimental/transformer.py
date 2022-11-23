import haiku as hk
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn


class TensorProductMultiLayerPerceptron(hk.Module):
    irreps_in1: e3nn.Irreps
    irreps_in2: e3nn.Irreps
    irreps_out: e3nn.Irreps

    def __init__(self, tp: e3nn.FunctionalTensorProduct, list_neurons, act):
        super().__init__()

        self.tp = tp
        self.mlp = e3nn.MultiLayerPerceptron(list_neurons, act)

        self.irreps_in1 = tp.irreps_in1.simplify()
        self.irreps_in2 = tp.irreps_in2.simplify()
        self.irreps_out = tp.irreps_out.simplify()

        assert all(i.has_weight for i in self.tp.instructions)

    def __call__(self, emb, x1: e3nn.IrrepsArray, x2: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        w = self.mlp(emb)

        w = [
            jnp.einsum(
                "x...,x->...",
                hk.get_parameter(
                    (
                        f"w[{i.i_in1},{i.i_in2},{i.i_out}] "
                        f"{self.tp.irreps_in1[i.i_in1]},{self.tp.irreps_in2[i.i_in2]},{self.tp.irreps_out[i.i_out]}"
                    ),
                    shape=(w.shape[0],) + i.path_shape,
                    init=hk.initializers.RandomNormal(i.weight_std),
                )
                / w.shape[0] ** 0.5,
                w,
            )
            for i in self.tp.instructions
        ]

        x1 = x1._convert(self.irreps_in1)
        x2 = x2._convert(self.irreps_in2)

        return self.tp.left_right(w, x1, x2)._convert(self.irreps_out)


def _instructions_uvu(irreps_in1, irreps_in2, out_ir_list):
    irreps_out = []
    instructions = []
    for i1, (mul, ir_in1) in enumerate(irreps_in1):
        for i2, (_, ir_in2) in enumerate(irreps_in2):
            for ir_out in ir_in1 * ir_in2:
                if ir_out in out_ir_list:
                    k = len(irreps_out)
                    irreps_out.append((mul, ir_out))
                    instructions.append((i1, i2, k, "uvu", True))
    irreps_out = e3nn.Irreps(irreps_out)

    irreps_out, p, _ = irreps_out.sort()
    instructions = [(i_1, i_2, p[i_out], mode, has_weight) for i_1, i_2, i_out, mode, has_weight in instructions]

    return irreps_out, instructions


def _tp_mlp_uvu(
    emb, input1: e3nn.IrrepsArray, input2: e3nn.IrrepsArray, out_ir_list, *, list_neurons, act
) -> e3nn.IrrepsArray:
    irreps_out, instructions = _instructions_uvu(input1.irreps, input2.irreps, out_ir_list)
    tp = e3nn.FunctionalTensorProduct(input1.irreps, input2.irreps, irreps_out, instructions)
    return TensorProductMultiLayerPerceptron(tp, list_neurons, act)(emb, input1, input2)


def _index_max(i, x, out_dim):
    return jnp.zeros((out_dim,) + x.shape[1:]).at[i].max(x)


class Transformer(hk.Module):
    def __init__(self, irreps_node_output, list_neurons, act, num_heads=1):
        super().__init__()

        self.irreps_node_output = e3nn.Irreps(irreps_node_output)
        self.list_neurons = list_neurons
        self.act = act
        self.num_heads = num_heads

    def __call__(
        self,
        edge_src,
        edge_dst,
        edge_scalar_attr,
        edge_weight_cutoff,
        edge_attr: e3nn.IrrepsArray,
        node_feat: e3nn.IrrepsArray,
    ) -> e3nn.IrrepsArray:
        r"""Equivariant Transformer.

        Args:
            edge_src (array of int32): source index of the edges
            edge_dst (array of int32): destination index of the edges
            edge_scalar_attr (array of float): scalar attributes of the edges (typically given by ``soft_one_hot_linspace``)
            edge_weight_cutoff (array of float): cutoff weight for the edges (typically given by ``sus``)
            edge_attr (e3nn.IrrepsArray): attributes of the edges (typically given by ``spherical_harmonics``)
            node_f (e3nn.IrrepsArray): features of the nodes

        Returns:
            e3nn.IrrepsArray: output features of the nodes
        """
        edge_src_feat = jax.tree_util.tree_map(lambda x: x[edge_src], node_feat)
        edge_dst_feat = jax.tree_util.tree_map(lambda x: x[edge_dst], node_feat)

        kw = dict(list_neurons=self.list_neurons, act=self.act)
        edge_k = jax.vmap(lambda w, x, y: _tp_mlp_uvu(w, x, y, edge_dst_feat.irreps, **kw))(
            edge_scalar_attr, edge_src_feat, edge_attr
        )  # IrrepData[edge, irreps]
        edge_v = jax.vmap(lambda w, x, y: _tp_mlp_uvu(w, x, y, self.irreps_node_output, **kw))(
            edge_scalar_attr, edge_src_feat, edge_attr
        )  # IrrepData[edge, irreps]

        edge_logit = e3nn.Linear(f"{self.num_heads}x0e")(e3nn.tensor_product(edge_dst_feat, edge_k)).array  # array[edge, head]
        node_logit_max = _index_max(edge_dst, edge_logit, node_feat.shape[0])  # array[node, head]
        exp = edge_weight_cutoff[:, None] * jnp.exp(edge_logit - node_logit_max[edge_dst])  # array[edge, head]
        z = e3nn.index_add(edge_dst, exp, out_dim=node_feat.shape[0])  # array[node, head]
        z = jnp.where(z == 0.0, 1.0, z)
        alpha = exp / z[edge_dst]  # array[edge, head]

        edge_v = edge_v.factor_mul_to_last_axis(self.num_heads)  # e3nn.IrrepsArray[edge, head, irreps_out]
        edge_v = edge_v * jnp.sqrt(jax.nn.relu(alpha))[:, :, None]  # e3nn.IrrepsArray[edge, head, irreps_out]
        edge_v = edge_v.repeat_mul_by_last_axis()  # e3nn.IrrepsArray[edge, irreps_out]

        node_out = e3nn.index_add(edge_dst, edge_v, out_dim=node_feat.shape[0])  # e3nn.IrrepsArray[node, irreps_out]
        return e3nn.Linear(self.irreps_node_output)(node_out)  # e3nn.IrrepsArray[edge, head, irreps_out]
