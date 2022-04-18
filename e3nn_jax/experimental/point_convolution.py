from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import (
    Irreps,
    IrrepsData,
    FunctionalTensorProduct,
    index_add,
    Linear,
    FullyConnectedTensorProduct,
    MultiLayerPerceptron,
)


class Convolution(hk.Module):
    r"""Equivariant Point Convolution

    Args:
        irreps_node_output : `e3nn.o3.Irreps`
            representation of the output node features

        fc_neurons : list of int
            number of neurons per layers in the fully connected network
            first layer and hidden layers but not the output layer

        num_neighbors : float
            typical number of nodes convolved over
    """

    def __init__(self, irreps_node_output, fc_neurons, num_neighbors, mixing_angle=jnp.pi / 8.0):
        super().__init__()

        self.irreps_node_output = Irreps(irreps_node_output)
        self.fc_neurons = fc_neurons
        self.num_neighbors = num_neighbors
        self.mixing_angle = mixing_angle

    @partial(jax.profiler.annotate_function, name="convolution")
    def __call__(
        self, node_input: IrrepsData, edge_src, edge_dst, edge_attr: IrrepsData, node_attr=None, edge_scalar_attr=None
    ) -> IrrepsData:
        assert isinstance(node_input, IrrepsData)
        assert isinstance(edge_attr, IrrepsData)

        if node_attr is not None:
            tmp = jax.vmap(FullyConnectedTensorProduct(node_input.irreps + self.irreps_node_output))(node_input, node_attr)
        else:
            tmp = jax.vmap(Linear(node_input.irreps + self.irreps_node_output))(node_input)

        node_features, node_self_out = tmp.split([len(node_input.irreps)])

        edge_features = jax.tree_map(lambda x: x[edge_src], node_features)
        del node_features

        ######################################################################################
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(node_input.irreps):
            for j, (_, ir_edge) in enumerate(edge_attr.irreps):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out.is_scalar():
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = Irreps(irreps_mid)

        assert irreps_mid.dim > 0, (
            f"irreps_node_input={node_input.irreps} "
            f"time irreps_edge_attr={edge_attr.irreps} "
            f"produces nothing in irreps_node_output={self.irreps_node_output}"
        )

        irreps_mid, p, _ = irreps_mid.sort()
        instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

        tp = FunctionalTensorProduct(
            node_input.irreps,
            edge_attr.irreps,
            irreps_mid,
            instructions,
        )

        if self.fc_neurons:
            weight = MultiLayerPerceptron(self.fc_neurons, jax.nn.gelu)(edge_scalar_attr)

            weight = [
                jnp.einsum(
                    "x...,ex->e...",
                    hk.get_parameter(
                        f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] {tp.irreps_in1[ins.i_in1]},{tp.irreps_in2[ins.i_in2]},{tp.irreps_out[ins.i_out]}",
                        shape=(weight.shape[1],) + ins.path_shape,
                        init=hk.initializers.RandomNormal(),
                    )
                    / weight.shape[1] ** 0.5,
                    weight,
                )
                for ins in tp.instructions
            ]

            edge_features: IrrepsData = jax.vmap(tp.left_right, (0, 0, 0), 0)(weight, edge_features, edge_attr)
        else:
            weight = [
                hk.get_parameter(
                    f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] {tp.irreps_in1[ins.i_in1]},{tp.irreps_in2[ins.i_in2]},{tp.irreps_out[ins.i_out]}",
                    shape=ins.path_shape,
                    init=hk.initializers.RandomNormal(),
                )
                for ins in tp.instructions
            ]
            edge_features: IrrepsData = jax.vmap(tp.left_right, (None, 0, 0), 0)(weight, edge_features, edge_attr)

        edge_features = edge_features.remove_nones().simplify()
        ######################################################################################

        node_features = jax.tree_map(lambda x: index_add(edge_dst, x, out_dim=node_input.shape[0]), edge_features)
        node_features = node_features / self.num_neighbors ** 0.5

        ######################################################################################

        if node_attr is not None:
            node_conv_out = jax.vmap(FullyConnectedTensorProduct(self.irreps_node_output))(node_features, node_attr)
        else:
            node_conv_out = jax.vmap(Linear(self.irreps_node_output))(node_features)

        ######################################################################################

        with jax.ensure_compile_time_eval():
            c = jnp.cos(self.mixing_angle)
            s = jnp.sin(self.mixing_angle)

        return c * node_self_out + s * node_conv_out
