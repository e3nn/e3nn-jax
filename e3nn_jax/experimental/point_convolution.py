from functools import partial
from typing import List

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp


class Convolution(hk.Module):
    r"""Equivariant Point Convolution.

    Args:
        irreps_node_output : `Irreps`
            representation of the output node features

        fc_neurons : list of int
            number of neurons per layers in the fully connected network
            first layer and hidden layers but not the output layer

        num_neighbors : float
            typical number of nodes convolved over

        mixing : float
            mixing between self interaction and neighbors interaction,
            0 for only self interaction, 1 for only neighbors interaction
    """

    def __init__(
        self,
        irreps_node_output: e3nn.Irreps,
        fc_neurons: List[int],
        num_neighbors: float,
        *,
        mixing: float = 0.15,
        mixing_angle: float = None,
    ):
        super().__init__()

        self.irreps_node_output = e3nn.Irreps(irreps_node_output)
        self.fc_neurons = fc_neurons
        self.num_neighbors = num_neighbors

        if mixing_angle is not None:
            self.mixing = jnp.sin(mixing_angle) ** 2
        else:
            self.mixing = mixing

    @partial(jax.profiler.annotate_function, name="convolution")
    def __call__(
        self,
        node_input: e3nn.IrrepsArray,
        edge_src: jnp.ndarray,
        edge_dst: jnp.ndarray,
        edge_attr: e3nn.IrrepsArray,
        node_attr: e3nn.IrrepsArray = None,
        edge_scalar_attr: jnp.ndarray = None,
    ) -> e3nn.IrrepsArray:
        assert isinstance(node_input, e3nn.IrrepsArray)
        assert node_input.ndim == 2  # [num_nodes, irreps]

        assert isinstance(edge_attr, e3nn.IrrepsArray)
        assert edge_attr.ndim == 2  # [num_edges, irreps]

        if node_attr is None:
            node_attr = e3nn.IrrepsArray.ones("0e", node_input.shape[:-1])

        node = e3nn.Linear(node_input.irreps + self.irreps_node_output)(e3nn.tensor_product(node_input, node_attr))
        # node_features, node_self_out = node.split([node_input.irreps, self.irreps_node_output])
        node_features, node_self_out = node[:, : node_input.irreps.dim], node[:, node_input.irreps.dim :]

        edge_features = node_features[edge_src]
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
        irreps_mid = e3nn.Irreps(irreps_mid)

        assert irreps_mid.dim > 0, (
            f"irreps_node_input={node_input.irreps} "
            f"time irreps_edge_attr={edge_attr.irreps} "
            f"produces nothing in irreps_node_output={self.irreps_node_output}"
        )

        irreps_mid, p, _ = irreps_mid.sort()
        instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

        tp = e3nn.FunctionalTensorProduct(
            node_input.irreps,
            edge_attr.irreps,
            irreps_mid,
            instructions,
        )

        if self.fc_neurons:
            weight = e3nn.MultiLayerPerceptron(self.fc_neurons, jax.nn.gelu)(edge_scalar_attr)

            weight = [
                jnp.einsum(
                    "x...,ex->e...",
                    hk.get_parameter(
                        (
                            f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] "
                            f"{tp.irreps_in1[ins.i_in1]},{tp.irreps_in2[ins.i_in2]},{tp.irreps_out[ins.i_out]}"
                        ),
                        shape=(weight.shape[1],) + ins.path_shape,
                        init=hk.initializers.RandomNormal(ins.weight_std),
                    )
                    / weight.shape[1] ** 0.5,
                    weight,
                )
                for ins in tp.instructions
            ]

            edge_features: e3nn.IrrepsArray = jax.vmap(tp.left_right, (0, 0, 0), 0)(weight, edge_features, edge_attr)
        else:
            weight = [
                hk.get_parameter(
                    (
                        f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] "
                        f"{tp.irreps_in1[ins.i_in1]},{tp.irreps_in2[ins.i_in2]},{tp.irreps_out[ins.i_out]}"
                    ),
                    shape=ins.path_shape,
                    init=hk.initializers.RandomNormal(ins.weight_std),
                )
                for ins in tp.instructions
            ]
            edge_features: e3nn.IrrepsArray = jax.vmap(tp.left_right, (None, 0, 0), 0)(weight, edge_features, edge_attr)

        edge_features = edge_features.remove_nones().simplify()

        ######################################################################################

        node_features = e3nn.index_add(edge_dst, edge_features, out_dim=node_input.shape[0])
        node_features = node_features / self.num_neighbors**0.5

        ######################################################################################

        node_conv_out = e3nn.Linear(self.irreps_node_output)(e3nn.tensor_product(node_features, node_attr))

        ######################################################################################

        return jnp.sqrt(1.0 - self.mixing) * node_self_out + jnp.sqrt(self.mixing) * node_conv_out
