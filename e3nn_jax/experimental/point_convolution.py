from typing import Callable, Sequence

import flax
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps, TensorProduct
from e3nn_jax.flax import MLP, FlaxFullyConnectedTensorProduct


class Convolution(flax.linen.Module):
    r"""equivariant convolution

    Parameters
    ----------
    irreps_node_input : `e3nn.o3.Irreps`
        representation of the input node features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the node attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    irreps_node_output : `e3nn.o3.Irreps` or None
        representation of the output node features

    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer

    num_neighbors : float
        typical number of nodes convolved over
    """
    irreps_node_input: Irreps
    irreps_node_attr: Irreps
    irreps_edge_attr: Irreps
    irreps_node_output: Irreps
    fc_neurons: Sequence[int]
    num_neighbors: float
    mixing_angle: float = 0.2
    weight_init: Callable = jax.random.normal

    @flax.linen.compact
    def __call__(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalar_attr):
        irreps_node_input = Irreps(self.irreps_node_input)
        irreps_node_attr = Irreps(self.irreps_node_attr)
        irreps_edge_attr = Irreps(self.irreps_edge_attr)
        irreps_node_output = Irreps(self.irreps_node_output)

        ######################################################################################

        tmp = FlaxFullyConnectedTensorProduct(
            irreps_node_input,
            irreps_node_attr,
            irreps_node_input + irreps_node_output
        )(node_input, node_attr)
        node_features, node_self_out = tmp[:, :irreps_node_input.dim], tmp[:, irreps_node_input.dim:]

        ######################################################################################

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(irreps_node_input):
            for j, (_, ir_edge) in enumerate(irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in irreps_node_output or ir_out.is_scalar():
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = Irreps(irreps_mid)

        assert irreps_mid.dim > 0, f"irreps_node_input={irreps_node_input} time irreps_edge_attr={irreps_edge_attr} produces nothing in irreps_node_output={irreps_node_output}"

        irreps_mid, p, _ = irreps_mid.sort()
        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        fc = MLP(
            self.fc_neurons,
            jax.nn.gelu
        )
        tp = TensorProduct(
            irreps_node_input,
            irreps_edge_attr,
            irreps_mid,
            instructions,
        )
        irreps_mid = irreps_mid.simplify()

        weight = fc(edge_scalar_attr)
        weight = [
            jnp.einsum(
                "x...,ex->e...",
                self.param(f'weight {ins.i_in1} x {ins.i_in2} -> {ins.i_out}', self.weight_init, (weight.shape[1],) + ins.path_shape),
                weight
            )
            for ins in tp.instructions
        ]
        edge_features = jax.vmap(tp.left_right, (0, 0, 0), 0)(weight, node_features[edge_src], edge_attr)

        ######################################################################################

        node_features = jax.ops.index_add(jnp.zeros((node_input.shape[0], edge_features.shape[1])), edge_dst, edge_features)
        node_features = node_features / self.num_neighbors**0.5

        ######################################################################################

        node_conv_out = FlaxFullyConnectedTensorProduct(
            irreps_mid,
            irreps_node_attr,
            irreps_node_output
        )(node_features, node_attr)

        ######################################################################################

        with jax.core.eval_context():
            c = jnp.cos(self.mixing_angle)
            s = jnp.sin(self.mixing_angle)
        return c * node_self_out + s * node_conv_out
