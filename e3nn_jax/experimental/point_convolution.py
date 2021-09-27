import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps, TensorProduct, index_add
from e3nn_jax.nn import HMLP, HFullyConnectedTensorProduct, HLinear
from functools import partial


class Convolution(hk.Module):
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
    def __init__(self, irreps_node_input, irreps_node_attr, irreps_edge_attr, irreps_node_output, fc_neurons, num_neighbors, mixing_angle=jnp.pi / 8.0):
        super().__init__()

        self.irreps_node_input = Irreps(irreps_node_input)
        self.irreps_node_attr = Irreps(irreps_node_attr)
        self.irreps_edge_attr = Irreps(irreps_edge_attr)
        self.irreps_node_output = Irreps(irreps_node_output)
        self.fc_neurons = fc_neurons
        self.num_neighbors = num_neighbors
        self.mixing_angle = mixing_angle

    def __call__(self, node_input, edge_src, edge_dst, edge_attr, node_attr=None, edge_scalar_attr=None):
        irreps_node_input = Irreps(self.irreps_node_input)
        irreps_node_attr = Irreps(self.irreps_node_attr)
        irreps_edge_attr = Irreps(self.irreps_edge_attr)
        irreps_node_output = Irreps(self.irreps_node_output)

        ######################################################################################

        if irreps_node_attr is not None and node_attr is not None:
            tmp = jax.vmap(HFullyConnectedTensorProduct(
                irreps_node_input,
                irreps_node_attr,
                irreps_node_input + irreps_node_output
            ), (0, 0, None), 0)(node_input, node_attr, output_list=True)
        else:
            tmp = jax.vmap(partial(HLinear(
                irreps_node_input,
                irreps_node_input + irreps_node_output
            ), output_list=True))(node_input)

        node_features, node_self_out = tmp[:len(irreps_node_input)], tmp[len(irreps_node_input):]

        edge_features = jax.tree_map(lambda x: x[edge_src], node_features)

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

        tp = TensorProduct(
            irreps_node_input,
            irreps_edge_attr,
            irreps_mid,
            instructions,
        )
        irreps_mid = irreps_mid.simplify()

        if self.fc_neurons:
            weight = HMLP(
                self.fc_neurons,
                jax.nn.gelu
            )(edge_scalar_attr)
            weight = [
                jnp.einsum(
                    "x...,ex->e...",
                    hk.get_parameter(f'weight {ins.i_in1} x {ins.i_in2} -> {ins.i_out}', shape=(weight.shape[1],) + ins.path_shape, init=hk.initializers.RandomNormal()),
                    weight
                )
                for ins in tp.instructions
            ]
            edge_features = jax.vmap(tp.left_right, (0, 0, 0), 0)(weight, edge_features, edge_attr, output_list=False)
        else:
            weight = [
                hk.get_parameter(f'weight {ins.i_in1} x {ins.i_in2} -> {ins.i_out}', shape=ins.path_shape, init=hk.initializers.RandomNormal())
                for ins in tp.instructions
            ]
            edge_features = jax.vmap(partial(tp.left_right, output_list=False), (None, 0, 0), 0)(weight, edge_features, edge_attr)

        # TODO irreps_mid = irreps_mid.simplify()

        ######################################################################################

        node_features = index_add(edge_dst, edge_features, out_dim=node_input[0].shape[0])

        node_features = node_features / self.num_neighbors**0.5

        ######################################################################################

        if irreps_node_attr is not None and node_attr is not None:
            node_conv_out = jax.vmap(HFullyConnectedTensorProduct(
                irreps_mid,
                irreps_node_attr,
                irreps_node_output
            ))(node_features, node_attr, output_list=True)
        else:
            node_conv_out = jax.vmap(partial(HLinear(
                irreps_mid,
                irreps_node_output
            ), output_list=True))(node_features)

        ######################################################################################

        with jax.core.eval_context():
            c = jnp.cos(self.mixing_angle)
            s = jnp.sin(self.mixing_angle)
        return jax.tree_multimap(lambda x, y: c * x + s * y, node_self_out, node_conv_out)
