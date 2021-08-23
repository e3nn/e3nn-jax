import flax
import jax
import jax.numpy as jnp
from e3nn_jax import FullyConnectedTensorProduct, Irreps, TensorProduct
from e3nn_jax.flax import MLP


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
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        num_neighbors
    ):
        self.irreps_node_input = Irreps(irreps_node_input)
        self.irreps_node_attr = Irreps(irreps_node_attr)
        self.irreps_edge_attr = Irreps(irreps_edge_attr)
        self.irreps_node_output = Irreps(irreps_node_output)
        self.fc_neurons = fc_neurons
        self.num_neighbors = num_neighbors

    @flax.linen.compact
    def __call__(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalar_attr):

        self_con = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output)

        lin1 = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out.is_scalar():
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = Irreps(irreps_mid)

        assert irreps_mid.dim > 0, f"irreps_node_input={self.irreps_node_input} time irreps_edge_attr={self.irreps_edge_attr} produces nothing in irreps_node_output={self.irreps_node_output}"

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
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
        )

        lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_node_output)

        # inspired by https://arxiv.org/pdf/2002.10444.pdf
        alpha = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")
        # alpha.weight.zero_()
        assert alpha.output_mask[0] == 1.0, f"irreps_mid={irreps_mid} and irreps_node_attr={self.irreps_node_attr} are not able to generate scalars"

        weight = fc(edge_scalar_attr)
        # es -MLP> ef
        # ef * f... -> e...

        node_features = lin1(node_input, node_attr)

        edge_features = tp(node_features[edge_src], edge_attr, weight)
        node_features = jax.ops.index_add(jnp.zeros((node_input.shape[0], edge_features.shape[1])), edge_dst, edge_features)
        node_features = node_features / self.num_neighbors**0.5

        node_conv_out = lin2(node_features, node_attr)
        alpha = alpha(node_features, node_attr)

        m = self_con.output_mask
        alpha = (1 - m) + alpha * m
        return self_con(node_input, node_attr) + alpha * node_conv_out
