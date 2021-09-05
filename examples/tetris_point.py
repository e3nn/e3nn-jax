import flax
import jax
import jax.numpy as jnp
import optax
from e3nn_jax import Gate, Irreps, index_add, radius_graph, spherical_harmonics
from e3nn_jax.experimental.point_convolution import Convolution
from flax.training import train_state
from tqdm.auto import tqdm


def tetris():
    pos = jnp.array([
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],  # zigzag
    ], dtype=jnp.float32)

    # Since chiral shapes are the mirror of one another we need an *odd* scalar to distinguish them
    labels = jnp.array([
        [+1, 0, 0, 0, 0, 0, 0],  # chiral_shape_1
        [-1, 0, 0, 0, 0, 0, 0],  # chiral_shape_2
        [0, 1, 0, 0, 0, 0, 0],  # square
        [0, 0, 1, 0, 0, 0, 0],  # line
        [0, 0, 0, 1, 0, 0, 0],  # corner
        [0, 0, 0, 0, 1, 0, 0],  # L
        [0, 0, 0, 0, 0, 1, 0],  # T
        [0, 0, 0, 0, 0, 0, 1],  # zigzag
    ], dtype=jnp.float32)

    pos = pos.reshape((8 * 4, 3))
    batch = jnp.arange(8 * 4) // 4

    return pos, labels, batch


class Model(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x, edge_src, edge_dst, edge_attr):
        gate = Gate('10x0e + 10x0o', [jax.nn.gelu, jnp.tanh], '10x0e', [jax.nn.sigmoid], '5x1e + 5x1o')
        g = jax.vmap(gate)

        kw = dict(
            irreps_node_attr=Irreps('0e'),
            irreps_edge_attr=Irreps('0e + 1o'),
            fc_neurons=None,
            num_neighbors=1.5,
        )

        x = g(
            Convolution(
                irreps_node_input=Irreps('0e'),
                irreps_node_output=gate.irreps_in,
                **kw
            )(x, edge_src, edge_dst, edge_attr)
        )
        print(x.shape)

        for _ in range(3):
            x = g(
                Convolution(
                    irreps_node_input=gate.irreps_out,
                    irreps_node_output=gate.irreps_in,
                    **kw
                )(x, edge_src, edge_dst, edge_attr)
            )
            print(x.shape)

        x = Convolution(
            irreps_node_input=gate.irreps_out,
            irreps_node_output=Irreps('0o + 6x0e'),
            **kw
        )(x, edge_src, edge_dst, edge_attr)
        print(x.shape)

        return x


@jax.jit
def apply_model(state, x, edge_src, edge_dst, edge_attr, labels, batch):
    """Computes gradients, loss and accuracy for a single batch."""
    def loss_fn(params):
        pred = Model().apply({'params': params}, x, edge_src, edge_dst, edge_attr)
        pred = index_add(batch, pred, 8)
        loss = jnp.mean((pred - labels)**2)
        return loss, pred

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.all(jnp.round(pred) == labels, axis=1))
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def main():
    pos, labels, batch = tetris()
    edge_src, edge_dst = radius_graph(pos, 1.1, batch)
    edge_attr = spherical_harmonics("0e + 1o", pos[edge_dst] - pos[edge_src], True, normalization='component')

    learning_rate = 0.1
    momentum = 0.9

    rng = jax.random.PRNGKey(3)

    model = Model()
    params = model.init(rng, jnp.ones((pos.shape[0], 1)), edge_src, edge_dst, edge_attr)

    tx = optax.sgd(learning_rate, momentum)
    st = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        tx=tx
    )

    for _ in tqdm(range(1000)):
        grads, loss, accuracy = apply_model(st, jnp.ones((pos.shape[0], 1)), edge_src, edge_dst, edge_attr, labels, batch)
        st = update_model(st, grads)
        # print(f"loss = {loss:.3f}")

    print(f"accuracy = {100 * accuracy:.0f}%")


if __name__ == '__main__':
    main()
