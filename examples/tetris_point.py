import time

import flax
import jax
import jax.numpy as jnp
import optax
from e3nn_jax import Gate, Irreps, index_add, radius_graph, spherical_harmonics
from e3nn_jax.experimental.point_convolution import Convolution
from flax.training import train_state


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
        gate = Gate('32x0e + 32x0o', [jax.nn.gelu, jnp.tanh], '8x0e + 8x0e', 2 * [jax.nn.sigmoid], '8x1e + 8x1o')
        g = jax.vmap(gate)

        kw = dict(
            irreps_node_attr=Irreps('0e'),
            irreps_edge_attr=Irreps('0e + 1o + 2e'),
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

        for _ in range(3):
            x = g(
                Convolution(
                    irreps_node_input=gate.irreps_out,
                    irreps_node_output=gate.irreps_in,
                    **kw
                )(x, edge_src, edge_dst, edge_attr)
            )

        x = Convolution(
            irreps_node_input=gate.irreps_out,
            irreps_node_output=Irreps('0o + 6x0e'),
            **kw
        )(x, edge_src, edge_dst, edge_attr)

        return x


def apply_model(state, node_input, edge_src, edge_dst, edge_attr, labels, batch):
    """Computes gradients, loss and accuracy for a single batch."""
    def loss_fn(params):
        pred = Model().apply({'params': params}, node_input, edge_src, edge_dst, edge_attr)
        pred = jnp.concatenate([x.reshape(x.shape[0], -1) for x in pred], axis=-1)
        pred = index_add(batch, pred, 8)
        loss = jnp.mean((pred - labels)**2)
        return loss, pred

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.all(jnp.round(pred) == labels, axis=1))
    return grads, loss, accuracy, pred


def update_model(state, grads):
    return state.apply_gradients(grads=grads)


@jax.jit
def make_steps(n, state, node_input, edge_src, edge_dst, edge_attr, labels, batch):
    def f(i, state):
        grads, loss, accuracy, pred = apply_model(state, node_input, edge_src, edge_dst, edge_attr, labels, batch)
        return update_model(state, grads)

    state = jax.lax.fori_loop(0, n - 1, f, state)
    grads, loss, accuracy, pred = apply_model(state, node_input, edge_src, edge_dst, edge_attr, labels, batch)
    state = update_model(state, grads)
    return state, loss, accuracy, pred


def main():
    pos, labels, batch = tetris()
    edge_src, edge_dst = radius_graph(pos, 1.1, batch)
    edge_attr = spherical_harmonics("0e + 1o + 2e", pos[edge_dst] - pos[edge_src], True, normalization='component')
    edge_attr = [edge_attr[:, 1].reshape(-1, 1, 1), edge_attr[:, 1:4].reshape(-1, 1, 3), edge_attr[:, 4:9].reshape(-1, 1, 5)]
    node_input = [jnp.ones((pos.shape[0], 1, 1))]

    learning_rate = 0.1
    momentum = 0.9

    rng = jax.random.PRNGKey(3)

    model = Model()
    params = model.init(rng, node_input, edge_src, edge_dst, edge_attr)

    tx = optax.sgd(learning_rate, momentum)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        tx=tx
    )

    # compile jit
    wall = time.perf_counter()
    print("compiling...")
    _, _, _, pred = make_steps(50, state, node_input, edge_src, edge_dst, edge_attr, labels, batch)
    print(pred.round(2))

    print(f"It took {time.perf_counter() - wall:.1f}s to compile jit.")

    wall = time.perf_counter()
    it = 0
    for _ in range(2000):
        state, loss, accuracy, pred = make_steps(100, state, node_input, edge_src, edge_dst, edge_attr, labels, batch)
        it += 100
        if accuracy == 1:
            break

    total = time.perf_counter() - wall
    print(f"100% accuracy has been reach in {total:.1f}s after {it} iterations ({1000 * total/it:.1f}ms/it).")

    print(f"accuracy = {100 * accuracy:.0f}%")

    print(pred.round(2))


if __name__ == '__main__':
    main()
