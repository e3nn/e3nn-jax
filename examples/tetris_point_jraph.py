import time

import flax
import jax
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import optax

import e3nn_jax as e3nn


def tetris() -> jraph.GraphsTuple:
    pos = jnp.array(
        [
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
            [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
            [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],  # zigzag
        ],
        dtype=jnp.float32,
    )

    # Since chiral shapes are the mirror of one another we need an *odd* scalar to distinguish them
    labels = jnp.array(
        [
            [+1, 1, 0, 0, 0, 0, 0, 0],  # chiral_shape_1
            [-1, 1, 0, 0, 0, 0, 0, 0],  # chiral_shape_2
            [0, 0, 1, 0, 0, 0, 0, 0],  # square
            [0, 0, 0, 1, 0, 0, 0, 0],  # line
            [0, 0, 0, 0, 1, 0, 0, 0],  # corner
            [0, 0, 0, 0, 0, 1, 0, 0],  # L
            [0, 0, 0, 0, 0, 0, 1, 0],  # T
            [0, 0, 0, 0, 0, 0, 0, 1],  # zigzag
        ],
        dtype=jnp.float32,
    )

    graphs = []

    for p, l in zip(pos, labels):
        senders, receivers = e3nn.radius_graph(p, 1.1)

        graphs += [
            jraph.GraphsTuple(
                nodes=p.reshape((4, 3)),  # [num_nodes, 3]
                edges=None,
                globals=l[None],  # [num_graphs, num_classes]
                senders=senders,  # [num_edges]
                receivers=receivers,  # [num_edges]
                n_node=jnp.array([4]),  # [num_graphs]
                n_edge=jnp.array([len(senders)]),  # [num_graphs]
            )
        ]

    return jraph.batch(graphs)


class Layer(flax.linen.Module):
    target_irreps: e3nn.Irreps
    avg_num_neighbors: float
    sh_lmax: int = 3

    @flax.linen.compact
    def __call__(self, graphs, positions):
        target_irreps = e3nn.Irreps(self.target_irreps)
        vectors = positions[graphs.receivers] - positions[graphs.senders]  # [n_edges, 1e or 1o]
        sh = e3nn.spherical_harmonics(list(range(1, self.sh_lmax + 1)), vectors, True)

        def update_edge_fn(edge_features, sender_features, receiver_features, globals):
            return e3nn.concatenate([sender_features, e3nn.tensor_product(sender_features, sh)]).regroup()

        def update_node_fn(node_features, sender_features, receiver_features, globals):
            shortcut = e3nn.flax.Linear(target_irreps, name="shortcut")(node_features)

            node_feats = receiver_features / jnp.sqrt(self.avg_num_neighbors)
            node_feats = e3nn.flax.Linear(target_irreps, name="linear_pre")(node_feats)
            node_feats = e3nn.scalar_activation(node_feats, even_act=jax.nn.gelu, odd_act=jax.nn.tanh)
            node_feats = e3nn.flax.Linear(target_irreps, name="linear_post")(node_feats)
            return shortcut + node_feats

        return jraph.GraphNetwork(update_edge_fn, update_node_fn)(graphs)


class Model(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, graphs):
        positions = e3nn.IrrepsArray("1o", graphs.nodes)

        graphs = graphs._replace(nodes=jnp.ones((len(positions), 1)))

        ann = 2.0
        graphs = Layer("32x0e + 32x0o + 8x1e + 8x1o + 8x2e + 8x2o", ann)(graphs, positions)
        graphs = Layer("32x0e + 32x0o + 8x1e + 8x1o + 8x2e + 8x2o", ann)(graphs, positions)
        graphs = Layer("0o + 7x0e", ann)(graphs, positions)
        return graphs.nodes


def train(seeds=20, steps=200, plot=True):
    model = Model()
    opt = optax.adam(learning_rate=0.01)

    def loss_pred(params, graphs):
        pred = model.apply(params, graphs)
        pred = e3nn.scatter_sum(pred, nel=graphs.n_node)  # [num_graphs, 1 + 7]
        assert pred.irreps == "0o + 7x0e", pred.irreps
        assert pred.shape == (len(graphs.n_node), 8), pred.shape
        pred = pred.array
        labels = graphs.globals  # [num_graphs, 1 + 7]
        loss_odd = jnp.where(
            labels[:, 0] != 0.0,
            jnp.log(1 + jnp.exp(-labels[:, 0] * pred[:, 0])),
            0.0,
        )
        loss_even = jnp.mean(-labels[:, 1:] * jax.nn.log_softmax(pred[:, 1:]), axis=1)
        loss = jnp.mean(loss_odd + loss_even)
        return loss, pred

    @jax.jit
    def update(params, opt_state, graphs):
        grad_fn = jax.value_and_grad(loss_pred, has_aux=True)
        (loss, pred), grads = grad_fn(params, graphs)
        labels = graphs.globals
        accuracy_odd = jnp.sign(jnp.round(pred[:, 0])) == labels[:, 0]
        accuracy_even = jnp.argmax(pred[:, 1:], axis=1) == jnp.argmax(labels[:, 1:], axis=1)
        accuracy = (jnp.mean(accuracy_odd) + jnp.mean(accuracy_even)) / 2
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, accuracy, pred

    graphs = tetris()

    init = jax.jit(model.init)
    params = init(jax.random.PRNGKey(3), graphs)
    opt_state = opt.init(params)

    # compile jit
    wall = time.perf_counter()
    print("compiling...")
    _, _, _, _, pred = update(params, opt_state, graphs)
    pred.block_until_ready()

    print(f"It took {time.perf_counter() - wall:.1f}s to compile jit.")

    iterations = []
    for seed in range(seeds):
        params = init(jax.random.PRNGKey(seed), graphs)
        opt_state = opt.init(params)

        losses = []
        done = False
        wall = time.perf_counter()
        for it in range(1, steps + 1):
            params, opt_state, loss, accuracy, pred = update(params, opt_state, graphs)
            losses.append(loss)

            if not done and accuracy == 1.0:
                done = True
                total = time.perf_counter() - wall
                print(
                    f"[{seed}] 100% accuracy reached in {1000 * total:.0f}ms "
                    f"after {it} iterations ({1000 * total/it:.1f}ms/it)."
                )
                iterations += [it]

        if plot:
            plt.plot(losses)

    jnp.set_printoptions(precision=2, suppress=True)
    print(pred)

    if plot:
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.xscale("log")
        plt.yscale("log")
        plt.show()


if __name__ == "__main__":
    train()
