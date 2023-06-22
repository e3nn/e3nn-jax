import time

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm

import e3nn_jax as e3nn
from e3nn_jax.experimental.voxel_convolution import ConvolutionFlax


def tetris():
    pos = [
        [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0]],  # chiral_shape_1
        [[1, 1, 1], [1, 1, 2], [2, 1, 1], [2, 0, 1]],  # chiral_shape_2
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],  # square
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],  # line
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],  # corner
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]],  # L
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1]],  # T
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]],  # zigzag
    ]

    voxels = np.zeros((8, 9, 9, 9), np.float32)
    for ps, v in zip(pos, voxels):
        for x, y, z in ps:
            v[4 + x, 4 + y, 4 + z] = 1

    # Since chiral shapes are the mirror of one another we need an *odd* scalar to distinguish them
    labels = np.arange(8)

    return voxels, labels


class Model(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x):
        mul0 = 16
        mul1 = 4

        g = e3nn.gate
        for _ in range(1 + 3):
            g = e3nn.utils.vmap(g)

        # Shallower and wider convolutions also works

        # kw = dict(irreps_sh=Irreps('0e + 1o'), diameter=5.5, num_radial_basis=3, steps=(1.0, 1.0, 1.0))
        kw = dict(
            irreps_sh=e3nn.Irreps("0e + 1o"),
            diameter=2 * 1.4,
            num_radial_basis=1,
            steps=(1.0, 1.0, 1.0),
        )

        x = e3nn.IrrepsArray("0e", x[..., None])

        # for _ in range(2):
        for _ in range(5):
            x = g(
                ConvolutionFlax(
                    f"{mul0}x0e + {mul0}x0o + {2 * mul1}x0e + {mul1}x1e + {mul1}x1o",
                    **kw,
                )(x)
            )

        x = ConvolutionFlax("0o + 7x0e", **kw)(x)

        x = x.array
        pred = jnp.sum(x, axis=(1, 2, 3))

        odd, even1, even2 = pred[:, :1], pred[:, 1:2], pred[:, 2:]
        logits = jnp.concatenate([odd * even1, -odd * even1, even2], axis=1)

        return logits


def train(steps=2000):
    model = Model()

    # Optimizer
    opt = optax.adam(learning_rate=0.1)

    def loss_fn(params, x, y):
        logits = model.apply(params, x)
        labels = y

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = jnp.mean(loss)
        return loss, logits

    @jax.jit
    def update_fn(params, opt_state, x, y):
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, logits = grad_fn(params, x, y)
        labels = y
        accuracy = jnp.mean(jnp.argmax(logits, axis=1) == labels)

        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, accuracy

    # Dataset
    x, y = tetris()

    # Init
    init = jax.jit(model.init)
    params = init(jax.random.PRNGKey(3), x)
    opt_state = opt.init(params)

    # compile jit
    wall = time.perf_counter()
    print("compiling...", flush=True)
    _, _, accuracy = update_fn(params, opt_state, x, y)
    print(f"initial accuracy = {100 * accuracy:.0f}%", flush=True)
    print(f"compilation took {time.perf_counter() - wall:.1f}s")

    # Train
    wall = time.perf_counter()
    print("training...", flush=True)
    for _ in tqdm(range(steps)):
        params, opt_state, accuracy = update_fn(params, opt_state, x, y)
        if accuracy == 1.0:
            break

    print(f"final accuracy = {100 * accuracy:.0f}%")


if __name__ == "__main__":
    train()
