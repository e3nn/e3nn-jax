import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from e3nn_jax import Gate, Irreps
from e3nn_jax.experimental.voxel_convolution import Convolution
from tqdm.auto import tqdm


def tetris():
    pos = [
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],  # zigzag
    ]

    voxels = np.zeros((8, 9, 9, 9, 1), np.float32)
    for ps, v in zip(pos, voxels):
        for x, y, z in ps:
            v[3 + x, 3 + y, 3 + z] = 1

    # Since chiral shapes are the mirror of one another we need an *odd* scalar to distinguish them
    labels = np.array([
        [+1, 0, 0, 0, 0, 0, 0],  # chiral_shape_1
        [-1, 0, 0, 0, 0, 0, 0],  # chiral_shape_2
        [0, 1, 0, 0, 0, 0, 0],  # square
        [0, 0, 1, 0, 0, 0, 0],  # line
        [0, 0, 0, 1, 0, 0, 0],  # corner
        [0, 0, 0, 0, 1, 0, 0],  # L
        [0, 0, 0, 0, 0, 1, 0],  # T
        [0, 0, 0, 0, 0, 0, 1],  # zigzag
    ], dtype=np.float32)

    return voxels, labels


def main():
    # Model
    @hk.without_apply_rng
    @hk.transform
    def model(x):
        mul0 = 16
        mul1 = 8
        gate = Gate(f'{mul0}x0e + {mul0}x0o', [jax.nn.gelu, jnp.tanh], f'{2 * mul1}x0e', [jax.nn.sigmoid], f'{mul1}x1e + {mul1}x1o')

        def g(x):
            y = jax.vmap(gate)(x.reshape(-1, x.shape[-1]))
            y = gate.irreps_out.as_tensor(y)
            return y.reshape(x.shape[:-1] + (-1,))

        kw = dict(irreps_sh=Irreps('0e + 1o'), diameter=3.0, num_radial_basis=2, steps=(1.0, 1.0, 1.0))

        x = g(Convolution(Irreps('0e'), gate.irreps_in, **kw)(x))

        for _ in range(2):
            x = g(Convolution(gate.irreps_out, gate.irreps_in, **kw)(x))

        x = Convolution(gate.irreps_out, Irreps('0o + 6x0e'), **kw)(x)

        x = jnp.sum(x, axis=(1, 2, 3))
        return 0.1 * x

    # Optimizer
    learning_rate = 0.1
    opt = optax.adam(learning_rate)

    # Update function
    @jax.jit
    def update(params, opt_state, x, y):
        def loss_fn(params):
            pred = model.apply(params, x)
            loss = jnp.sum(jnp.abs(pred - y)) / y.shape[0]
            return loss, pred

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, pred), grads = grad_fn(params)
        accuracy = jnp.mean(jnp.all(jnp.round(pred) == y, axis=1))

        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, accuracy

    # Dataset
    x, y = tetris()

    # Init
    rng = jax.random.PRNGKey(3)
    params = model.init(rng, x)
    opt_state = opt.init(params)

    # Train
    for _ in tqdm(range(100)):
        params, opt_state, loss, accuracy = update(params, opt_state, x, y)
        loss.block_until_ready()
        print(f"{loss:.2f}")

    print(f"accuracy = {100 * accuracy:.0f}%")


if __name__ == '__main__':
    main()
