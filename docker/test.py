import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps, IrrepsData
from e3nn_jax.experimental.voxel_convolution import Convolution


irreps_in = Irreps("2x0e + 3x1e + 2x2e")
irreps_out = Irreps("0e + 2x1e + 2e")
irreps_sh = Irreps("0e + 1e + 2e")


@hk.without_apply_rng
@hk.transform
def c(x, z):
    x = IrrepsData.from_contiguous(irreps_in, x)
    x = Convolution(
        irreps_out=irreps_out,
        irreps_sh=irreps_sh,
        diameter=3.9,
        num_radial_basis={0: 3, 1: 2, 2: 1},
        relative_starts={0: 0.0, 1: 0.0, 2: 0.5},
        steps=((1.0, 1.0, 1.0), z),
    )(x)
    return x.contiguous


f = jax.jit(c.apply)

x0 = irreps_in.randn(jax.random.PRNGKey(0), (3, 8, 8, 8, -1))
x0 = jnp.pad(x0, ((0, 0), (4, 4), (4, 4), (4, 4), (0, 0)))

w = c.init(jax.random.PRNGKey(0), x0, jnp.array([1.0, 1.0, 1.0]))
y = f(w, x0, jnp.array([1.0, 1.02, 0.98]))

print(y.dtype, y.device())
