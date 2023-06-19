import jax
import jax.numpy as jnp


def get_pytree_dtype(*args, default_dtype=jnp.float32, real_part=False):
    leaves = jax.tree_util.tree_leaves(args)
    if len(leaves) == 0:
        return jnp.dtype(default_dtype)

    if real_part:
        return jax.eval_shape(
            lambda xs: sum(jnp.sum(jnp.real(x)) for x in xs), leaves
        ).dtype
    return jax.eval_shape(lambda xs: sum(jnp.sum(x) for x in xs), leaves).dtype
