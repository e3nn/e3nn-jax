import jax
import jax.numpy as jnp


def get_pytree_dtype(*args, default_dtype=jnp.float32):
    leaves = jax.tree_util.tree_leaves(args)
    if len(leaves) == 0:
        return default_dtype

    return jax.eval_shape(lambda xs: sum(jnp.sum(x) for x in xs), leaves).dtype
