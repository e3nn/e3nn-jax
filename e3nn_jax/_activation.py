import jax
import jax.numpy as jnp

from e3nn_jax import Irreps, IrrepsData


def normalize_function(phi):
    with jax.ensure_compile_time_eval():
        k = jax.random.PRNGKey(0)
        x = jax.random.normal(k, (1_000_000,))
        c = jnp.mean(phi(x)**2)**0.5

        if jnp.allclose(c, 1.0):
            return phi
        else:
            def rho(x):
                return phi(x) / c
            return rho


def parity_function(phi):
    with jax.ensure_compile_time_eval():
        x = jnp.linspace(0.0, 10.0, 256)

        a1, a2 = phi(x), phi(-x)
        if jnp.max(jnp.abs(a1 - a2)) < 1e-5:
            return 1
        elif jnp.max(jnp.abs(a1 + a2)) < 1e-5:
            return -1
        else:
            return 0


class ScalarActivation:
    irreps_in: Irreps
    irreps_out: Irreps

    def __init__(self, irreps_in, acts):
        irreps_in = Irreps(irreps_in)
        assert len(irreps_in) == len(acts), (irreps_in, acts)

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError("Activation: cannot apply an activation function to a non-scalar input.")

                p_out = parity_function(act) if p_in == -1 else p_in
                irreps_out.append((mul, (0, p_out)))

                if p_out == 0:
                    raise ValueError("Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd.")
            else:
                irreps_out.append((mul, (l_in, p_in)))

        # normalize the second moment
        acts = [normalize_function(act) if act is not None else None for act in acts]

        self.irreps_in = irreps_in
        self.irreps_out = Irreps(irreps_out)
        self.acts = acts

    def __call__(self, features):
        features = IrrepsData.new(self.irreps_in, features)
        # TODO fix the case cos(None) = 1 (not None)
        list = [x if act is None or x is None else act(x) for act, x in zip(self.acts, features.list)]
        if self.acts and self.acts.count(self.acts[0]) == len(self.acts):
            # for performance, if all the activation functions are the same, we can apply it to the contiguous array as well
            contiguous = features.contiguous if self.acts[0] is None else self.acts[0](features.contiguous)
            return IrrepsData(self.irreps_out, contiguous, list)
        return IrrepsData.from_list(self.irreps_out, list)


class KeyValueActivation:
    irreps_key: Irreps
    irreps_value: Irreps
    irreps_out: Irreps

    def __init__(self, irreps_key, irreps_value, phi):
        self.irreps_key = Irreps(irreps_key)
        self.irreps_value = Irreps(irreps_value)

        # irreps_out =

    def __call__(self, keys, values):
        return [jax.vmap(key_value_activation)(self.phi, k, v) for k, v in zip(keys, values)]


def key_value_activation(phi, key, value):
    assert key.ndim == 1
    assert value.ndim == 1

    d = value.shape[0]
    key = key / jnp.sqrt(1/16 + jnp.sum(key**2))  # 1/16 is arbitrary small... but not too small...
    scalar = jnp.sum(key * value)
    scalar = normalize_function(phi)(scalar)
    return d**0.5 * scalar * key  # component normalized
