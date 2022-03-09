
import haiku as hk
import jax.numpy as jnp
import jax

from e3nn_jax import Irreps, IrrepsData


class BatchNorm(hk.Module):
    """Batch normalization for orthonormal representations
    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.

    Args:
        irreps: Irreducible representations
        eps (float): small number to avoid division by zero
        momentum: momentum for moving average
        affine: whether to include learnable biases
        reduce: reduce mode, either 'mean' or 'max'
        instance: whether to use instance normalization
        normalization: normalization mode, either 'norm' or 'component'
    """
    def __init__(self, irreps, eps=1e-5, momentum=0.1, affine=True, reduce='mean', instance=False, normalization='component'):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.instance = instance

        self.num_scalar = sum(mul for mul, ir in self.irreps if ir.is_scalar())
        self.num_features = self.irreps.num_irreps

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ['mean', 'max'], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps}, momentum={self.momentum})"

    def _roll_avg(self, curr, update):
        return (1 - self.momentum) * curr + self.momentum * jax.lax.stop_gradient(update)

    def __call__(self, input, is_training=True):
        r"""evaluate the batch normalization

        Args:
            input: input tensor of shape ``(..., irreps.dim)``
            is_training: whether to train or evaluate

        Returns:
            output: normalized tensor of shape ``(..., irreps.dim)``
        """
        if not self.instance:
            running_mean = hk.get_state("running_mean", shape=(self.num_scalar,), init=jnp.zeros)
            running_var = hk.get_state("running_var", shape=(self.num_features,), init=jnp.ones)
        if self.affine:
            weight = hk.get_parameter("weight", shape=(self.num_features,), init=jnp.ones)
            bias = hk.get_parameter("bias", shape=(self.num_scalar,), init=jnp.zeros)

        input = IrrepsData.new(self.irreps, input)
        batch, *size = input._shape_from_list()
        input = input.list
        input = [x.reshape(batch, -1, mul, ir.dim) for (mul, ir), x in zip(self.irreps, input)]

        if is_training and not self.instance:
            new_means = []
            new_vars = []

        fields = []

        # You need all of these constants because of the ordering of the irreps
        i = 0
        irm = 0
        ib = 0

        for (mul, ir), field in zip(self.irreps, input):
            k = i + mul

            # [batch, sample, mul, repr]

            if ir.is_scalar():  # scalars
                if is_training or self.instance:
                    if self.instance:
                        field_mean = field.mean(1).reshape(batch, mul)  # [batch, mul]
                    else:
                        field_mean = field.mean([0, 1]).reshape(mul)  # [mul]
                        new_means.append(
                            self._roll_avg(running_mean[irm: irm + mul], field_mean)
                        )
                else:
                    field_mean = running_mean[irm: irm + mul]
                irm += mul

                # [batch, sample, mul, repr]
                field = field - field_mean.reshape(-1, 1, mul, 1)

            if is_training or self.instance:
                if self.normalization == 'norm':
                    field_norm = jnp.square(field).sum(3)  # [batch, sample, mul]
                elif self.normalization == 'component':
                    field_norm = jnp.square(field).mean(3)  # [batch, sample, mul]
                else:
                    raise ValueError("Invalid normalization option {}".format(self.normalization))

                if self.reduce == 'mean':
                    field_norm = field_norm.mean(1)  # [batch, mul]
                elif self.reduce == 'max':
                    field_norm = field_norm.max(1)  # [batch, mul]
                else:
                    raise ValueError("Invalid reduce option {}".format(self.reduce))

                if not self.instance:
                    field_norm = field_norm.mean(0)  # [mul]
                    new_vars.append(self._roll_avg(running_var[i: k], field_norm))
            else:
                field_norm = running_var[i: k]

            field_norm = jax.lax.rsqrt(field_norm + self.eps)  # [(batch,) mul]

            if self.affine:
                sub_weight = weight[i: k]  # [mul]
                field_norm = field_norm * sub_weight  # [(batch,) mul]

            field = field * field_norm.reshape(-1, 1, mul, 1)  # [batch, sample, mul, repr]

            if self.affine and ir.is_scalar():  # scalars
                sub_bias = bias[ib: ib + mul]  # [mul]
                field += sub_bias.reshape(mul, 1)  # [batch, sample, mul, repr]
                ib += mul

            fields.append(field)  # [batch, sample, mul, repr]
            i = k

        if is_training and not self.instance:
            if len(new_means):
                hk.set_state("running_mean", jnp.concatenate(new_means))
            if len(new_vars):
                hk.set_state("running_var", jnp.concatenate(new_vars))

        output = [x.reshape(batch, *size, mul, ir.dim) for (mul, ir), x in zip(self.irreps, fields)]
        return IrrepsData.from_list(self.irreps, output)
