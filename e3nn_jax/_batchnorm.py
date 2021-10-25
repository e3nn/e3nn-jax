
import haiku as hk
import jax.numpy as jnp
import jax

from e3nn_jax import Irreps


class BatchNorm(hk.Module):
    """Batch normalization for orthonormal representations
    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.
    Parameters
    ----------
    irreps : `Irreps`
        representation
    eps : float
        avoid division by zero when we normalize by the variance
    momentum : float
        momentum of the running average
    affine : bool
        do we have weight and bias parameters
    reduce : {'mean', 'max'}
        method used to reduce
    instance : bool
        apply instance norm instead of batch norm
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
        """evaluate
        Parameters
        ----------
        input : `DeviceArray`
            tensor of shape ``(batch, ..., irreps.dim)``
        Returns
        -------
        `DeviceArray`
            tensor of shape ``(batch, ..., irreps.dim)``
        """
        if not self.instance:
            running_mean = hk.get_state("running_mean", shape=(self.num_scalar,), init=jnp.zeros)
            running_var = hk.get_state("running_var", shape=(self.num_features,), init=jnp.ones)
        if self.affine:
            weight = hk.get_parameter("weight", shape=(self.num_features,), init=jnp.ones)
            bias = hk.get_parameter("bias", shape=(self.num_scalar,), init=jnp.zeros)

        batch, *size, dim = input.shape
        input = input.reshape(batch, -1, dim)  # [batch, sample, stacked features]

        if is_training and not self.instance:
            new_means = []
            new_vars = []

        fields = []
        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:
            d = ir.dim
            field = input[:, :, ix: ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d

            # [batch, sample, mul, repr]
            field = field.reshape(batch, -1, mul, d)

            if ir.is_scalar():  # scalars
                if is_training or self.instance:
                    if self.instance:
                        field_mean = field.mean(1).reshape(batch, mul)  # [batch, mul]
                    else:
                        field_mean = field.mean([0, 1]).reshape(mul)  # [mul]
                        new_means.append(
                            self._roll_avg(running_mean[irm:irm + mul], field_mean)
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
                    new_vars.append(self._roll_avg(running_var[irv: irv + mul], field_norm))
            else:
                field_norm = running_var[irv: irv + mul]
            irv += mul

            field_norm = jax.lax.rsqrt(field_norm + self.eps)  # [(batch,) mul]

            if self.affine:
                sub_weight = weight[iw: iw + mul]  # [mul]
                iw += mul

                field_norm = field_norm * sub_weight  # [(batch,) mul]

            field = field * field_norm.reshape(-1, 1, mul, 1)  # [batch, sample, mul, repr]

            if self.affine and ir.is_scalar():  # scalars
                sub_bias = bias[ib: ib + mul]  # [mul]
                ib += mul
                field += sub_bias.reshape(mul, 1)  # [batch, sample, mul, repr]

            fields.append(field.reshape(batch, -1, mul * d))  # [batch, sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        if is_training and not self.instance:
            assert irm == running_mean.size
            assert irv == running_var.shape[0]
        if self.affine:
            assert iw == weight.shape[0]
            assert ib == bias.size

        if is_training and not self.instance:
            if len(new_means):
                hk.set_state("running_mean", jnp.concatenate(new_means))
            if len(new_vars):
                hk.set_state("running_var", jnp.concatenate(new_vars))

        output = jnp.concatenate(fields, axis=2)  # [batch, sample, stacked features]
        return output.reshape(batch, *size, dim)