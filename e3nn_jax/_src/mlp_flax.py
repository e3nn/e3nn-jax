from typing import Callable, Optional, Tuple, Union

import flax
import jax.numpy as jnp
import e3nn_jax as e3nn


class MultiLayerPerceptron(flax.linen.Module):
    r"""Just a simple MLP for scalars. No equivariance here.

    Args:
        list_neurons (list of int): number of neurons in each layer (excluding the input layer)
        act (optional callable): activation function
        gradient_normalization (str or float): normalization of the gradient

            - "element": normalization done in initialization variance of the weights, (the default in pytorch)
                gives the same importance to each neuron, a layer with more neurons will have a higher importance
                than a layer with less neurons
            - "path" (default): normalization done explicitly in the forward pass,
                gives the same importance to every layer independently of the number of neurons
    """
    list_neurons: Tuple[int, ...]
    act: Optional[Callable] = None
    gradient_normalization: Union[str, float] = None
    output_activation: Union[Callable, bool] = True

    @flax.linen.compact
    def __call__(
        self, x: Union[jnp.ndarray, e3nn.IrrepsArray]
    ) -> Union[jnp.ndarray, e3nn.IrrepsArray]:
        """Evaluate the MLP

        Input and output are either `jax.numpy.ndarray` or `IrrepsArray`.
        If the input is a `IrrepsArray`, it must contain only scalars.

        Args:
            x (IrrepsArray): input of shape ``[..., input_size]``

        Returns:
            IrrepsArray: output of shape ``[..., list_neurons[-1]]``
        """
        output_activation = self.output_activation

        if output_activation is True:
            output_activation = self.act
        elif output_activation is False:
            output_activation = None
        else:
            assert callable(output_activation)

        gradient_normalization = self.gradient_normalization
        if gradient_normalization is None:
            gradient_normalization = e3nn.config("gradient_normalization")
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[
                gradient_normalization
            ]

        if isinstance(x, e3nn.IrrepsArray):
            if not x.irreps.is_scalar():
                raise ValueError("MLP only works on scalar (0e) input.")
            x = x.array
            output_irrepsarray = True
        else:
            output_irrepsarray = False

        act = None if self.act is None else e3nn.normalize_function(self.act)
        last_act = (
            None
            if output_activation is None
            else e3nn.normalize_function(output_activation)
        )

        for i, h in enumerate(self.list_neurons):
            alpha = 1 / x.shape[-1]
            d = flax.linen.Dense(
                features=h,
                use_bias=False,
                kernel_init=flax.linen.initializers.normal(
                    stddev=jnp.sqrt(alpha) ** (1.0 - gradient_normalization)
                ),
                param_dtype=x.dtype,
            )
            x = jnp.sqrt(alpha) ** gradient_normalization * d(x)
            if i < len(self.list_neurons) - 1:
                if act is not None:
                    x = act(x)
            else:
                if last_act is not None:
                    x = last_act(x)

        if output_irrepsarray:
            x = e3nn.IrrepsArray(e3nn.Irreps(f"{x.shape[-1]}x0e"), x)
        return x
