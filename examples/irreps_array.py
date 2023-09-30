import jax.numpy as jnp

import e3nn_jax as e3nn


x1 = e3nn.IrrepsArray("0e + 1o", jnp.array([1.0, 2.0, 3.0, 4.0]))
x2 = e3nn.IrrepsArray("2e", jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))

y = e3nn.tensor_product(x1, x2)


print(x1)
print("tensor product with")
print(x2)
print("is")
print(y)
