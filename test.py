import jax
import jax.numpy as np

import e3nn_jax as e3nn

# some dummy e3-array
a = e3nn.IrrepsArray.zeros("8x0e + 8x1o + 8x2e", (1,))
# some scalar array
n = a.irreps.num_irreps
b = e3nn.IrrepsArray.zeros(f"{n}x0e", (1,))

for i in range(3):
    print(a * b)
