Euclidean neural networks
=========================

What is ``e3nn-jax``?
---------------------

``e3nn-jax`` is a python library based on jax_ to create equivariant neural networks for the group :math:`O(3)`.

Example
-------

.. jupyter-execute::
    :hide-code:

    import jax
    jax.numpy.set_printoptions(precision=3, suppress=True)

``e3nn-jax`` contains many tools to manipulate irreps of the group :math:`O(3)`.

.. jupyter-execute::

    import jax
    import jax.numpy as jnp
    import e3nn_jax as e3nn

    # Create a vector and compute its tensor product
    x = e3nn.IrrepsArray("1o", jnp.array([1.0, 2.0, 0.0]))
    y = e3nn.tensor_product(x, x)
    print(y)
    print(e3nn.norm(y))

And contains haiku modules to make learnable neural networks.

.. jupyter-execute::

    import haiku as hk

    # Create a neural network
    @hk.without_apply_rng
    @hk.transform
    def net(x, f):
        # the inputs and outputs are all of type e3nn.IrrepsArray
        Y = e3nn.spherical_harmonics([0, 1, 2], x, False)
        f = e3nn.tensor_product(Y, f)
        return e3nn.Linear("0e + 0o + 1o")(f)

    # Create random inputs
    f = e3nn.normal("4x0e + 1o + 1e", jax.random.PRNGKey(0), (16,))
    print(f"feature vector: {f.shape}")

    # Initialize the neural network
    w = net.init(jax.random.PRNGKey(0), x, f)
    print(jax.tree_util.tree_map(jnp.shape, w))

    # Evaluate the neural network
    f = net.apply(w, x, f)
    print(f"feature vector: {f.shape}")


Irreps
------

If two tensors :math:`x` and :math:`y` transforms as :math:`D_x = 2 \times 1_o` (two vectors) and :math:`D_y = 0_e + 1_e` (a scalar and a pseudovector) respectively, where the indices :math:`e` and :math:`o` stand for even and odd -- the representation of parity,

.. jupyter-execute::

    irreps_x = e3nn.Irreps("2x1o")
    irreps_y = e3nn.Irreps("0e + 1e")

    x = e3nn.normal(irreps_x, jax.random.PRNGKey(0), ())
    y = e3nn.normal(irreps_y, jax.random.PRNGKey(1), ())

    irreps_x.dim, irreps_y.dim


their outer product is a :math:`6 \times 4` matrix of two indices :math:`A_{ij} = x_i y_j`.

.. jupyter-execute::

    A = jnp.einsum("i,j", x.array, y.array)
    A


If a rotation is applied to the system, this matrix will transform with the representation :math:`D_x \otimes D_y` (the tensor product representation).

.. math::

    A = x y^t \longrightarrow A' = D_x A D_y^t

Which can be represented by

.. jupyter-execute::
    :hide-code:

    import matplotlib.pyplot as plt

.. jupyter-execute::

    R = e3nn.rand_matrix(jax.random.PRNGKey(2), ())
    D_x = irreps_x.D_from_matrix(R)
    D_y = irreps_y.D_from_matrix(R)

    plt.imshow(jnp.kron(D_x, D_y), cmap="bwr", vmin=-1, vmax=1);


This representation is not irreducible (is reducible). It can be decomposed into irreps by a change of basis. The outerproduct followed by the change of basis is done by the class `e3nn_jax.tensor_product`.

.. jupyter-execute::

    tp = e3nn.tensor_product(x, y)
    tp


As a sanity check, we can verify that the representation of the tensor prodcut is block diagonal and of the same dimension.

.. jupyter-execute::

    D = tp.irreps.D_from_matrix(R)
    plt.imshow(D, cmap="bwr", vmin=-1, vmax=1);



jax jit capabilities
--------------------

``e3nn-jax`` expoit the capabilities of ``jax.jit`` to optimize the code.
It can for instance get rid of the dead code:

.. jupyter-execute::
    :hide-code:

    from e3nn_jax.util.jit import jit_code

.. jupyter-execute::

    def f(x):
        y = jnp.exp(x)
        return x + 1

    print(jit_code(f, 1.0))

It will reuse the same expression instead of computing it again. The following code calls twice the exponential function, but it will only compute it once.

.. jupyter-execute::

    def f(x):
        return jnp.exp(x) + jnp.exp(x)

    print(jit_code(f, 1.0))

This mechanism is quite robust.

.. jupyter-execute::

    def f(x):
        x = jnp.stack([x, x])
        y1 = g(x[0])
        y2 = h(x[1])
        x = jnp.array([y1, y2])
        return jnp.sum(x)

    @jax.jit
    def g(x):
        return jnp.exp((x + 1) - 1)

    @jax.jit
    def h(x):
        return jnp.exp(jnp.cos(0) * x)

    print(jit_code(f, 1.0))

IrrepsArray
-----------

`e3nn_jax.IrrepsArray` contains the data of an irreducible representation.
It rely on the ``jax.jit`` compiler because it contains both a ``array`` and a ``list`` representation of the data.

.. jupyter-execute::

    x = e3nn.IrrepsArray("2x0e + 1o", jnp.array(
        [
            [1.0, 0.0,  0.0, 0.0, 0.0],
            [0.0, 1.0,  1.0, 0.0, 0.0],
            [0.0, 0.0,  0.0, 1.0, 0.0],
        ]
    ))
    x

.. jupyter-execute::

    y = e3nn.IrrepsArray("0o + 2x0e", jnp.array(
        [
            [1.5,  0.0, 1.0],
            [0.5, -1.0, 2.0],
            [0.5,  1.0, 1.5],
        ]
    ))

The irrep index is always the last index.

.. jupyter-execute::

    assert x.irreps.dim == x.shape[-1]
    x.shape

`e3nn_jax.IrrepsArray` handles binary operations:

.. jupyter-execute::

    x + x

.. jupyter-execute::

    2.0 * x

.. jupyter-execute::

    x / 2.0

.. jupyter-execute::

    x * y

.. jupyter-execute::

    x / y

.. jupyter-execute::

    1.0 / y

.. jupyter-execute::

    x == x

Indexing:

.. jupyter-execute::

    x[0]

.. jupyter-execute::

    x[1, "1o"]

.. jupyter-execute::

    x[..., "1o"]

.. jupyter-execute::

    x[..., "2x0e + 1o"]

Reductions:

.. jupyter-execute::

    e3nn.mean(y)

.. jupyter-execute::

    e3nn.sum(x)

.. jupyter-execute::

    e3nn.sum(x, axis=0)

.. jupyter-execute::

    e3nn.sum(x, axis=1)

And other operations:

.. jupyter-execute::

    e3nn.concatenate([x, x], axis=0)

.. jupyter-execute::

    e3nn.concatenate([x, y], axis=1)

.. jupyter-execute::

    x.reshape((1, 3, -1))

.. jupyter-execute::

    x1, x2, x3 = x
    x1





Tensor prodcut with weights
---------------------------

We use ``dm-haiku`` to create parameterized modules.

.. jupyter-execute::

    import haiku as hk

``dm-haiku`` ask to create a function that does only take the inputs as arguments (no parameters) and then this function is transformed.

.. jupyter-execute::

    @hk.without_apply_rng
    @hk.transform
    def tp(x1, x2):
        return e3nn.Linear("1e")(e3nn.tensor_product(x1, x2))

Note that the inputs irreps are not yet specified, ``"1e"`` here specify the output.
Let's define two random inputs and initialize the parameters:

.. jupyter-execute::

    x1 = e3nn.normal("1e", jax.random.PRNGKey(0), (10,))
    x2 = e3nn.normal("1e", jax.random.PRNGKey(1), (10,))
    w = tp.init(jax.random.PRNGKey(2), x1, x2)

Now that we have the weights, we can use them to compute the output.

.. jupyter-execute::

    tp.apply(w, x1, x2)


Spherical Harmonics
-------------------

Let's compute the sphercal harmonics of degree :math:`L=2` for :math:`\vec x = (0, 0, 1)`.

.. jupyter-execute::

    e3nn.sh(2, jnp.array([0.0, 0.0, 1.0]), normalize=True)


Note the ``normalize`` option. If ``normalize`` is ``False``, the function is an homogeneous polynomial.

.. jupyter-execute::

    x = jnp.array([0.0, 0.0, 1.0])
    a = 3.5

    assert jnp.allclose(
        e3nn.sh(2, a * x, False),
        a**2 * e3nn.sh(2, x, False),
    )


API
---

.. toctree::
    :maxdepth: 2

    api/irreps
    api/irreps_array
    api/operations
    api/math
    api/nn
    api/extra

.. _jax: https://github.com/google/jax
