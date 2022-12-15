Euclidean neural networks
=========================

What is ``e3nn-jax``?
---------------------

``e3nn-jax`` is a python library based on jax_ to create equivariant neural networks for the group :math:`O(3)`.

Amuse-bouche
------------

.. jupyter-execute::
    :hide-code:

    import jax
    jax.numpy.set_printoptions(precision=3, suppress=True)

``e3nn-jax`` contains many tools to manipulate irreps of the group :math:`O(3)`.

.. jupyter-execute::

    import jax
    import jax.numpy as jnp
    import haiku as hk

    import e3nn_jax as e3nn


    # Create a neural network
    @hk.without_apply_rng
    @hk.transform
    def net(x, f):
        # the inputs and outputs are all of type e3nn.IrrepsArray
        Y = e3nn.spherical_harmonics([0, 1, 2], x, False)
        f = e3nn.tensor_product(Y, f)
        return e3nn.Linear("0e + 0o + 1o")(f)

    # Create some inputs
    x = e3nn.IrrepsArray("1o", jnp.array([1.0, 2.0, 0.0]))
    f = e3nn.normal("4x0e + 1o + 1e", jax.random.PRNGKey(0), (16,))
    print(f"feature vector: {f.shape}")

    # Initialize the neural network
    w = net.init(jax.random.PRNGKey(0), x, f)
    print(jax.tree_util.tree_map(jnp.shape, w))

    # Evaluate the neural network
    f = net.apply(w, x, f)
    print(f"feature vector: {f.shape}")


Why rewrite e3nn in jax?
------------------------

Jax has two beautiful function transformations: `jax.grad` and `jax.vmap`.

On top of that it is very powerful to optimize code.
It can for instance get rid of the dead code:

.. jupyter-execute::
    :hide-code:

    from e3nn_jax._src.util.jit import jit_code

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
        return jax.grad(jnp.exp)((x + 1) - 1)

    @jax.jit
    def h(x):
        return jnp.exp(jnp.cos(0) * x)

    print(jit_code(f, 1.0))


Irreps
------

In e3nn we have a notation to define direct sums of irreducible representations of :math:`O(3)`.

.. jupyter-execute::

    e3nn.Irreps("0e + 2x1o")


This mean one scalar and two vectors.
``0e`` stands for the even irrep ``L=0`` and ``1o`` stands for the odd irrep ``L=1``.
The suffixes ``e`` and ``o`` stand for even and odd -- the representation of parity.

The class `Irreps` has many methods to manipulate the representations.

IrrepsArray
-----------

`IrrepsArray` contains an ``irreps`` attribute of class `Irreps` and an ``array`` attribute of class `jax.numpy.ndarray`.

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

`IrrepsArray` handles

* binary operations:

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


* Indexing:

.. jupyter-execute::

    x[0]

.. jupyter-execute::

    x[1, "1o"]

.. jupyter-execute::

    x[..., "1o"]

.. jupyter-execute::

    x[..., "2x0e + 1o"]

.. jupyter-execute::

    x[..., 2:]


* Reductions:

.. jupyter-execute::

    e3nn.mean(y)

.. jupyter-execute::

    e3nn.sum(x)

.. jupyter-execute::

    e3nn.sum(x, axis=0)

.. jupyter-execute::

    e3nn.sum(x, axis=1)


* And other operations:

.. jupyter-execute::

    e3nn.concatenate([x, x], axis=0)

.. jupyter-execute::

    z = e3nn.concatenate([x, y], axis=1)
    z

.. jupyter-execute::

    z.sorted().simplify()

.. jupyter-execute::

    x.reshape((1, 3, -1))

.. jupyter-execute::

    x1, x2, x3 = x
    x1





Tensor prodcut
--------------

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

The function `e3nn_jax.sh` is a wrapper of `e3nn_jax.spherical_harmonics` for which inputs and outputs are `IrrepsArray`.


Gradient
--------

The gradient of an equivariant function is also equivariant.
If a function inputs and outputs `IrrepsArray`, we can compute its gradient using `e3nn_jax.grad`.

.. jupyter-execute::

    def cross_product(x, y):
        return e3nn.tensor_product(x, y)["1e"]

    x = e3nn.IrrepsArray("1o", jnp.array([1.0, 2.0, 0.0]))
    y = e3nn.IrrepsArray("1o", jnp.array([2.0, 2.0, 1.0]))
    e3nn.grad(cross_product)(x, y)


Here is a vector and its spherical harmonics of degree :math:`L=2`: :math:`Y^2`

.. jupyter-execute::

    x = e3nn.IrrepsArray("1o", jnp.array([1.0, 2, 3]))
    e3nn.spherical_harmonics("2e", x, False)

We can verify that it can also be obtained by taking the gradient of :math:`Y^3`

.. jupyter-execute::

    f = e3nn.grad(lambda x: e3nn.spherical_harmonics("3o", x, False))
    0.18443 * f(x)["2e"]

API
---

.. toctree::
    :maxdepth: 2

    api/irreps
    api/irreps_array
    api/tensor_products
    api/operations
    api/math
    api/nn
    api/radial
    api/haiku
    api/flax
    api/extra

.. _jax: https://github.com/google/jax
