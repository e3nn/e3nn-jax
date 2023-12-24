Euclidean neural networks
=========================

.. jupyter-execute::
    :hide-code:

    import plotly.graph_objects as go
    import e3nn_jax as e3nn

    signal = e3nn.SphericalSignal.zeros(40, 80, "soft")
    sh = e3nn.spherical_harmonics("2e", signal.grid_vectors, True)
    signal.grid_values = sh.array[:, :, 2]
    go.Figure(
        data=[
            go.Surface(
                signal.plotly_surface(scale_radius_by_amplitude=True),
                showscale=False,
                colorscale=[[0, "rgb(0,50,255)"], [0.5, "rgb(200,200,200)"], [1, "rgb(255,50,0)"]],
                cmin=-1.5,
                cmax=1.5,
            )
        ],
        layout=dict(
            width=500,
            height=400,
            scene=dict(
                xaxis=dict(
                    title="x",
                    tickvals=[],
                ),
                yaxis=dict(title="y", tickvals=[]),
                zaxis=dict(title="z", tickvals=[]),
                camera=dict(
                    eye=dict(x=2.6, y=0.0, z=0.0),
                    center=dict(x=0.0, y=0.0, z=0.0),
                    up=dict(x=0.0, y=1.0, z=0.0),
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
        ),
    )


What is ``e3nn-jax``?
---------------------

``e3nn-jax`` is a python library built on jax_ to create :math:`O(3)`-equivariant neural networks.



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
        return e3nn.haiku.Linear("0e + 0o + 1o")(f)

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

    from e3nn_jax._src.utils.jit import jit_code

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

    z.sort().simplify()

.. jupyter-execute::

    x.reshape((1, 3, -1))

.. jupyter-execute::

    x1, x2, x3 = x
    x1





Tensor prodcut
--------------

Let's create a list of 10 vectors (``1o`` irreps) and a list of 10 ``2e`` irreps and compute their tensor product.

.. jupyter-execute::

    x1 = e3nn.normal("1e", jax.random.PRNGKey(0), (10,))
    x2 = e3nn.normal("2e", jax.random.PRNGKey(1), (10,))
    e3nn.tensor_product(x1, x2)



Learnable Modules
-----------------

We use ``dm-haiku`` or ``flax`` to create parameterized modules.
Here is an example using ``e3nn_jax.flax.Linear``.

.. jupyter-execute::

    import flax

    model = e3nn.flax.Linear("1e")

    x = e3nn.IrrepsArray("2x1e", jnp.array([1.0, 0.0, 0.0, 0.0, 2.0, 0.0]))

    # initialize the parameters randomly
    w = model.init(jax.random.PRNGKey(0), x)

    # apply the module
    model.apply(w, x)


Spherical Harmonics
-------------------

Let's compute the sphercal harmonics of degree :math:`L=2` for :math:`\vec x = (0, 0, 1)` using the function `e3nn_jax.spherical_harmonics`.

.. jupyter-execute::

    vector = e3nn.IrrepsArray("1o", jnp.array([0.0, 0.0, 1.0]))
    e3nn.spherical_harmonics(2, vector, normalize=True)


Note the ``normalize`` option. If ``normalize`` is ``False``, the function becomes an homogeneous polynomial, see below:

.. jupyter-execute::

    a = 3.5

    assert jnp.allclose(
        e3nn.spherical_harmonics(2, a * vector, False).array,
        a**2 * e3nn.spherical_harmonics(2, vector, False).array,
    )


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

Content
-------

.. toctree::
    :maxdepth: 1

    api/index
    tuto/index
    BENCHMARK

.. _jax: https://github.com/google/jax
