Euclidean neural networks
=========================

What is ``e3nn-jax``?
---------------------

``e3nn-jax`` is a python library based on jax_ to create equivariant neural networks for the group :math:`O(3)`.


Irreps
------

If two tensors :math:`x` and :math:`y` transforms as :math:`D_x = 2 \times 1_o` (two vectors) and :math:`D_y = 0_e + 1_e` (a scalar and a pseudovector) respectively, where the indices :math:`e` and :math:`o` stand for even and odd -- the representation of parity,

.. jupyter-execute::
    :hide-code:

    import jax
    jax.numpy.set_printoptions(precision=3, suppress=True)

.. jupyter-execute::

    import jax
    import jax.numpy as jnp
    import e3nn_jax as e3nn

    irreps_x = e3nn.Irreps("2x1o")
    irreps_y = e3nn.Irreps("0e + 1e")

    x = irreps_x.randn(jax.random.PRNGKey(0), (-1,))
    y = irreps_y.randn(jax.random.PRNGKey(1), (-1,))

    irreps_x.dim, irreps_y.dim


their outer product is a :math:`6 \times 4` matrix of two indices :math:`A_{ij} = x_i y_j`.

.. jupyter-execute::

    A = jnp.einsum("i,j", x, y)
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


This representation is not irreducible (is reducible). It can be decomposed into irreps by a change of basis. The outerproduct followed by the change of basis is done by the class `e3nn_jax.full_tensor_product`.

.. jupyter-execute::

    tp = e3nn.full_tensor_product(e3nn.IrrepsArray(irreps_x, x), e3nn.IrrepsArray(irreps_y, y))
    tp


As a sanity check, we can verify that the representation of the tensor prodcut is block diagonal and of the same dimension.

.. jupyter-execute::

    D = tp.irreps.D_from_matrix(R)
    plt.imshow(D, cmap="bwr", vmin=-1, vmax=1);


`e3nn_jax.full_tensor_product` is a special case of `e3nn_jax.FunctionalTensorProduct`, other ones like `e3nn_jax.FullyConnectedTensorProduct` can contained weights what can be learned, very useful to create neural networks.


jax jit capabilities
--------------------

``e3nn-jax`` expoit the capabilities of ``jax.jit`` to optimize the code.
It can for instance get rid of the dead code:

.. jupyter-execute::
    :hide-code:

    def jit_code(f, *args, **kwargs):
        c = jax.xla_computation(f)(*args, **kwargs)
        backend = jax.lib.xla_bridge.get_backend()
        e = backend.compile(c)
        import jaxlib.xla_extension as xla_ext

        option = xla_ext.HloPrintOptions.fingerprint()
        option.print_operand_shape = False
        option.print_result_shape = False
        option.print_program_shape = True
        code = e.hlo_modules()[0].to_string(option)

        code = code.split("ENTRY")[1]
        code = code.split("\n}")[0]
        code = "\n".join(x[2:] for x in code.split("\n")[1:])

        return code

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

The irrep index is always the last index.

.. jupyter-execute::

    assert x.irreps.dim == x.shape[-1]
    x.shape

The list contains the data split into different arrays.

.. jupyter-execute::

    jax.tree_util.tree_map(lambda x: x.shape, x.list)

Here is the example of the tensor product of the two vectors.

.. jupyter-execute::

    out = e3nn.full_tensor_product(
        e3nn.IrrepsArray("1o", jnp.array([2.0, 0.0, 0.0])),
        e3nn.IrrepsArray("1o", jnp.array([0.0, 2.0, 0.0]))
    )
    out

The output is an `e3nn_jax.IrrepsArray` object and therefore also contains a ``list`` representation of the data.

.. jupyter-execute::

    out.list

The two fields `array` and `list` contain the same information under different forms.
This is not a performence issue, we rely on `jax.jit` to ignore the dead code.

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
        return e3nn.FullyConnectedTensorProduct("1e")(x1, x2)

Note that the inputs irreps are not yet specified, ``"1e"`` here specify the output.
Let's define two random inputs and initialize the parameters:

.. jupyter-execute::

    x1 = e3nn.IrrepsArray.randn("1e", jax.random.PRNGKey(0), (10,))
    x2 = e3nn.IrrepsArray.randn("1e", jax.random.PRNGKey(1), (10,))
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
    api/irreps_data
    api/spherical_harmonics
    api/tensor_product
    api/so3
    api/poly_envelope

.. _jax: https://github.com/google/jax
