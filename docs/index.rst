Euclidean neural networks
=========================

What is ``e3nn-jax``?
-----------------

``e3nn-jax`` is a python library based on jax_ to create equivariant neural networks for the group :math:`O(3)`.


Demonstration
-------------

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

    tp = e3nn.full_tensor_product(e3nn.IrrepsData.from_contiguous(irreps_x, x), e3nn.IrrepsData.from_contiguous(irreps_y, y))
    tp


As a sanity check, we can verify that the representation of the tensor prodcut is block diagonal and of the same dimension.

.. jupyter-execute::

    D = tp.irreps.D_from_matrix(R)
    plt.imshow(D, cmap="bwr", vmin=-1, vmax=1);


`e3nn_jax.full_tensor_product` is a special case of `e3nn_jax.FunctionalTensorProduct`, other ones like `e3nn_jax.FullyConnectedTensorProduct` can contained weights what can be learned, very useful to create neural networks.


.. _jax: https://github.com/google/jax


.. autoclass:: e3nn_jax.Irreps
    :members:


.. autoclass:: e3nn_jax.IrrepsData
    :members:


.. autofunction:: e3nn_jax.full_tensor_product


.. autoclass:: e3nn_jax.FunctionalTensorProduct
    :members:


.. autofunction:: e3nn_jax.spherical_harmonics
