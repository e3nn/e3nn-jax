import e3nn_jax as e3nn
import jax
import numpy as np


def test_antisymmetric_matrix():
    e3nn.reduced_tensor_product_basis("ij=-ji", i="5x0e + 1e")


def test_reduce_tensor_Levi_Civita_symbol():
    Q = e3nn.reduced_tensor_product_basis("ijk=-ikj=-jik", i="1e")
    assert Q.irreps == "0e"

    np.testing.assert_allclose(Q.array, -np.einsum("ijkx->ikjx", Q.array), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(Q.array, -np.einsum("ijkx->jikx", Q.array), atol=1e-6, rtol=1e-6)


def test_reduce_tensor_elasticity_tensor():
    Q = e3nn.reduced_tensor_product_basis("ijkl=jikl=klij", i="1e")
    assert Q.irreps.dim == 21


def test_reduce_tensor_elasticity_tensor_parity():
    Q = e3nn.reduced_tensor_product_basis("ijkl=jikl=klij", i="1o")
    assert Q.irreps.dim == 21
    assert all(ir.p == 1 for _, ir in Q.irreps)

    np.testing.assert_allclose(Q.array, np.einsum("ijklx->jiklx", Q.array), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(Q.array, np.einsum("ijklx->ijlkx", Q.array), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(Q.array, np.einsum("ijklx->klijx", Q.array), atol=1e-6, rtol=1e-6)


def test_reduced_symmetric_tensor_product_basis():
    Q = e3nn.reduced_symmetric_tensor_product_basis("1e", 5)
    P = e3nn.reduced_tensor_product_basis("ijklm=jiklm=jklmi", i="1e")
    np.testing.assert_equal(Q.array, P.array)


def test_tensor_product_basis():
    Q = e3nn.reduced_tensor_product_basis(["1e", "1o", "2e"])
    P = e3nn.reduced_tensor_product_basis("ijk", i="1e", j="1o", k="2e")
    np.testing.assert_equal(Q.array, P.array)


def test_tensor_product_basis_equivariance(keys):
    irreps = e3nn.Irreps("0e+1o+2e+0e")
    Q = e3nn.reduced_symmetric_tensor_product_basis(irreps, 3)

    jax.config.update("jax_enable_x64", True)  # to make precise rotation matrices
    q = e3nn.rand_quaternion(keys[0], ())
    Q1 = Q.transform_by_quaternion(q, k=1).array
    D = irreps.D_from_quaternion(q, k=1)
    Q2 = np.einsum("ijkz,iu,jv,kw->uvwz", Q.array, D, D, D)
    np.testing.assert_allclose(Q1, Q2, atol=1e-6, rtol=1e-6)
    jax.config.update("jax_enable_x64", False)
