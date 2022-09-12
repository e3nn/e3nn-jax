import e3nn_jax as e3nn
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
