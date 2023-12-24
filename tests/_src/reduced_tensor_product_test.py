import e3nn_jax as e3nn
import jax
import numpy as np


def test_antisymmetric_matrix():
    e3nn.reduced_tensor_product_basis("ij=-ji", i="5x0e + 1e")


def test_reduce_tensor_Levi_Civita_symbol():
    Q = e3nn.reduced_tensor_product_basis("ijk=-ikj=-jik", i="1e")
    assert Q.irreps == "0e"

    np.testing.assert_allclose(
        Q.array, -np.einsum("ijkx->ikjx", Q.array), atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        Q.array, -np.einsum("ijkx->jikx", Q.array), atol=1e-6, rtol=1e-6
    )


def test_reduce_tensor_elasticity_tensor():
    Q = e3nn.reduced_tensor_product_basis("ijkl=jikl=klij", i="1e")
    assert Q.irreps.dim == 21


def test_reduce_tensor_elasticity_tensor_parity():
    Q = e3nn.reduced_tensor_product_basis("ijkl=jikl=klij", i="1o")
    assert Q.irreps.dim == 21
    assert all(ir.p == 1 for _, ir in Q.irreps)

    np.testing.assert_allclose(
        Q.array, np.einsum("ijklx->jiklx", Q.array), atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        Q.array, np.einsum("ijklx->ijlkx", Q.array), atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        Q.array, np.einsum("ijklx->klijx", Q.array), atol=1e-6, rtol=1e-6
    )


def test_reduced_symmetric_tensor_product_basis():
    Q = e3nn.reduced_symmetric_tensor_product_basis("1e", 5)
    P = e3nn.reduced_tensor_product_basis("ijklm=jiklm=jklmi", i="1e")
    np.testing.assert_equal(Q.array, P.array)


def test_tensor_product_basis():
    Q = e3nn.reduced_tensor_product_basis(["1e", "1o", "2e"])
    P = e3nn.reduced_tensor_product_basis("ijk", i="1e", j="1o", k="2e")
    np.testing.assert_equal(Q.array, P.array)


def test_tensor_product_basis_equivariance(keys):
    irreps_i = e3nn.Irreps("1o+2e+2x0e")
    irreps_k = e3nn.Irreps("1o")
    Q = e3nn.reduced_tensor_product_basis("ijk=jik", i=irreps_i, k=irreps_k)

    jax.config.update("jax_enable_x64", True)  # to make precise rotation matrices
    q = e3nn.rand_quaternion(keys[0], (), dtype=np.float64)
    Q1 = Q.transform_by_quaternion(q, k=1).array

    Di = irreps_i.D_from_quaternion(q, k=1)
    Dj = Di
    Dk = irreps_k.D_from_quaternion(q, k=1)
    Q2 = np.einsum("ijkz,iu,jv,kw->uvwz", Q.array, Di, Dj, Dk)

    np.testing.assert_allclose(Q1, Q2, atol=1e-6, rtol=1e-6)
    jax.config.update("jax_enable_x64", False)


def test_optimized_reduced_symmetric_tensor_product_basis_order_2():
    Q = e3nn.reduced_symmetric_tensor_product_basis(
        "1e + 2o", 2, _use_optimized_implementation=True
    )
    P = e3nn.reduced_symmetric_tensor_product_basis(
        "1e + 2o", 2, _use_optimized_implementation=False
    )
    assert Q.irreps == P.irreps
    np.testing.assert_almost_equal(Q.array, P.array)

    Q = Q.array.reshape(-1, Q.irreps.dim)
    P = P.array.reshape(-1, P.irreps.dim)
    np.testing.assert_allclose(Q @ Q.T, P @ P.T, atol=1e-6, rtol=1e-6)


def test_optimized_reduced_symmetric_tensor_product_basis_order_3a():
    Q = e3nn.reduced_symmetric_tensor_product_basis(
        "0e + 1e", 3, _use_optimized_implementation=True
    )
    P = e3nn.reduced_symmetric_tensor_product_basis(
        "0e + 1e", 3, _use_optimized_implementation=False
    )
    assert Q.irreps == P.irreps

    Q = Q.array.reshape(-1, Q.irreps.dim)
    P = P.array.reshape(-1, P.irreps.dim)
    np.testing.assert_allclose(Q @ Q.T, P @ P.T, atol=1e-6, rtol=1e-6)


def test_optimized_reduced_symmetric_tensor_product_basis_order_3b():
    Q = e3nn.reduced_symmetric_tensor_product_basis(
        "3x0e + 1e", 3, _use_optimized_implementation=True
    )
    P = e3nn.reduced_symmetric_tensor_product_basis(
        "3x0e + 1e", 3, _use_optimized_implementation=False
    )
    assert Q.irreps == P.irreps

    Q = Q.array.reshape(-1, Q.irreps.dim)
    P = P.array.reshape(-1, P.irreps.dim)
    np.testing.assert_allclose(Q @ Q.T, P @ P.T, atol=1e-6, rtol=1e-6)


def test_optimized_reduced_symmetric_tensor_product_basis_order_3c():
    irreps = "1o + 2e + 4e"
    Q = e3nn.reduced_symmetric_tensor_product_basis(
        irreps, 3, keep_ir="0e + 1o", _use_optimized_implementation=True
    )
    P = e3nn.reduced_symmetric_tensor_product_basis(
        irreps, 3, keep_ir="0e + 1o", _use_optimized_implementation=False
    )
    assert Q.irreps == P.irreps

    Q = Q.array.reshape(-1, Q.irreps.dim)
    P = P.array.reshape(-1, P.irreps.dim)
    np.testing.assert_allclose(Q @ Q.T, P @ P.T, atol=1e-6, rtol=1e-6)


def test_optimized_reduced_symmetric_tensor_product_basis_order_4():
    Q = e3nn.reduced_symmetric_tensor_product_basis(
        "1e + 2o", 4, _use_optimized_implementation=True
    )
    P = e3nn.reduced_symmetric_tensor_product_basis(
        "1e + 2o", 4, _use_optimized_implementation=False
    )
    assert Q.irreps == P.irreps

    Q = Q.array.reshape(-1, Q.irreps.dim)
    P = P.array.reshape(-1, P.irreps.dim)
    np.testing.assert_allclose(Q @ Q.T, P @ P.T, atol=1e-6, rtol=1e-6)


def test_symmetric_tensor_product_basis_1():
    Q = e3nn.reduced_symmetric_tensor_product_basis("0e + 1e", 3)
    P = e3nn.reduced_tensor_product_basis("ijk=jki=ikj", i="0e + 1e")
    assert Q.irreps == P.irreps

    Q = Q.array.reshape(-1, Q.irreps.dim)
    P = P.array.reshape(-1, P.irreps.dim)
    np.testing.assert_equal(Q @ Q.T, P @ P.T)


def test_trivial_case_1():
    Q = e3nn.reduced_symmetric_tensor_product_basis("0e", 3)
    assert Q.irreps == "0e"
    np.testing.assert_equal(Q.array, np.ones((1, 1, 1, 1)))


def test_trivial_case_2():
    Q = e3nn.reduced_symmetric_tensor_product_basis("0e + 1e + 2e", 1)
    assert Q.irreps == "0e + 1e + 2e"
    np.testing.assert_equal(Q.array, np.eye(Q.irreps.dim))
