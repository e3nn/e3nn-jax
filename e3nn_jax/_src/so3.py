import numpy as np

from e3nn_jax._src.su2 import su2_clebsch_gordan, su2_generators


def change_basis_real_to_complex(l: int) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    q = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / np.sqrt(2)
        q[l + m, l - abs(m)] = -1j / np.sqrt(2)
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / np.sqrt(2)
        q[l + m, l - abs(m)] = 1j * (-1) ** m / np.sqrt(2)
    return (
        -1j
    ) ** l * q  # Added factor of 1j**l to make the Clebsch-Gordan coefficients real


def clebsch_gordan(l1: int, l2: int, l3: int) -> np.ndarray:
    r"""The Clebsch-Gordan coefficients of the real irreducible representations of :math:`SO(3)`.

    Args:
        l1 (int): the representation order of the first irrep
        l2 (int): the representation order of the second irrep
        l3 (int): the representation order of the third irrep

    Returns:
        np.ndarray: the Clebsch-Gordan coefficients
    """
    C = su2_clebsch_gordan(l1, l2, l3)
    Q1 = change_basis_real_to_complex(l1)
    Q2 = change_basis_real_to_complex(l2)
    Q3 = change_basis_real_to_complex(l3)
    C = np.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, np.conj(Q3.T), C)

    assert np.all(np.abs(np.imag(C)) < 1e-5)
    return np.real(C)


def generators(l: int) -> np.ndarray:
    r"""The generators of the real irreducible representations of :math:`SO(3)`.

    Args:
        l (int): the representation order of the irrep

    Returns:
        np.ndarray: the generators
    """
    X = su2_generators(l)
    Q = change_basis_real_to_complex(l)
    X = np.conj(Q.T) @ X @ Q

    assert np.all(np.abs(np.imag(X)) < 1e-5)
    return np.real(X)
