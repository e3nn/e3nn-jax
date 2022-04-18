import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np

from ._su2 import su2_clebsch_gordan, su2_generators


def naive_broadcast_decorator(func):
    def wrapper(*args):
        args = [jnp.asarray(a) for a in args]
        shape = jnp.broadcast_shapes(*(arg.shape for arg in args))
        args = [jnp.broadcast_to(arg, shape) for arg in args]
        f = func
        for _ in range(len(shape)):
            f = jax.vmap(f)
        return f(*args)

    return wrapper


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
    return (-1j) ** l * q  # Added factor of 1j**l to make the Clebsch-Gordan coefficients real


def clebsch_gordan(l1: int, l2: int, l3: int) -> np.ndarray:
    C = su2_clebsch_gordan(l1, l2, l3)
    Q1 = change_basis_real_to_complex(l1)
    Q2 = change_basis_real_to_complex(l2)
    Q3 = change_basis_real_to_complex(l3)
    C = np.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, np.conj(Q3.T), C)

    assert np.all(np.abs(np.imag(C)) < 1e-5)
    return np.real(C)


def generators(l: int) -> np.ndarray:
    X = su2_generators(l)
    Q = change_basis_real_to_complex(l)
    X = np.conj(Q.T) @ X @ Q

    assert np.all(np.abs(np.imag(X)) < 1e-5)
    return np.real(X)


def wigner_D(l: int, alpha: jnp.ndarray, beta: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
    alpha = alpha % (2 * jnp.pi)
    beta = beta % (2 * jnp.pi)
    gamma = gamma % (2 * jnp.pi)
    X = generators(l)

    def f(a, b, c):
        return jax.scipy.linalg.expm(a * X[1]) @ jax.scipy.linalg.expm(b * X[0]) @ jax.scipy.linalg.expm(c * X[1])

    return naive_broadcast_decorator(f)(alpha, beta, gamma)
