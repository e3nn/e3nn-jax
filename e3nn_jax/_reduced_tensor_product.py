import functools
import itertools
from typing import FrozenSet, List, Optional, Tuple

import numpy as np

import e3nn_jax as e3nn
from e3nn_jax import perm


def reduce_basis_product(
    basis1: e3nn.IrrepsArray,
    basis2: e3nn.IrrepsArray,
    filter_ir_out: Optional[List[e3nn.Irrep]] = None,
) -> e3nn.IrrepsArray:
    """Reduce the product of two basis"""
    new_irreps: List[Tuple[int, e3nn.Irrep]] = []
    new_list = []

    for (mul1, ir1), x1 in zip(basis1.irreps, basis1.list):
        for (mul2, ir2), x2 in zip(basis2.irreps, basis2.list):
            for ir in ir1 * ir2:
                if filter_ir_out is not None and ir not in filter_ir_out:
                    continue

                x = np.einsum(
                    "...ui,...vj,ijk->...uvk",
                    x1,
                    x2,
                    np.sqrt(ir.dim) * e3nn.clebsch_gordan(ir1.l, ir2.l, ir.l),
                )
                x = np.reshape(x, x.shape[:-3] + (mul1 * mul2, ir.dim))
                new_irreps.append((mul1 * mul2, ir))
                new_list.append(x)

    new = e3nn.IrrepsArray.from_list(new_irreps, new_list, np.broadcast_shapes(basis1.shape[:-1], basis2.shape[:-1]))
    return new.sorted().simplify()


def normalize_integer_ratio(n, d):
    g = np.gcd(n, d)
    g = np.where(d < 0, -g, g)
    return n // g, d // g


def _as_approx_integer_ratio(x):
    # only for 0 <= x <= 1
    big = 1 << 52 - 1  # mantissa is 52 bits

    n = np.floor(x * big).astype(np.int64)
    with np.errstate(invalid="ignore"):
        d = np.round(n / x).astype(np.int64)
    d = np.where(n == 0, np.ones(d.shape, dtype=np.int64), d)
    return n, d


def as_approx_integer_ratio(x):
    assert x.dtype == np.float64
    sign = np.sign(x).astype(np.int64)
    x = np.abs(x)

    with np.errstate(divide="ignore", over="ignore"):
        n, d = np.where(
            x <= 1,
            _as_approx_integer_ratio(x),
            _as_approx_integer_ratio(1 / x)[::-1],
        )
    return normalize_integer_ratio(sign * n, d)


def limit_denominator(n, d, max_denominator=1_000_000):
    # (n, d) = must be normalized
    n0, d0 = n, d
    p0, q0, p1, q1 = np.zeros_like(n), np.ones_like(n), np.ones_like(n), np.zeros_like(n)
    while True:
        a = n // d
        q2 = q0 + a * q1
        stop = (q2 > max_denominator) | (d0 <= max_denominator)
        if np.all(stop):
            break
        p0, q0, p1, q1 = np.where(stop, (p0, q0, p1, q1), (p1, q1, p0 + a * p1, q2))
        n, d = np.where(stop, (n, d), (d, n - a * d))

    with np.errstate(divide="ignore"):
        k = (max_denominator - q0) // q1
    n1, d1 = p0 + k * p1, q0 + k * q1
    n2, d2 = p1, q1
    with np.errstate(over="ignore"):
        mask = np.abs(d1 * (n2 * d0 - n0 * d2)) <= np.abs(d2 * (n1 * d0 - n0 * d1))
    return np.where(
        d0 < max_denominator,
        (n0, d0),
        np.where(mask, (n2, d2), (n1, d1)),
    )


def _round_to_sqrt_rational(x, max_denominator):
    sign = np.sign(x)
    n, d = as_approx_integer_ratio(x**2)
    n, d = limit_denominator(n, d, max_denominator**2 + 1)
    return sign * np.sqrt(n / d)


def round_to_sqrt_rational(x: np.ndarray, max_denominator=4096) -> np.ndarray:
    x = np.array(x)
    if np.iscomplex(x).any():
        return _round_to_sqrt_rational(np.real(x), max_denominator) + 1j * _round_to_sqrt_rational(np.imag(x), max_denominator)
    return _round_to_sqrt_rational(np.real(x), max_denominator)


def gram_schmidt(A: np.ndarray, *, epsilon=1e-4, round_fn=lambda x: x) -> np.ndarray:
    """
    Orthogonalize a matrix using the Gram-Schmidt process.
    """
    assert A.ndim == 2, "Gram-Schmidt process only works for matrices."
    assert A.dtype in [np.float64, np.complex128], "Gram-Schmidt process only works for float64 matrices."
    Q = []
    for i in range(A.shape[0]):
        v = A[i]
        for w in Q:
            v -= np.dot(np.conj(w), v) * w
        norm = np.linalg.norm(v)
        if norm > epsilon:
            v = round_fn(v / norm)
            Q += [v]
    return np.stack(Q) if len(Q) > 0 else np.empty((0, A.shape[1]))


def basis_intersection(basis1: np.ndarray, basis2: np.ndarray, *, epsilon=1e-4, round_fn=lambda x: x) -> np.ndarray:
    """Compute the intersection of two bases

    Args:
        basis1 (np.ndarray): A basis
        basis2 (np.ndarray): Another basis
        epsilon (float, optional): Tolerance for the norm of the vectors. Defaults to 1e-4.
        round_fn (function, optional): Function to round the vectors. Defaults to lambda x: x.

    Returns:
        np.ndarray: A projection matrix that projects vectors of the first basis in the intersection of the two bases.
        np.ndarray: A projection matrix that projects vectors of the second basis in the intersection of the two bases.

    Example:
        >>> basis1 = np.array([[1, 0, 0], [0, 0, 1.0]])
        >>> basis2 = np.array([[1, 1, 0], [0, 1, 0.0]])
        >>> P1, P2 = basis_intersection(basis1, basis2)
        >>> P1 @ basis1
        array([[1., 0., 0.]])
    """
    assert basis1.ndim == 2
    assert basis2.ndim == 2
    assert basis1.shape[1] == basis2.shape[1]

    p = np.concatenate(
        [
            np.concatenate([basis1 @ basis1.T, -basis1 @ basis2.T], axis=1),
            np.concatenate([-basis2 @ basis1.T, basis2 @ basis2.T], axis=1),
        ],
        axis=0,
    )
    p = round_fn(p)

    w, v = np.linalg.eigh(p)
    v = v[:, w < epsilon]

    x1 = v[: basis1.shape[0], :]
    x1 = gram_schmidt(x1 @ x1.T, epsilon=epsilon, round_fn=round_fn)

    x2 = v[basis1.shape[0] :, :]
    x2 = gram_schmidt(x2 @ x2.T, epsilon=epsilon, round_fn=round_fn)
    return x1, x2


def constrain_rotation_basis_by_permutation_basis(
    rotation_basis: e3nn.IrrepsArray, permutation_basis: np.ndarray, *, epsilon=1e-4, round_fn=lambda x: x
) -> e3nn.IrrepsArray:
    """Constrain a rotation basis by a permutation basis.

    Args:
        rotation_basis (e3nn.IrrepsArray): A rotation basis
        permutation_basis (np.ndarray): A permutation basis

    Returns:
        e3nn.IrrepsArray: A rotation basis that is constrained by the permutation basis.
    """
    assert rotation_basis.shape[:-1] == permutation_basis.shape[1:]

    perm = np.reshape(permutation_basis, (permutation_basis.shape[0], -1))  # (free, dim)

    new_irreps: List[Tuple[int, e3nn.Irrep]] = []
    new_list: List[np.ndarray] = []

    for (mul, ir), rot_basis in zip(rotation_basis.irreps, rotation_basis.list):
        R = rot_basis[..., 0]
        R = np.reshape(R, (-1, mul)).T  # (mul, dim)
        P, _ = basis_intersection(R, perm, epsilon=epsilon, round_fn=round_fn)

        if P.shape[0] > 0:
            new_irreps.append((P.shape[0], ir))
            new_list.append(np.einsum("vu,...ui->...vi", P, rot_basis))

    return e3nn.IrrepsArray.from_list(new_irreps, new_list, rotation_basis.shape[:-1])


@functools.lru_cache(maxsize=None)
def germinate_formulas(formula: str) -> Tuple[str, FrozenSet[Tuple[int, Tuple[int, ...]]]]:
    formulas = [(-1 if f.startswith("-") else 1, f.replace("-", "")) for f in formula.split("=")]
    s0, f0 = formulas[0]
    assert s0 == 1

    for _s, f in formulas:
        if len(set(f)) != len(f) or set(f) != set(f0):
            raise RuntimeError(f"{f} is not a permutation of {f0}")
        if len(f0) != len(f):
            raise RuntimeError(f"{f0} and {f} don't have the same number of indices")

    # `formulas` is a list of (sign, permutation of indices)
    # each formula can be viewed as a permutation of the original formula
    formulas = {(s, tuple(f.index(i) for i in f0)) for s, f in formulas}  # set of generators (permutations)

    # they can be composed, for instance if you have ijk=jik=ikj
    # you also have ijk=jki
    # applying all possible compositions creates an entire group
    while True:
        n = len(formulas)
        formulas = formulas.union([(s, perm.inverse(p)) for s, p in formulas])
        formulas = formulas.union([(s1 * s2, perm.compose(p1, p2)) for s1, p1 in formulas for s2, p2 in formulas])
        if len(formulas) == n:
            break  # we break when the set is stable => it is now a group \o/

    return f0, frozenset(formulas)


def subgroup_formulas(
    sub_f0: str, f0: str, formulas: FrozenSet[Tuple[int, Tuple[int, ...]]]
) -> FrozenSet[Tuple[int, Tuple[int, ...]]]:
    return frozenset(
        {
            (s, tuple(sub_f0.index(f0[i]) for i in p if f0[i] in sub_f0))
            for s, p in formulas
            if all(f0[i] in sub_f0 or i == j for j, i in enumerate(p))
        }
    )


def reduce_permutation(f0: str, formulas: FrozenSet[Tuple[int, Tuple[int, ...]]], **dims) -> np.ndarray:
    # here we check that each index has one and only one dimension
    for _s, p in formulas:
        f = "".join(f0[i] for i in p)
        for i, j in zip(f0, f):
            if i in dims and j in dims and dims[i] != dims[j]:
                raise RuntimeError(f"dimension of {i} and {j} should be the same")
            if i in dims:
                dims[j] = dims[i]
            if j in dims:
                dims[i] = dims[j]

    for i in f0:
        if i not in dims:
            raise RuntimeError(f"index {i} has no dimension associated to it")

    dims = tuple(dims[i] for i in f0)
    return _reduce_permutation(formulas, dims)


@functools.lru_cache(maxsize=None)
def _reduce_permutation(formulas: FrozenSet[Tuple[int, Tuple[int, ...]]], dims: Tuple[int, ...]) -> np.ndarray:
    full_base = list(itertools.product(*(range(d) for d in dims)))  # (0, 0, 0), (0, 0, 1), (0, 0, 2), ... (3, 3, 3)
    # len(full_base) degrees of freedom in an unconstrained tensor

    # but there is constraints given by the group `formulas`
    # For instance if `ij=-ji`, then 00=-00, 01=-01 and so on
    base = set()
    for x in full_base:
        # T[x] is a coefficient of the tensor T and is related to other coefficient T[y]
        # if x and y are related by a formula
        xs = {(s, tuple(x[i] for i in p)) for s, p in formulas}
        # s * T[x] are all equal for all (s, x) in xs
        # if T[x] = -T[x] it is then equal to 0 and we lose this degree of freedom
        if not (-1, x) in xs:
            # the sign is arbitrary, put both possibilities
            base.add(frozenset({frozenset(xs), frozenset({(-s, x) for s, x in xs})}))

    # len(base) is the number of degrees of freedom in the tensor.

    base = sorted(
        [sorted([sorted(xs) for xs in x]) for x in base]
    )  # requested for python 3.7 but not for 3.8 (probably a bug in 3.7)

    # First we compute the change of basis (projection) between full_base and base
    d_sym = len(base)
    Q = np.zeros((d_sym, len(full_base)))

    for i, x in enumerate(base):
        x = max(x, key=lambda xs: sum(s for s, x in xs))
        for s, e in x:
            j = 0
            for k, d in zip(e, dims):
                j *= d
                j += k
            Q[i, j] = s / len(x) ** 0.5

    np.testing.assert_allclose(Q @ Q.T, np.eye(d_sym))

    return Q.reshape(d_sym, *dims)


def reduce_subgroup_permutation(sub_f0: str, f0: str, formulas: FrozenSet[Tuple[int, Tuple[int, ...]]], **dims) -> np.ndarray:
    sub_formulas = subgroup_formulas(sub_f0, f0, formulas)
    permutation_basis = reduce_permutation(sub_f0, sub_formulas, **dims)
    return np.reshape(permutation_basis, (-1,) + tuple(dims[f0[i]] if f0[i] in sub_f0 else 1 for i in range(len(f0))))


def reduced_tensor_product_basis(
    formula: str,
    *,
    epsilon: float = 1e-9,
    **irreps,
):
    f0, formulas = germinate_formulas(formula)

    irreps = {i: e3nn.Irreps(irs) for i, irs in irreps.items()}

    for i in irreps:
        if len(i) != 1:
            raise TypeError(f"got an unexpected keyword argument '{i}'")

    for _sign, p in formulas:
        f = "".join(f0[i] for i in p)
        for i, j in zip(f0, f):
            if i in irreps and j in irreps and irreps[i] != irreps[j]:
                raise RuntimeError(f"irreps of {i} and {j} should be the same")
            if i in irreps:
                irreps[j] = irreps[i]
            if j in irreps:
                irreps[i] = irreps[j]

    for i in f0:
        if i not in irreps:
            raise RuntimeError(f"index {i} has no irreps associated to it")

    for i in irreps:
        if i not in f0:
            raise RuntimeError(f"index {i} has an irreps but does not appear in the fomula")

    def _recursion(bases):
        if len(bases) == 1:
            f, b = bases[0]
            assert f == f0
            return b

        # greedy algorithm
        min_p = np.inf
        best = None

        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                (fa, a) = bases[i]
                (fb, b) = bases[j]
                f = "".join(sorted(fa + fb, key=lambda i: f0.index(i)))
                p = reduce_subgroup_permutation(f, f0, formulas, **{i: irs.dim for i, irs in irreps.items()})
                if p.shape[0] < min_p:
                    min_p = p.shape[0]
                    best = (i, j, a, b, f, p)

        i, j, a, b, f, p = best
        del bases[j]
        del bases[i]
        ab = reduce_basis_product(a, b)
        ab = constrain_rotation_basis_by_permutation_basis(ab, p, epsilon=epsilon, round_fn=round_to_sqrt_rational)
        return _recursion([(f, ab)] + bases)

    initial_bases = [
        e3nn.IrrepsArray(
            irreps,
            np.reshape(np.eye(irreps.dim), (1,) * i + (irreps.dim,) + (1,) * (len(f0) - i - 1) + (irreps.dim,)),
        )
        for i, irreps in ((i, irreps[f0[i]]) for i in range(len(f0)))
    ]
    return _recursion(list(zip(f0, initial_bases)))
