import functools
import itertools
from typing import FrozenSet, List, Optional, Tuple

import numpy as np

import e3nn_jax as e3nn
from e3nn_jax import perm
from e3nn_jax.util.math_numpy import basis_intersection, round_to_sqrt_rational
from e3nn_jax.util import prod


def reduced_tensor_product_basis(
    formula: str,
    *,
    epsilon: float = 1e-5,
    **irreps,
):
    r"""Reduce a tensor product of multiple irreps subject to some permutation symmetry given by a formula.

    Args:
        formula (str): a formula of the form ``ijk=jik=ikj`` or ``ijk=-jki``.
            The left hand side is the original formula and the right hand side are the signed permutations.

        epsilon (float): the tolerance for the Gram-Schmidt orthogonalization. Default: ``1e-5``
        irreps (dict): the irreps of each index of the formula. For instance ``i="1x1o"``.

    Returns:
        IrrepsArray: The change of basis
            The shape is ``(d1, ..., dn, irreps.dim)``
            where ``di`` is the dimension of the index ``i`` and ``n`` is the number of indices in the formula.

    Example:
        >>> np.set_printoptions(precision=3, suppress=True)
        >>> reduced_tensor_product_basis("ij=-ji", i="1x1o")
        1x1e
        [[[ 0.     0.     0.   ]
          [ 0.     0.     0.707]
          [ 0.    -0.707  0.   ]]
        <BLANKLINE>
         [[ 0.     0.    -0.707]
          [ 0.     0.     0.   ]
          [ 0.707  0.     0.   ]]
        <BLANKLINE>
         [[ 0.     0.707  0.   ]
          [-0.707  0.     0.   ]
          [ 0.     0.     0.   ]]]
    """
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

    irreps = tuple(irreps[i] for i in f0)

    return _reduced_tensor_product_basis(irreps, formulas, epsilon)


@functools.lru_cache(maxsize=None)
def _reduced_tensor_product_basis(
    irreps: Tuple[e3nn.Irreps], formulas: FrozenSet[Tuple[int, Tuple[int, ...]]], epsilon: float
) -> np.ndarray:
    dims = tuple(irps.dim for irps in irreps)

    def _recursion(bases: List[Tuple[FrozenSet[int], e3nn.IrrepsArray]]) -> e3nn.IrrepsArray:
        if len(bases) == 1:
            f, b = bases[0]
            assert f == frozenset(range(len(irreps)))
            return b

        if len(bases) == 2:
            (fa, a) = bases[0]
            (fb, b) = bases[1]
            f = frozenset(fa | fb)
            ab = reduce_basis_product(a, b)
            p = reduce_subgroup_permutation(f, formulas, dims)
            ab = constrain_rotation_basis_by_permutation_basis(ab, p, epsilon=epsilon, round_fn=round_to_sqrt_rational)
            return ab

        # greedy algorithm
        min_p = np.inf
        best = None

        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                (fa, _) = bases[i]
                (fb, _) = bases[j]
                f = frozenset(fa | fb)
                p_dim = reduce_subgroup_permutation(f, formulas, dims, return_dim=True)
                if p_dim < min_p:
                    min_p = p_dim
                    best = (i, j, f)

        i, j, f = best
        del bases[j]
        del bases[i]
        sub_irreps = tuple(irreps[i] for i in f)
        sub_formulas = sub_formula_fn(f, formulas)
        ab = _reduced_tensor_product_basis(sub_irreps, sub_formulas, epsilon)
        ab = ab.reshape(tuple(dims[i] if i in f else 1 for i in range(len(dims))) + (-1,))
        return _recursion([(f, ab)] + bases)

    initial_bases = [
        e3nn.IrrepsArray(
            irps,
            np.reshape(np.eye(irps.dim), (1,) * i + (irps.dim,) + (1,) * (len(irreps) - i - 1) + (irps.dim,)),
        )
        for i, irps in enumerate(irreps)
    ]
    return _recursion([(frozenset({i}), base) for i, base in enumerate(initial_bases)])


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


def constrain_rotation_basis_by_permutation_basis(
    rotation_basis: e3nn.IrrepsArray, permutation_basis: np.ndarray, *, epsilon=1e-5, round_fn=lambda x: x
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


def sub_formula_fn(
    sub_f0: FrozenSet[int], formulas: FrozenSet[Tuple[int, Tuple[int, ...]]]
) -> FrozenSet[Tuple[int, Tuple[int, ...]]]:
    sor = sorted(sub_f0)
    return frozenset(
        {
            (s, tuple(sor.index(i) for i in p if i in sub_f0))
            for s, p in formulas
            if all(i in sub_f0 or i == j for j, i in enumerate(p))
        }
    )


def reduce_subgroup_permutation(
    sub_f0: FrozenSet[int], formulas: FrozenSet[Tuple[int, Tuple[int, ...]]], dims: Tuple[int, ...], return_dim: bool = False
) -> np.ndarray:
    sub_formulas = sub_formula_fn(sub_f0, formulas)
    sub_dims = tuple(dims[i] for i in sub_f0)
    base = reduce_permutation_base(sub_formulas, sub_dims)
    if return_dim:
        return len(base)
    permutation_basis = reduce_permutation_matrix(base, sub_dims)
    return np.reshape(permutation_basis, (-1,) + tuple(dims[i] if i in sub_f0 else 1 for i in range(len(dims))))


@functools.lru_cache(maxsize=None)
def full_base_fn(dims: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    return list(itertools.product(*(range(d) for d in dims)))


@functools.lru_cache(maxsize=None)
def reduce_permutation_base(
    formulas: FrozenSet[Tuple[int, Tuple[int, ...]]], dims: Tuple[int, ...]
) -> FrozenSet[FrozenSet[FrozenSet[Tuple[int, Tuple[int, ...]]]]]:
    full_base = full_base_fn(dims)  # (0, 0, 0), (0, 0, 1), (0, 0, 2), ... (3, 3, 3)
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

    return frozenset(base)


@functools.lru_cache(maxsize=None)
def reduce_permutation_matrix(
    base: FrozenSet[FrozenSet[FrozenSet[Tuple[int, Tuple[int, ...]]]]], dims: Tuple[int, ...]
) -> np.ndarray:
    base = sorted(
        [sorted([sorted(xs) for xs in x]) for x in base]
    )  # requested for python 3.7 but not for 3.8 (probably a bug in 3.7)

    # First we compute the change of basis (projection) between full_base and base
    d_sym = len(base)
    Q = np.zeros((d_sym, prod(dims)))

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
