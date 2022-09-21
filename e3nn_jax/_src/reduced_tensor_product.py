# Partially based on https://github.com/songk42/ReducedTensorProduct.jl
import functools
import itertools
from typing import FrozenSet, List, Optional, Tuple, Union

import e3nn_jax as e3nn
import numpy as np
from e3nn_jax import perm
from e3nn_jax._src.util.math_numpy import basis_intersection, round_to_sqrt_rational
from e3nn_jax._src.util.prod import prod


def reduced_tensor_product_basis(
    formula_or_irreps_list: Union[str, List[e3nn.Irreps]],
    *,
    epsilon: float = 1e-5,
    keep_ir: Optional[List[e3nn.Irrep]] = None,
    **irreps_dict,
) -> e3nn.IrrepsArray:
    r"""Reduce a tensor product of multiple irreps subject to some permutation symmetry given by a formula.

    Args:
        formula_or_irreps_list (str or list of Irreps): a formula of the form ``ijk=jik=ikj`` or ``ijk=-jki``.
            The left hand side is the original formula and the right hand side are the signed permutations.
            If no index symmetry is present, a list of irreps can be given instead.

        epsilon (float): the tolerance for the Gram-Schmidt orthogonalization. Default: ``1e-5``
        irreps_dict (dict): the irreps of each index of the formula. For instance ``i="1x1o"``.

    Returns:
        IrrepsArray: The change of basis
            The shape is ``(d1, ..., dn, irreps_out.dim)``
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
    if keep_ir is not None:
        keep_ir = frozenset(e3nn.Irrep(ir) for ir in keep_ir)

    if isinstance(formula_or_irreps_list, (tuple, list)):
        irreps_list = formula_or_irreps_list
        irreps_tuple = tuple(e3nn.Irreps(irreps) for irreps in irreps_list)
        formulas: FrozenSet[Tuple[int, Tuple[int, ...]]] = frozenset({(1, tuple(range(len(irreps_tuple))))})
        return _reduced_tensor_product_basis(irreps_tuple, formulas, keep_ir, epsilon)

    formula = formula_or_irreps_list
    f0, perm_repr = germinate_perm_repr(formula)

    irreps_dict = {i: e3nn.Irreps(irs) for i, irs in irreps_dict.items()}

    for i in irreps_dict:
        if len(i) != 1:
            raise TypeError(f"got an unexpected keyword argument '{i}'")

    for _sign, p in perm_repr:
        f = "".join(f0[i] for i in p)
        for i, j in zip(f0, f):
            if i in irreps_dict and j in irreps_dict and irreps_dict[i] != irreps_dict[j]:
                raise RuntimeError(f"irreps of {i} and {j} should be the same")
            if i in irreps_dict:
                irreps_dict[j] = irreps_dict[i]
            if j in irreps_dict:
                irreps_dict[i] = irreps_dict[j]

    for i in f0:
        if i not in irreps_dict:
            raise RuntimeError(f"index {i} has no irreps associated to it")

    for i in irreps_dict:
        if i not in f0:
            raise RuntimeError(f"index {i} has an irreps but does not appear in the fomula")

    irreps_tuple = tuple(irreps_dict[i] for i in f0)

    return _reduced_tensor_product_basis(irreps_tuple, perm_repr, keep_ir, epsilon)


def reduced_symmetric_tensor_product_basis(
    irreps: e3nn.Irreps,
    order: int,
    *,
    keep_ir: Optional[List[e3nn.Irrep]] = None,
    epsilon: float = 1e-5,
) -> e3nn.IrrepsArray:
    r"""Reduce a symmetric tensor product.

    Args:
        irreps (Irreps): the irreps of each index.
        order (int): the order of the tensor product. i.e. the number of indices.

    Returns:
        IrrepsArray: The change of basis
            The shape is ``(d, ..., d, irreps_out.dim)``
            where ``d`` is the dimension of ``irreps``.
    """
    if keep_ir is not None:
        keep_ir = frozenset(e3nn.Irrep(ir) for ir in keep_ir)

    irreps = e3nn.Irreps(irreps)
    perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]] = frozenset((1, p) for p in itertools.permutations(range(order)))
    return _reduced_tensor_product_basis(tuple([irreps] * order), perm_repr, keep_ir, epsilon)


@functools.lru_cache(maxsize=None)
def _reduced_tensor_product_basis(
    irreps_tuple: Tuple[e3nn.Irreps],
    perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]],
    keep_ir: Optional[FrozenSet[e3nn.Irrep]],
    epsilon: float,
) -> e3nn.IrrepsArray:
    dims = tuple(irps.dim for irps in irreps_tuple)

    def _recursion(bases: List[Tuple[FrozenSet[int], e3nn.IrrepsArray]]) -> e3nn.IrrepsArray:
        if len(bases) == 1:
            f, b = bases[0]
            assert f == frozenset(range(len(irreps_tuple)))
            return b if keep_ir is None else b.filtered(keep_ir)

        if len(bases) == 2:
            (fa, a) = bases[0]
            (fb, b) = bases[1]
            f = frozenset(fa | fb)
            ab = reduce_basis_product(a, b, keep_ir)
            if len(subrepr_permutation(f, perm_repr)) == 1:
                return ab
            p = reduce_subgroup_permutation(f, perm_repr, dims)
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
                p_dim = reduce_subgroup_permutation(f, perm_repr, dims, return_dim=True)
                if p_dim < min_p:
                    min_p = p_dim
                    best = (i, j, f)

        i, j, f = best
        del bases[j]
        del bases[i]
        sub_irreps = tuple(irreps_tuple[i] for i in f)
        sub_perm_repr = subrepr_permutation(f, perm_repr)
        ab = _reduced_tensor_product_basis(sub_irreps, sub_perm_repr, None, epsilon)
        ab = ab.reshape(tuple(dims[i] if i in f else 1 for i in range(len(dims))) + (-1,))
        return _recursion([(f, ab)] + bases)

    initial_bases = [
        e3nn.IrrepsArray(
            irps,
            np.reshape(np.eye(irps.dim), (1,) * i + (irps.dim,) + (1,) * (len(irreps_tuple) - i - 1) + (irps.dim,)),
        )
        for i, irps in enumerate(irreps_tuple)
    ]
    return _recursion([(frozenset({i}), base) for i, base in enumerate(initial_bases)])


@functools.lru_cache(maxsize=None)
def germinate_perm_repr(formula: str) -> Tuple[str, FrozenSet[Tuple[int, Tuple[int, ...]]]]:
    """Convert the formula (generators) into a group."""
    formulas = [(-1 if f.startswith("-") else 1, f.replace("-", "")) for f in formula.split("=")]
    s0, f0 = formulas[0]
    assert s0 == 1

    for _s, f in formulas:
        if len(set(f)) != len(f) or set(f) != set(f0):
            raise RuntimeError(f"{f} is not a permutation of {f0}")
        if len(f0) != len(f):
            raise RuntimeError(f"{f0} and {f} don't have the same number of indices")

    # `perm_repr` is a list of (sign, permutation of indices)
    # each formula can be viewed as a permutation of the original formula
    perm_repr = {(s, tuple(f.index(i) for i in f0)) for s, f in formulas}  # set of generators (permutations)

    # they can be composed, for instance if you have ijk=jik=ikj
    # you also have ijk=jki
    # applying all possible compositions creates an entire group
    while True:
        n = len(perm_repr)
        perm_repr = perm_repr.union([(s, perm.inverse(p)) for s, p in perm_repr])
        perm_repr = perm_repr.union([(s1 * s2, perm.compose(p1, p2)) for s1, p1 in perm_repr for s2, p2 in perm_repr])
        if len(perm_repr) == n:
            break  # we break when the set is stable => it is now a group \o/

    return f0, frozenset(perm_repr)


def reduce_basis_product(
    basis1: e3nn.IrrepsArray,
    basis2: e3nn.IrrepsArray,
    filter_ir_out: Optional[List[e3nn.Irrep]] = None,
) -> e3nn.IrrepsArray:
    """Reduce the product of two basis."""
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


def subrepr_permutation(
    sub_f0: FrozenSet[int], perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]]
) -> FrozenSet[Tuple[int, Tuple[int, ...]]]:
    sor = sorted(sub_f0)
    return frozenset(
        {
            (s, tuple(sor.index(i) for i in p if i in sub_f0))
            for s, p in perm_repr
            if all(i in sub_f0 or i == j for j, i in enumerate(p))
        }
    )


def reduce_subgroup_permutation(
    sub_f0: FrozenSet[int], perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]], dims: Tuple[int, ...], return_dim: bool = False
) -> np.ndarray:
    sub_perm_repr = subrepr_permutation(sub_f0, perm_repr)
    sub_dims = tuple(dims[i] for i in sub_f0)
    if len(sub_perm_repr) == 1:
        if return_dim:
            return prod(sub_dims)
        return np.eye(prod(sub_dims)).reshape((prod(sub_dims),) + sub_dims)
    base = reduce_permutation_base(sub_perm_repr, sub_dims)
    if return_dim:
        return len(base)
    permutation_basis = reduce_permutation_matrix(base, sub_dims)
    return np.reshape(permutation_basis, (-1,) + tuple(dims[i] if i in sub_f0 else 1 for i in range(len(dims))))


@functools.lru_cache(maxsize=None)
def full_base_fn(dims: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    return list(itertools.product(*(range(d) for d in dims)))


@functools.lru_cache(maxsize=None)
def reduce_permutation_base(
    perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]], dims: Tuple[int, ...]
) -> FrozenSet[FrozenSet[FrozenSet[Tuple[int, Tuple[int, ...]]]]]:
    full_base = full_base_fn(dims)  # (0, 0, 0), (0, 0, 1), (0, 0, 2), ... (3, 3, 3)
    # len(full_base) degrees of freedom in an unconstrained tensor

    # but there is constraints given by the group `formulas`
    # For instance if `ij=-ji`, then 00=-00, 01=-01 and so on
    base = set()
    for x in full_base:
        # T[x] is a coefficient of the tensor T and is related to other coefficient T[y]
        # if x and y are related by a formula
        xs = {(s, tuple(x[i] for i in p)) for s, p in perm_repr}
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
