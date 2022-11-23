"""
History of the different versions of the code:
- Initially developed by Mario Geiger in `e3nn`
- Ported in julia by Song Kim https://github.com/songk42/ReducedTensorProduct.jl
- Ported in `e3nn-jax` by Mario Geiger
"""
import functools
import itertools
import os
from typing import FrozenSet, List, Optional, Tuple, Union

import numpy as np

import e3nn_jax as e3nn
from e3nn_jax import perm
from e3nn_jax._src.util.math_numpy import basis_intersection, round_to_sqrt_rational
from e3nn_jax._src.util.prod import prod


def reduced_tensor_product_basis(
    formula_or_irreps_list: Union[str, List[e3nn.Irreps]],
    *,
    epsilon: float = 1e-5,
    keep_ir: Optional[List[e3nn.Irrep]] = None,
    max_order: Optional[int] = None,
    **irreps_dict,
) -> e3nn.IrrepsArray:
    r"""Reduce a tensor product of multiple irreps subject to some permutation symmetry given by a formula.

    Args:
        formula_or_irreps_list (str or list of Irreps): a formula of the form ``ijk=jik=ikj`` or ``ijk=-jki``.
            The left hand side is the original formula and the right hand side are the signed permutations.
            If no index symmetry is present, a list of irreps can be given instead.

        epsilon (float): the tolerance for the Gram-Schmidt orthogonalization. Default: ``1e-5``
        keep_ir (list of Irrep): irrep to keep in the output. Default: keep all irrep
        max_order (int): the maximum polynomial order assuming the input to be order ``l``. Default: no limit
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
        return _reduced_tensor_product_basis(irreps_tuple, formulas, keep_ir, epsilon, max_order)[0].simplify()

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

    return _reduced_tensor_product_basis(irreps_tuple, perm_repr, keep_ir, epsilon, max_order)[0].simplify()


def _symmetric_perm_repr(n: int):
    return frozenset((1, p) for p in itertools.permutations(range(n)))


def reduced_symmetric_tensor_product_basis(
    irreps: e3nn.Irreps,
    order: int,
    *,
    epsilon: float = 1e-5,
    keep_ir: Optional[List[e3nn.Irrep]] = None,
    max_order: Optional[int] = None,
) -> e3nn.IrrepsArray:
    r"""Reduce a symmetric tensor product.

    Args:
        irreps (Irreps): the irreps of each index.
        order (int): the order of the tensor product. i.e. the number of indices.
        epsilon (float): the tolerance for the Gram-Schmidt orthogonalization. Default: ``1e-5``
        keep_ir (list of Irrep): irrep to keep in the output. Default: keep all irrep
        max_order (int): the maximum polynomial order assuming the input to be order ``l``. Default: no limit

    Returns:
        IrrepsArray: The change of basis
            The shape is ``(d, ..., d, irreps_out.dim)``
            where ``d`` is the dimension of ``irreps``.
    """
    # TODO add antisymmetric tensor product
    if keep_ir is not None:
        keep_ir = frozenset(e3nn.Irrep(ir) for ir in keep_ir)

    irreps = e3nn.Irreps(irreps)
    perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]] = _symmetric_perm_repr(order)
    return _reduced_tensor_product_basis(tuple([irreps] * order), perm_repr, keep_ir, epsilon, max_order)[0].simplify()


def _simplify(irreps_array: e3nn.IrrepsArray, orders: Tuple[int, ...]) -> Tuple[e3nn.IrrepsArray, Tuple[int, ...]]:
    new_irreps = []
    new_orders = []
    new_list = []
    for (mul, ir), order, x in zip(irreps_array.irreps, orders, irreps_array.list):
        if len(new_irreps) > 0 and new_irreps[-1][1] == ir and new_orders[-1] == order:
            new_irreps[-1][0] += mul
            new_list[-1] = np.concatenate([new_list[-1], x], axis=-2)
        else:
            new_irreps.append([mul, ir])
            new_orders.append(order)
            new_list.append(x)
    return e3nn.IrrepsArray.from_list(new_irreps, new_list, irreps_array.shape[:-1]), tuple(new_orders)


def _sort(irreps_array: e3nn.IrrepsArray, orders: Tuple[int, ...]) -> Tuple[e3nn.IrrepsArray, Tuple[int, ...]]:
    out = [(ir, o, i, mul) for i, ((mul, ir), o) in enumerate(zip(irreps_array.irreps, orders))]
    out = sorted(out)
    inv = tuple(i for _, _, i, _ in out)
    new_irreps = [irreps_array.irreps[i] for i in inv]
    new_list = [irreps_array.list[i] for i in inv]
    new_orders = [orders[i] for i in inv]
    return e3nn.IrrepsArray.from_list(new_irreps, new_list, irreps_array.shape[:-1]), tuple(new_orders)


def _sort_simplify(irreps_array: e3nn.IrrepsArray, orders: Tuple[int, ...]) -> Tuple[e3nn.IrrepsArray, Tuple[int, ...]]:
    irreps_array, orders = _sort(irreps_array, orders)
    irreps_array, orders = _simplify(irreps_array, orders)
    return irreps_array, orders


def _filter_ir(
    irreps_array: e3nn.IrrepsArray, orders: Tuple[int, ...], keep_ir: FrozenSet[e3nn.Irrep]
) -> Tuple[e3nn.IrrepsArray, Tuple[int, ...]]:
    orders = [o for o, (_, ir) in zip(orders, irreps_array.irreps) if ir in keep_ir]
    irreps_array = irreps_array.filtered(keep_ir)
    return irreps_array, tuple(orders)


def _filter_order(irreps_array: e3nn.IrrepsArray, orders, max_order: int) -> Tuple[e3nn.IrrepsArray, Tuple[int, ...]]:
    irreps_array = e3nn.IrrepsArray.from_list(
        [mul_ir for mul_ir, o in zip(irreps_array.irreps, orders) if o <= max_order],
        [x for x, o in zip(irreps_array.list, orders) if o <= max_order],
        irreps_array.shape[:-1],
    )
    orders = [o for o in orders if o <= max_order]
    return irreps_array, tuple(orders)


def _check_database(
    irreps_tuple: Tuple[e3nn.Irreps],
    perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]],
    keep_ir: Optional[FrozenSet[e3nn.Irrep]],
    max_order: Optional[int] = None,
) -> Optional[Tuple[e3nn.IrrepsArray, Tuple[int, ...]]]:
    path = os.path.join(os.path.dirname(__file__), "rtp.npz")
    if not os.path.exists(path):
        return None

    if max_order is not None:
        return None

    key = None
    if perm_repr == _symmetric_perm_repr(len(irreps_tuple)):
        key = f"symmetric_{irreps_tuple[0]}_{len(irreps_tuple)}_"

    if key is None:
        return None

    with np.load(path) as f:
        for k in f:
            if k.startswith(key):
                out = e3nn.IrrepsArray(e3nn.Irreps(k.split("_")[-1]), f[k]).filtered(keep_ir)
                return out, (0,) * len(out.irreps)

    return None


@functools.lru_cache(maxsize=None)
def _reduced_tensor_product_basis(
    irreps_tuple: Tuple[e3nn.Irreps],
    perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]],
    keep_ir: Optional[FrozenSet[e3nn.Irrep]],
    epsilon: float,
    max_order: Optional[int] = None,
) -> Tuple[e3nn.IrrepsArray, Tuple[int, ...]]:
    out = _check_database(irreps_tuple, perm_repr, keep_ir, max_order)
    if out is not None:
        return out

    dims = tuple(irreps.dim for irreps in irreps_tuple)

    bases = [
        (
            frozenset({i}),
            e3nn.IrrepsArray(
                irreps,
                np.reshape(np.eye(irreps.dim), (1,) * i + (irreps.dim,) + (1,) * (len(irreps_tuple) - i - 1) + (irreps.dim,)),
            ),
            tuple(ir.l for _, ir in irreps),  # order of the polynomial
        )
        for i, irreps in enumerate(irreps_tuple)
    ]

    while True:
        if len(bases) == 1:
            f, b, ord = bases[0]
            assert f == frozenset(range(len(irreps_tuple)))
            if max_order is not None:
                (b, ord) = _filter_order(b, ord, max_order)
            if keep_ir is not None:
                (b, ord) = _filter_ir(b, ord, keep_ir)
            return _sort_simplify(b, ord)

        if len(bases) == 2:
            (fa, a, oa) = bases[0]
            (fb, b, ob) = bases[1]
            f = frozenset(fa | fb)
            ab, ord = reduce_basis_product(a, oa, b, ob, max_order, keep_ir, round_fn=round_to_sqrt_rational)
            if len(subrepr_permutation(f, perm_repr)) == 1:
                return _sort_simplify(ab, ord)
            p = reduce_subgroup_permutation(f, perm_repr, dims)
            ab, ord = constrain_rotation_basis_by_permutation_basis(
                ab, p, ord, epsilon=epsilon, round_fn=round_to_sqrt_rational
            )
            return _sort_simplify(ab, ord)

        # greedy algorithm
        min_p = np.inf
        best = None

        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                (fa, _, _) = bases[i]
                (fb, _, _) = bases[j]
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
        ab, ord = _reduced_tensor_product_basis(sub_irreps, sub_perm_repr, None, epsilon, max_order)
        ab = ab.reshape(tuple(dims[i] if i in f else 1 for i in range(len(dims))) + (-1,))
        bases = [(f, ab, ord)] + bases


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
    order1: Tuple[int, ...],
    basis2: e3nn.IrrepsArray,
    order2: Tuple[int, ...],
    max_order: Optional[int] = None,
    filter_ir_out: Optional[List[e3nn.Irrep]] = None,
    round_fn=lambda x: x,
) -> Tuple[e3nn.IrrepsArray, Tuple[int, ...]]:
    """Reduce the product of two basis."""
    basis1, order1 = _sort_simplify(basis1, order1)
    basis2, order2 = _sort_simplify(basis2, order2)

    new_irreps: List[Tuple[int, e3nn.Irrep]] = []
    new_list = []
    new_orders = []

    for (mul1, ir1), x1, o1 in zip(basis1.irreps, basis1.list, order1):
        for (mul2, ir2), x2, o2 in zip(basis2.irreps, basis2.list, order2):
            if max_order is not None and o1 + o2 > max_order:
                continue

            for ir in ir1 * ir2:
                if filter_ir_out is not None and ir not in filter_ir_out:
                    continue

                x = np.einsum(
                    "...ui,...vj,ijk->...uvk",
                    x1,
                    x2,
                    np.sqrt(ir.dim) * e3nn.clebsch_gordan(ir1.l, ir2.l, ir.l),
                )
                x = round_fn(x)
                x = np.reshape(x, x.shape[:-3] + (mul1 * mul2, ir.dim))
                new_irreps.append((mul1 * mul2, ir))
                new_list.append(x)
                new_orders.append(o1 + o2)

    new = e3nn.IrrepsArray.from_list(new_irreps, new_list, np.broadcast_shapes(basis1.shape[:-1], basis2.shape[:-1]))
    return _sort_simplify(new, tuple(new_orders))


def constrain_rotation_basis_by_permutation_basis(
    rotation_basis: e3nn.IrrepsArray,
    permutation_basis: np.ndarray,
    orders: Tuple[int, ...],
    *,
    epsilon=1e-5,
    round_fn=lambda x: x,
) -> Tuple[e3nn.IrrepsArray, Tuple[int, ...]]:
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
    new_orders: List[int] = []

    for ir in sorted({ir for mul, ir in rotation_basis.irreps}):
        idx = [i for i, (mul, ir_) in enumerate(rotation_basis.irreps) if ir == ir_]
        rot_basis = np.concatenate([rotation_basis.list[i] for i in idx], axis=-2)
        ord = np.array([orders[i] for i in idx for _ in range(rotation_basis.irreps[i].mul)])
        mul = rot_basis.shape[-2]
        R = rot_basis[..., 0]
        R = np.reshape(R, (-1, mul)).T  # (mul, dim)

        # optimization:
        perm_opt = perm[~np.all(perm[:, ~np.all(R == 0, axis=0)] == 0, axis=1)]
        # NOTE: this optimization work only because perm rows don't share non-zero elements

        P, _ = basis_intersection(R, perm_opt, epsilon=epsilon, round_fn=round_fn)

        for p in P:
            new_irreps.append((1, ir))
            new_list.append(round_fn(np.einsum("u,...ui->...i", p, rot_basis)[..., None, :]))
            new_orders.append(int(np.max(ord * (p != 0.0))))

    return e3nn.IrrepsArray.from_list(new_irreps, new_list, rotation_basis.shape[:-1]), tuple(new_orders)


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
