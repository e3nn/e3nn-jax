import itertools
from collections import defaultdict
from functools import lru_cache

import sympy

from e3nn_jax._wigner import wigner_3j_sympy
from e3nn_jax.util import prod


@lru_cache(maxsize=None)
def classes_of_combinations_with_replacement(n, k):
    """
    Return all [n]**k regrouped by the permutations
    """
    result = []
    for x in itertools.combinations_with_replacement(range(n), k):
        result.append(list(set(itertools.permutations(x))))
    return result


def symmetric_terms(xx):
    d = xx.shape[0]
    k = len(xx.shape)
    for c in classes_of_combinations_with_replacement(d, k):
        x = {xx[i] for i in c}
        x = list(x)
        for y in x[1:]:
            yield x[0] - y


def is_symmetric(xx):
    return all(x == 0 for x in symmetric_terms(xx))


def new_array(*shape):
    return sympy.Array(sympy.zeros(prod(shape), 1)).reshape(*shape)


def tensordot(a, b, i, j):
    perm = list(range(len(a.shape)))
    del perm[i]
    perm.append(i)
    a = sympy.permutedims(a, perm)

    perm = list(range(len(b.shape)))
    del perm[j]
    perm.insert(0, j)
    b = sympy.permutedims(b, perm)

    a_shape = a.shape
    b_shape = b.shape

    a = a.reshape(prod(a_shape[:-1]), a_shape[-1])
    b = b.reshape(b_shape[0], prod(b_shape[1:]))

    out = sympy.SparseMatrix(a) @ sympy.SparseMatrix(b)
    out = sympy.Array(out)
    out = out.reshape(*(a_shape[:-1] + b_shape[1:]))
    return out


def product_lll(output, input_1, input_2):
    if isinstance(input_1, int):
        l1 = input_1
    else:
        l1 = input_1.shape[0] // 2

    if isinstance(input_2, int):
        l2 = input_2
    else:
        l2 = input_2.shape[0] // 2

    out = wigner_3j_sympy(output, l1, l2)  # ijk

    if not isinstance(input_1, int):
        out = tensordot(out, input_1, 1, 0)

    if not isinstance(input_2, int):
        out = tensordot(out, input_2, 1, 0)

    return out


def sum_axis0(xs):
    x = xs[0]
    for y in xs[1:]:
        x = x + y
    return x


def dot(xs, ys):
    xs = sympy.flatten(xs)
    ys = sympy.flatten(ys)
    res = 0
    for x, y in zip(xs, ys):
        res = res + x * y
    return res


def orthonormalize(original):
    r"""orthonomalize vectors

    Parameters
    ----------
    original : list of vectors
        list of the original vectors :math:`x`

    Returns
    -------
    final : list of vectors
        list of orthonomalized vectors :math:`y`

    matrix : matrix
        the matrix :math:`A` such that :math:`y = A x`
    """
    final = []
    matrix = []

    for i, x in enumerate(original):
        # x = sum_i cx_i original_i
        cx = sympy.zeros(1, len(original))
        cx[i] = 1
        for j, y in enumerate(final):
            c = dot(x, y)
            x = x - c * y
            cx = cx - c * matrix[j]

        norm = sympy.sqrt(dot(x, x))
        if norm > 0:
            c = 1 / norm
            x = c * x
            cx = c * cx
            for a in sympy.flatten(x):
                if a != 0:
                    if a < 0:
                        x = -x
                        cx = -cx
                    break
            final += [x]
            matrix += [cx]

    return final, sympy.Matrix(matrix)


def solve_symmetric(candidates):
    assert len(candidates) >= 1

    if len(candidates) == 1:
        candidate = candidates[0]
        if all(is_symmetric(x) for x in candidate):
            return [candidate]
        else:
            return []

    variables = [sympy.symbols(f"x{i}") for i in range(len(candidates))]
    tensors = sum_axis0([x * c for x, c in zip(variables, candidates)])
    constraints = tuple({sympy.expand(x) for tensor in tensors[:1] for x in symmetric_terms(tensor)})

    solutions = sympy.solve(constraints, variables, manual=True, dict=True)
    assert len(solutions) == 1
    solution = solutions[0]
    tensors = tensors.subs(solution)

    tensors = [tensors.subs(x, 1).subs(zip(variables, [0] * len(variables))) for x in variables]
    norms = [sympy.sqrt(sum(sympy.flatten(s.applyfunc(lambda x: x**2)))) for s in tensors]
    solutions = [s / n for s, n in zip(tensors, norms) if not n.is_zero]

    # assert all(is_symmetric(x) for sol in solutions for x in sol)
    return solutions


def contract_with_variables(solution):
    n = solution.shape[1]
    assert all(x == n for x in solution.shape[1:])

    variables = [sympy.symbols(f"x{i}") for i in range(n)]
    out = solution
    while len(out.shape) > 1:
        out = sympy.tensorproduct(out, variables)
        out = sympy.tensorcontraction(out, (1, len(out.shape) - 1))

    return sympy.simplify(out)


@lru_cache(maxsize=None)
def symmetric_powers(l, n, lmax):
    assert n > 0
    assert l <= lmax

    if n == 1:
        return {
            l: [sympy.Array(sympy.eye(2 * l + 1))]
        }

    res = defaultdict(lambda: [])

    if n % 2 == 0:
        sub = symmetric_powers(l, n // 2, lmax)

        for l1 in sub.keys():
            for l2 in sub.keys():
                for lout in range(abs(l1 - l2), min(lmax, l1 + l2) + 1):
                    for a in sub[l1]:
                        for b in sub[l2]:
                            res[lout].append(product_lll(lout, a, b))

    else:
        sub1 = symmetric_powers(l, n // 2, lmax)
        sub2 = symmetric_powers(l, n // 2 + 1, lmax)

        for l1 in sub1.keys():
            for l2 in sub2.keys():
                for lout in range(abs(l1 - l2), min(lmax, l1 + l2) + 1):
                    for a in sub1[l1]:
                        for b in sub2[l2]:
                            res[lout].append(product_lll(lout, a, b))

    res = {l: solve_symmetric(z) for l, z in res.items()}
    res = {l: orthonormalize(z)[0] for l, z in res.items()}
    res = {l: z for l, z in res.items() if len(z) > 0}
    res = {l: [sympy.simplify(x) for x in z] for l, z in res.items()}
    return res
