import itertools
from collections import defaultdict
from functools import lru_cache

import sympy

from e3nn_jax._wigner import wigner_3j_sympy


def symmetric_terms(xx):
    n = len(xx.shape)
    for per in itertools.permutations(range(n)):
        yy = sympy.permutedims(xx, per)
        for x in sympy.flatten(xx - yy):
            yield x


def is_symmetric(xx):
    return all(x == 0 for x in symmetric_terms(xx))


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
        out = sympy.tensorproduct(out, input_1)  # ijk,j...
        out = sympy.tensorcontraction(out, (1, 3))  # ik...

    if not isinstance(input_2, int):
        n = len(out.shape)
        out = sympy.tensorproduct(out, input_2)  # ik...,k...
        out = sympy.tensorcontraction(out, (1, n))  # i......

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
    variables = [sympy.symbols(f"x{i}") for i in range(len(candidates))]
    tensors = sum_axis0([x * c for x, c in zip(variables, candidates)])
    tensor = tensors[0]  # XXX solve for first component only
    constraints = tuple(set(symmetric_terms(tensor)))
    solution = sympy.solve(constraints, variables)
    tensors = tensors.subs(solution)

    tensors = [tensors.subs(x, 1).subs(zip(variables, [0] * len(variables))) for x in variables]
    norms = [sympy.sqrt(sum(sympy.flatten(s.applyfunc(lambda x: x**2)))) for s in tensors]
    return [s / n for s, n in zip(tensors, norms) if not n.is_zero]


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
def symmetric_powers(l, n):
    assert n > 0

    if n == 1:
        return {
            l: [sympy.Array(sympy.eye(2 * l + 1))]
        }

    res = defaultdict(lambda: [])

    if n % 2 == 0:
        sub = symmetric_powers(l, n // 2)

        for l1 in sub.keys():
            for l2 in sub.keys():
                for lout in range(abs(l1 - l2), l1 + l2 + 1):
                    for a in sub[l1]:
                        for b in sub[l2]:
                            res[lout].append(product_lll(lout, a, b))

    else:
        sub = symmetric_powers(l, n - 1)

        for l1 in sub.keys():
            for lout in range(abs(l1 - l), l1 + l + 1):
                for a in sub[l1]:
                    res[lout].append(product_lll(lout, a, l))

    res = {l: orthonormalize(solve_symmetric(z))[0] for l, z in res.items()}
    res = {l: z for l, z in res.items() if len(z) > 0}
    return res
