import functools
import operator


def prod(xs):
    return functools.reduce(operator.mul, xs, 1)
