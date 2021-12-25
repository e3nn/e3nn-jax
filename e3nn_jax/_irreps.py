import collections
import itertools
from functools import partial
from typing import List

import jax
import copy
import jax.numpy as jnp
import jax.scipy
from jax import lax

from e3nn_jax import matrix_to_angles, perm, quaternion_to_angles, wigner_D


class Irrep(tuple):
    r"""Irreducible representation of :math:`O(3)`

    This class does not contain any data, it is a structure that describe the representation.
    It is typically used as argument of other classes of the library to define the input and output representations of functions.

    Args:
        l: non-negative integer, the degree of the representation, :math:`l = 0, 1, \dots`
        p: {1, -1}, the parity of the representation

    Examples:
        Create a scalar representation (:math:`l=0`) of even parity.

        >>> Irrep(0, 1)
        0e

        Create a pseudotensor representation (:math:`l=2`) of odd parity.

        >>> Irrep(2, -1)
        2o

        Create a vector representation (:math:`l=1`) of the parity of the spherical harmonics (:math:`-1^l` gives odd parity).

        >>> Irrep("1y")
        1o

        >>> Irrep("2o").dim
        5

        >>> Irrep("2e") in Irrep("1o") * Irrep("1o")
        True

        >>> Irrep("1o") + Irrep("2o")
        1x1o+1x2o
    """
    def __new__(cls, l, p=None):
        if p is None:
            if isinstance(l, Irrep):
                return l

            if isinstance(l, str):
                try:
                    name = l.strip()
                    l = int(name[:-1])
                    assert l >= 0
                    p = {
                        'e': 1,
                        'o': -1,
                        'y': (-1)**l,
                    }[name[-1]]
                except Exception:
                    raise ValueError(f"unable to convert string \"{name}\" into an Irrep")
            elif isinstance(l, tuple):
                l, p = l

        assert isinstance(l, int) and l >= 0, l
        assert p in [-1, 1], p
        return super().__new__(cls, (l, p))

    @property
    def l(self) -> int:
        r"""The degree of the representation, :math:`l = 0, 1, \dots`."""
        return self[0]

    @property
    def p(self) -> int:
        r"""The parity of the representation, :math:`p = \pm 1`."""
        return self[1]

    def __repr__(self):
        p = {+1: 'e', -1: 'o'}[self.p]
        return f"{self.l}{p}"

    @classmethod
    def iterator(cls, lmax=None):
        r"""Iterator through all the irreps of :math:`O(3)`

        Examples:
            >>> it = Irrep.iterator()
            >>> next(it), next(it), next(it), next(it)
            (0e, 0o, 1o, 1e)
        """
        for l in itertools.count():
            yield Irrep(l, (-1)**l)
            yield Irrep(l, -(-1)**l)

            if l == lmax:
                break

    def D_from_angles(self, alpha, beta, gamma, k=0):
        r"""Matrix :math:`p^k D^l(\alpha, \beta, \gamma)`

        (matrix) Representation of :math:`O(3)`. :math:`D` is the representation of :math:`SO(3)`, see `wigner_D`.

        Args:
            alpha (`jnp.ndarray`): of shape :math:`(...)`
                Rotation :math:`\alpha` around Y axis, applied third.
            beta (`jnp.ndarray`): of shape :math:`(...)`
                Rotation :math:`\beta` around X axis, applied second.
            gamma (`jnp.ndarray`): of shape :math:`(...)`
                Rotation :math:`\gamma` around Y axis, applied first.
            k (optional `jnp.ndarray`): of shape :math:`(...)`
                How many times the parity is applied.

        Returns:
            `jnp.ndarray`: of shape :math:`(..., 2l+1, 2l+1)`

        See Also:
            o3.wigner_D
            Irreps.D_from_angles
        """
        alpha, beta, gamma, k = jnp.broadcast_arrays(alpha, beta, gamma, k)
        k = jnp.asarray(k)
        return wigner_D(self.l, alpha, beta, gamma) * self.p**k[..., None, None]

    def D_from_quaternion(self, q, k=0):
        r"""Matrix of the representation, see `Irrep.D_from_angles`

        Args:
            q (`jnp.ndarray`): shape :math:`(..., 4)`
            k (optional `jnp.ndarray`): shape :math:`(...)`

        Returns:
            `jnp.ndarray`: shape :math:`(..., 2l+1, 2l+1)`
        """
        return self.D_from_angles(*quaternion_to_angles(q), k)

    def D_from_matrix(self, R):
        r"""Matrix of the representation

        Args:
            R (`jnp.ndarray`): array of shape :math:`(..., 3, 3)`
            k (`jnp.ndarray`, optional): array of shape :math:`(...)`

        Returns:
            `jnp.ndarray`: array of shape :math:`(..., 2l+1, 2l+1)`

        Examples:
            >>> m = Irrep(1, -1).D_from_matrix(-jnp.eye(3))
            >>> m + 0.0
            DeviceArray([[-1.,  0.,  0.],
                         [ 0., -1.,  0.],
                         [ 0.,  0., -1.]], dtype=float32)

        See Also:
            `Irrep.D_from_angles`
        """
        d = jnp.sign(jnp.linalg.det(R))
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.D_from_angles(*matrix_to_angles(R), k)

    @property
    def dim(self) -> int:
        """The dimension of the representation, :math:`2 l + 1`."""
        return 2 * self.l + 1

    def is_scalar(self) -> bool:
        """Equivalent to ``l == 0 and p == 1``"""
        return self.l == 0 and self.p == 1

    def __mul__(self, other):
        r"""Generate the irreps from the product of two irreps.

        Returns:
            generator of `Irrep`
        """
        other = Irrep(other)
        p = self.p * other.p
        lmin = abs(self.l - other.l)
        lmax = self.l + other.l
        for l in range(lmin, lmax + 1):
            yield Irrep(l, p)

    def count(self, _value):
        raise NotImplementedError

    def index(self, _value):
        raise NotImplementedError

    def __rmul__(self, other):
        r"""
        >>> 3 * Irrep('1e')
        3x1e
        """
        assert isinstance(other, int)
        return Irreps([(other, self)])

    def __add__(self, other):
        return Irreps(self) + Irreps(other)

    def __contains__(self, _object):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class _MulIr(tuple):
    def __new__(cls, mul, ir=None):
        if ir is None:
            mul, ir = mul

        assert isinstance(mul, int)
        assert isinstance(ir, Irrep)
        return super().__new__(cls, (mul, ir))

    @property
    def mul(self) -> int:
        return self[0]

    @property
    def ir(self) -> int:
        return self[1]

    @property
    def dim(self) -> int:
        return self.mul * self.ir.dim

    def __repr__(self):
        return f"{self.mul}x{self.ir}"

    def count(self, _value):
        raise NotImplementedError

    def index(self, _value):
        raise NotImplementedError


class Irreps(tuple):
    r"""Direct sum of irreducible representations of :math:`O(3)`

    This class does not contain any data, it is a structure that describe the representation.
    It is typically used as argument of other classes of the library to define the input and output representations of functions.

    Attributes:
        dim (int): the total dimension of the representation
        num_irreps (int): number of irreps. the sum of the multiplicities
        ls (list of int): list of :math:`l` values
        lmax (int): maximum :math:`l` value

    Examples:
        >>> x = Irreps([(100, (0, 1)), (50, (1, 1))])
        >>> x
        100x0e+50x1e

        >>> x.dim
        250

        >>> Irreps("100x0e + 50x1e")
        100x0e+50x1e

        >>> Irreps("100x0e + 50x1e + 0x2e")
        100x0e+50x1e+0x2e

        >>> Irreps("100x0e + 50x1e + 0x2e").lmax
        1

        >>> Irrep("2e") in Irreps("0e + 2e")
        True

        Empty Irreps

        >>> Irreps(), Irreps("")
        (, )
    """
    def __new__(cls, irreps=None):
        if isinstance(irreps, Irreps):
            return super().__new__(cls, irreps)

        out = []
        if isinstance(irreps, Irrep):
            out.append(_MulIr(1, Irrep(irreps)))
        elif isinstance(irreps, str):
            try:
                if irreps.strip() != "":
                    for mul_ir in irreps.split('+'):
                        if 'x' in mul_ir:
                            mul, ir = mul_ir.split('x')
                            mul = int(mul)
                            ir = Irrep(ir)
                        else:
                            mul = 1
                            ir = Irrep(mul_ir)

                        assert isinstance(mul, int) and mul >= 0
                        out.append(_MulIr(mul, ir))
            except Exception:
                raise ValueError(f"Unable to convert string \"{irreps}\" into an Irreps")
        elif irreps is None:
            pass
        else:
            for mul_ir in irreps:
                if isinstance(mul_ir, str):
                    mul = 1
                    ir = Irrep(mul_ir)
                elif isinstance(mul_ir, Irrep):
                    mul = 1
                    ir = mul_ir
                elif isinstance(mul_ir, _MulIr):
                    mul, ir = mul_ir
                elif isinstance(mul_ir, int):
                    mul, ir = 1, Irrep(l=mul_ir, p=1)
                elif len(mul_ir) == 2:
                    mul, ir = mul_ir
                    ir = Irrep(ir)
                else:
                    mul = None
                    ir = None

                if not (isinstance(mul, int) and mul >= 0 and ir is not None):
                    raise ValueError(f"Unable to interpret \"{mul_ir}\" as an irrep.")

                out.append(_MulIr(mul, ir))
        return super().__new__(cls, out)

    @staticmethod
    def spherical_harmonics(lmax, p=-1):
        r"""representation of the spherical harmonics

        Args:
            lmax (int): maximum :math:`l`
            p (optional {1, -1}): the parity of the representation

        Returns:
            `Irreps`: representation of :math:`(Y^0, Y^1, \dots, Y^{\mathrm{lmax}})`

        Examples:
            >>> Irreps.spherical_harmonics(3)
            1x0e+1x1o+1x2e+1x3o
            >>> Irreps.spherical_harmonics(4, p=1)
            1x0e+1x1e+1x2e+1x3e+1x4e
        """
        return Irreps([(1, (l, p**l)) for l in range(lmax + 1)])

    def slices(self):
        r"""List of slices corresponding to indices for each irrep.

        Examples:
            >>> Irreps('2x0e + 1e').slices()
            [slice(0, 2, None), slice(2, 5, None)]
        """
        s = []
        i = 0
        for mul_ir in self:
            s.append(slice(i, i + mul_ir.dim))
            i += mul_ir.dim
        return s

    def randn(self, key, size, *, normalization='component'):
        r"""Random tensor.

        Args:
            *size (list of int): size of the output tensor, needs to contains a ``-1``
            normalization : {'component', 'norm'}

        Returns:
            `jnp.ndarray`: array of shape ``size`` where ``-1`` is replaced by ``self.dim``

        Examples:
            >>> key = jax.random.PRNGKey(0)
            >>> Irreps("5x0e + 10x1o").randn(key, (5, -1, 5), normalization='norm').shape
            (5, 35, 5)

            >>> random_tensor = Irreps("2o").randn(key, (2, -1, 3), normalization='norm')
            >>> jnp.max(jnp.abs(jnp.linalg.norm(random_tensor, axis=1) - 1)) < 1e-5
            DeviceArray(True, dtype=bool)
        """
        di = size.index(-1)
        lsize = size[:di]
        rsize = size[di + 1:]

        if normalization == 'component':
            return jax.random.normal(key, lsize + (self.dim,) + rsize)
        elif normalization == 'norm':
            x = jnp.zeros(lsize + (self.dim,) + rsize)
            for s, (mul, ir) in zip(self.slices(), self):
                key, k = jax.random.split(key)
                r = jax.random.normal(k, lsize + (mul, ir.dim) + rsize)
                r = r / jnp.linalg.norm(r, axis=di + 1, keepdims=True)
                i = di * (slice(None, None, None),) + (s,)
                x = x.at[i].set(r.reshape(*lsize, -1, *rsize))
            return x
        else:
            raise ValueError("Normalization needs to be 'norm' or 'component'")

    def __getitem__(self, i):
        x = super().__getitem__(i)
        if isinstance(i, slice):
            return Irreps(x)
        return x

    def __contains__(self, ir) -> bool:
        ir = Irrep(ir)
        return ir in (irrep for _, irrep in self)

    def count(self, ir) -> int:
        r"""Multiplicity of ``ir``.

        Args:
            ir (`Irrep`):

        Returns:
            `int`: total multiplicity of ``ir``
        """
        ir = Irrep(ir)
        return sum(mul for mul, irrep in self if ir == irrep)

    def index(self, _object):
        raise NotImplementedError

    def __add__(self, irreps):
        irreps = Irreps(irreps)
        return Irreps(super().__add__(irreps))

    def __mul__(self, other):
        r"""
        >>> (Irreps('2x1e') * 3).simplify()
        6x1e
        """
        if isinstance(other, Irreps):
            raise NotImplementedError("Use o3.TensorProduct for this, see the documentation")
        return Irreps(super().__mul__(other))

    def __rmul__(self, other):
        r"""
        >>> 2 * Irreps('0e + 1e')
        1x0e+1x1e+1x0e+1x1e
        """
        return Irreps(super().__rmul__(other))

    def simplify(self):
        """Simplify the representations.

        Examples:
            Note that simplify does not sort the representations.

            >>> Irreps("1e + 1e + 0e").simplify()
            2x1e+1x0e

            Equivalent representations which are separated from each other are not combined.

            >>> Irreps("1e + 1e + 0e + 1e").simplify()
            2x1e+1x0e+1x1e
        """
        out = []
        for mul, ir in self:
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            elif mul > 0:
                out.append((mul, ir))
        return Irreps(out)

    def remove_zero_multiplicities(self):
        """Remove any irreps with multiplicities of zero.

        Examples:
            >>> Irreps("4x0e + 0x1o + 2x3e").remove_zero_multiplicities()
            4x0e+2x3e
        """
        return Irreps([(mul, ir) for mul, ir in self if mul > 0])

    def sort(self):
        r"""Sort the representations.

        Returns:
            irreps (`Irreps`): sorted irreps
            p (tuple of int): permutation of the indices
            inv (tuple of int): inverse permutation of the indices

        Examples:
            >>> Irreps("1e + 0e + 1e").sort().irreps
            1x0e+1x1e+1x1e
            >>> Irreps("2o + 1e + 0e + 1e").sort().p
            (3, 1, 0, 2)
            >>> Irreps("2o + 1e + 0e + 1e").sort().inv
            (2, 1, 3, 0)
        """
        Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
        out = [(ir, i, mul) for i, (mul, ir) in enumerate(self)]
        out = sorted(out)
        inv = tuple(i for _, i, _ in out)
        p = perm.inverse(inv)
        irreps = Irreps([(mul, ir) for ir, _, mul in out])
        return Ret(irreps, p, inv)

    @property
    def dim(self) -> int:
        return sum(mul * ir.dim for mul, ir in self)

    @property
    def num_irreps(self) -> int:
        return sum(mul for mul, _ in self)

    @property
    def ls(self) -> List[int]:
        return [l for mul, (l, p) in self for _ in range(mul)]

    @property
    def lmax(self) -> int:
        if len(self) == 0:
            raise ValueError("Cannot get lmax of empty Irreps")
        return max(self.ls)

    def __repr__(self):
        return "+".join(f"{mul_ir}" for mul_ir in self)

    @partial(jax.jit, static_argnums=(0, 1, 3), inline=True)
    def extract(self, indices, x, axis=-1):
        r"""Extract sub sets of irreps

        Args:
            indices (tuple of int):
            x (`jnp.ndarray`):

        Returns:
            `jnp.ndarray`: ``[self[i] for i in indices]``

        Examples:
            >>> irreps = Irreps("0e + 0e + 0e + 1e")
            >>> irreps.extract((0, 2), jnp.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]))
            DeviceArray([1., 3.], dtype=float32)
        """
        # TODO: input a list?
        s = self.slices()
        s = [s[i] for i in indices]

        i = 0
        while i + 1 < len(s):
            if s[i].stop == s[i + 1].start:
                s[i] = slice(s[i].start, s[i + 1].stop)
                del s[i + 1]
            else:
                i = i + 1

        if len(s) == 1 and s[0] == slice(0, self.dim):
            return x

        # TODO output a list?
        return jnp.concatenate([
            lax.slice_in_dim(x, i.start, i.stop, axis=axis)
            for i in s
        ], axis=axis)

    def assert_compatible(self, x):
        r"""

        Examples:
            >>> irreps = Irreps("0e + 0e + 0e + 1e")
            >>> irreps.assert_compatible(jnp.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]))
            >>> irreps.assert_compatible([jnp.array([[1.0]]), None, None, jnp.array([[0.0, 0.0, 0.0]])])
        """
        if isinstance(x, list):
            if any(a is None for a in x):
                # list containing some None
                assert len(x) == len(self), "list with None is only compatible if the length match"
                shape = None
                for (mul, ir), a in zip(self, x):
                    if a is None:
                        continue
                    assert a.shape[-1] == ir.dim, f"shape mismatch: {a.shape[-1]} and {ir.dim}"
                    assert a.shape[-2] == mul, f"shape mismatch: {a.shape[-2]} and {mul}"
                    if shape is None:
                        shape = a.shape[:-2]
                    else:
                        assert shape == a.shape[:-2], f"shape mismatch: {shape} and {a.shape[:-2]}"
            else:
                # list without any None
                Info = collections.namedtuple("info", ["mul", "dim"])
                assert len({a.shape[:-2] for a in x}) <= 1, "all arrays must have the same shape"
                x_ = [Info(a.shape[-2], a.shape[-1]) for a in x]
                x = copy.deepcopy(x_)
                y = [Info(mul, ir.dim) for mul, ir in self]
                xi = 0
                yi = 0
                while True:
                    if xi >= len(x) and yi >= len(y):
                        break
                    if xi < len(x) and x[xi].mul == 0:
                        xi += 1
                        continue
                    if yi < len(y) and y[yi].mul == 0:
                        yi += 1
                        continue
                    if xi >= len(x):
                        raise ValueError(f"the data contains less irreps than expected: {x_} < {self}")
                    if yi >= len(y):
                        raise ValueError(f"the data contains more irreps than expected: {x_} > {self}")
                    assert x[xi].dim == y[yi].dim, f"dimension mismatch: {x[xi].dim} and {y[yi].dim}"
                    mul = min(x[xi].mul, y[yi].mul)
                    x[xi] = Info(x[xi].mul - mul, x[xi].dim)
                    y[yi] = Info(y[yi].mul - mul, y[yi].dim)
        else:
            # array
            assert x.shape[-1] == self.dim, f"shape mismatch: {x.shape[-1]} and {self.dim}"

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def to_list(self, x):
        r"""Split irreps into blocks

        Args:
            x (`jnp.ndarray` or list of `jnp.ndarray`): array of shape :math:`(..., d)`

        Returns:
            list of `jnp.ndarray` of shape :math:`(..., mul, 2 l + 1)`

        Examples:
            >>> irreps = Irreps("0e + 1e")
            >>> irreps.to_list(jnp.array([1.0, 0.0, 0.0, 0.0]))
            [DeviceArray([[1.]], dtype=float32), DeviceArray([[0., 0., 0.]], dtype=float32)]
            >>> irreps.to_list([jnp.array([[1.0]]), None])
            [DeviceArray([[1.]], dtype=float32), None]
            >>> irreps = Irreps("2x0e")
            >>> irreps.to_list([jnp.array([[1.0]]), jnp.array([[1.0]])])
            [DeviceArray([[1.],
                         [1.]], dtype=float32)]
        """
        self.assert_compatible(x)

        if isinstance(x, list):
            if len(x) == 0:
                return []

            if any(a is None for a in x):
                # list containing some None
                assert len(x) == len(self), "list with None is only compatible if the length match"
                return x
            else:
                # list without any None
                out = []
                r = x.pop(0)

                for mul, ir in self[:-1]:
                    assert r.shape[-1] == ir.dim

                    while r.shape[-2] < mul:
                        r = jnp.concatenate([r, x.pop(0)], axis=-2)

                    if r.shape[-2] == mul:
                        out.append(r)
                        r = x.pop(0)
                    else:
                        out.append(r[..., :mul, :])
                        r = r[..., mul:, :]

                mul, ir = self[-1]
                assert r.shape[-1] == ir.dim

                while r.shape[-2] < mul:
                    r = jnp.concatenate([r, x.pop(0)], axis=-2)

                assert r.shape[-2] == mul
                assert len(x) == 0

                out.append(r)

                return out

        # array
        shape = x.shape[:-1]
        if len(self) == 1:
            mul, ir = self[0]
            return [jnp.reshape(x, shape + (mul, ir.dim))]
        else:
            return [
                jnp.reshape(x[..., i], shape + (mul, ir.dim))
                for i, (mul, ir) in zip(self.slices(), self)
            ]

    def shape_of(self, x):
        r"""Infers the shape of the data

        Args:
            x (`jnp.ndarray` or list of optional `jnp.ndarray`): data compatible with the irreps.

        Returns:
            tuple of int: shape of the data
        """
        if isinstance(x, list):
            for a in x:
                if a is not None:
                    return a.shape[:-2]
            raise ValueError(f"cannot get the shape of {x}")
        return x.shape[:-1]

    def replace_none_with_zeros(self, x):
        r"""Replace None with zeros

        Only works for list of same length as the irreps
        """
        assert isinstance(x, list)
        assert len(x) == len(self)
        shape = self.shape_of(x)
        out = []
        for (mul, ir), a in zip(self, x):
            if a is None:
                a = jnp.zeros(shape + (mul, ir.dim))
            out.append(a)
        return out

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def to_contiguous(self, x):
        r"""Convert data into a contiguous array

        If the data is a list containing some `None`, it has to have the same length as the irreps.

        Args:
            x (`jnp.ndarray` or list of optional `jnp.ndarray`): data compatible with the irreps.

        Returns:
            `jnp.ndarray`
        """
        self.assert_compatible(x)
        if isinstance(x, list):
            shape = self.shape_of(x)
            if any(a is None for a in x):
                x = self.replace_none_with_zeros(x)
            return jnp.concatenate([
                a.reshape(shape + (a.shape[-2] * a.shape[-1],))
                for a in x
            ], axis=-1)
        return x

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def transform_by_angles(self, x, alpha, beta, gamma, k=0):
        r"""Rotate the data by angles according to the irreps

        Args:
            x (`jnp.ndarray` or list of optional `jnp.ndarray`): data compatible with the irreps.
            alpha (float): third rotation angle around the second axis (in radians)
            beta (float): second rotation angle around the first axis (in radians)
            gamma (float): first rotation angle around the second axis (in radians)
            k (int): parity operation

        Returns:
            `jnp.ndarray` or list of optional `jnp.ndarray`
        """
        shape = self.shape_of(x)
        D = {ir: ir.D_from_angles(alpha, beta, gamma, k) for ir in {ir for _mul, ir in self}}
        result = [
            jnp.reshape(jnp.einsum("ij,...uj->...ui", D[ir], x), shape + (mul, ir.dim))
            if x is not None else None
            for (mul, ir), x in zip(self, self.to_list(x))
        ]
        if isinstance(x, list):
            return result
        else:
            return self.to_contiguous(result)

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def transform_by_quaternion(self, x, q, k=0):
        r"""Rotate data by a rotation given by a quaternion

        Args:
            x (`jnp.ndarray` or list of optional `jnp.ndarray`): data compatible with the irreps.
            q (`jnp.ndarray`): quaternion
            k (int): parity operation

        Returns:
            `jnp.ndarray` or list of optional `jnp.ndarray`

        Examples:
            >>> irreps = Irreps("0e + 1o + 2e")
            >>> irreps.transform_by_quaternion(
            ...     jnp.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            ...     jnp.array([1.0, 0.0, 0.0, 0.0]),
            ...     1
            ... ) + 0.0
            DeviceArray([ 1.,  0., -1.,  0.,  0.,  0.,  1.,  0.,  0.], dtype=float32)
            >>> irreps.transform_by_quaternion(
            ...     [None, jnp.array([[1.0, 1.0, 1.0]]), None],
            ...     jnp.array([1.0, 0.0, 0.0, 0.0]),
            ...     1
            ... )
            [None, DeviceArray([[-1., -1., -1.]], dtype=float32), None]
        """
        return self.transform_by_angles(x, *quaternion_to_angles(q), k)

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def transform_by_matrix(self, x, R):
        r"""Rotate data by a rotation given by a matrix

        Args:
            x (`jnp.ndarray` or list of optional `jnp.ndarray`): data compatible with the irreps.
            R (`jnp.ndarray`): rotation matrix

        Returns:
            `jnp.ndarray` or list of optional `jnp.ndarray`
        """
        d = jnp.sign(jnp.linalg.det(R))
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.transform_by_angles(x, *matrix_to_angles(R), k)

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def D_from_angles(self, alpha, beta, gamma, k=0):
        r"""Compute the D matrix from the angles

        Args:
            alpha (float): third rotation angle around the second axis (in radians)
            beta (float): second rotation angle around the first axis (in radians)
            gamma (float): first rotation angle around the second axis (in radians)
            k (int): parity operation

        Returns:
            `jnp.ndarray`: array of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return jax.scipy.linalg.block_diag(*[ir.D_from_angles(alpha, beta, gamma, k) for mul, ir in self for _ in range(mul)])

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def D_from_quaternion(self, q, k=0):
        r"""Matrix of the representation

        Args:
            q (`jnp.ndarray`): array of shape :math:`(..., 4)`
            k (`jnp.ndarray`, optional): array of shape :math:`(...)`

        Returns:
            `jnp.ndarray`: array of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return self.D_from_angles(*quaternion_to_angles(q), k)

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def D_from_matrix(self, R):
        r"""Matrix of the representation

        Args:
            R (`jnp.ndarray`): array of shape :math:`(..., 3, 3)`

        Returns:
            `jnp.ndarray`: array of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        d = jnp.sign(jnp.linalg.det(R))
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.D_from_angles(*matrix_to_angles(R), k)
