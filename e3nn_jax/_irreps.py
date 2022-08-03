import collections
import dataclasses
import itertools
from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import jax.scipy

from e3nn_jax import axis_angle_to_angles, config, matrix_to_angles, perm, quaternion_to_angles, wigner_D


@dataclasses.dataclass(init=False, frozen=True)
class Irrep:
    r"""Irreducible representation of :math:`O(3)`

    This class does not contain any data, it is a structure that describe the representation.
    It is typically used as argument of other classes of the library to define the input and output
    representations of functions.

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
    l: int
    p: int

    def __init__(self, l, p=None):
        if p is None:
            if isinstance(l, Irrep):
                p = l.p
                l = l.l

            if isinstance(l, str):
                try:
                    name = l.strip()
                    l = int(name[:-1])
                    assert l >= 0
                    p = {
                        "e": 1,
                        "o": -1,
                        "y": (-1) ** l,
                    }[name[-1]]
                except Exception:
                    raise ValueError(f'unable to convert string "{name}" into an Irrep')
            elif isinstance(l, tuple):
                l, p = l

        assert isinstance(l, int) and l >= 0, l
        assert p in [-1, 1], p
        object.__setattr__(self, "l", l)
        object.__setattr__(self, "p", p)

    def __repr__(self):
        p = {+1: "e", -1: "o"}[self.p]
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
            yield Irrep(l, (-1) ** l)
            yield Irrep(l, -((-1) ** l))

            if l == lmax:
                break

    def D_from_angles(self, alpha, beta, gamma, k=0):
        r"""Matrix :math:`p^k D^l(\alpha, \beta, \gamma)`

        (matrix) Representation of :math:`O(3)`. :math:`D` is the representation of :math:`SO(3)`, see `wigner_D`.

        Args:
            alpha (``jnp.ndarray``): of shape :math:`(...)`
                Rotation :math:`\alpha` around Y axis, applied third.
            beta (``jnp.ndarray``): of shape :math:`(...)`
                Rotation :math:`\beta` around X axis, applied second.
            gamma (``jnp.ndarray``): of shape :math:`(...)`
                Rotation :math:`\gamma` around Y axis, applied first.
            k (optional ``jnp.ndarray``): of shape :math:`(...)`
                How many times the parity is applied.

        Returns:
            ``jnp.ndarray``: of shape :math:`(..., 2l+1, 2l+1)`

        See Also:
            o3.wigner_D
            Irreps.D_from_angles
        """
        alpha, beta, gamma, k = jnp.broadcast_arrays(alpha, beta, gamma, k)
        k = jnp.asarray(k)
        return wigner_D(self.l, alpha, beta, gamma) * self.p ** k[..., None, None]

    def D_from_quaternion(self, q, k=0):
        r"""Matrix of the representation, see `Irrep.D_from_angles`

        Args:
            q (``jnp.ndarray``): shape :math:`(..., 4)`
            k (optional ``jnp.ndarray``): shape :math:`(...)`

        Returns:
            ``jnp.ndarray``: shape :math:`(..., 2l+1, 2l+1)`
        """
        return self.D_from_angles(*quaternion_to_angles(q), k)

    def D_from_matrix(self, R):
        r"""Matrix of the representation

        Args:
            R (``jnp.ndarray``): array of shape :math:`(..., 3, 3)`
            k (``jnp.ndarray``, optional): array of shape :math:`(...)`

        Returns:
            ``jnp.ndarray``: array of shape :math:`(..., 2l+1, 2l+1)`

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

    def __rmul__(self, other):
        r"""
        >>> 3 * Irrep('1e')
        3x1e
        """
        assert isinstance(other, int)
        return Irreps([(other, self)])

    def __add__(self, other):
        return Irreps(self) + Irreps(other)

    def __iter__(self):
        yield self.l
        yield self.p

    def __lt__(self, other):
        return (self.l, self.p) < (other.l, other.p)


jax.tree_util.register_pytree_node(Irrep, lambda ir: ((), ir), lambda ir, _: ir)


@dataclasses.dataclass(init=False, frozen=True)
class MulIrrep:
    mul: int
    ir: Irrep

    def __init__(self, mul, ir=None):
        if ir is None:
            mul, ir = mul

        object.__setattr__(self, "mul", mul)
        object.__setattr__(self, "ir", ir)

    @property
    def dim(self) -> int:
        return self.mul * self.ir.dim

    def __repr__(self):
        return f"{self.mul}x{self.ir}"

    def __iter__(self):
        yield self.mul
        yield self.ir

    def __lt__(self, other):
        return (self.mul, self.ir) < (other.mul, other.ir)


jax.tree_util.register_pytree_node(MulIrrep, lambda mulir: ((), mulir), lambda mulir, _: mulir)


class Irreps(tuple):
    r"""Direct sum of irreducible representations of :math:`O(3)`

    This class does not contain any data, it is a structure that describe the representation.
    It is typically used as argument of other classes of the library to define the input and output
    representations of functions.

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
            out.append(MulIrrep(1, Irrep(irreps)))
        elif isinstance(irreps, str):
            try:
                if irreps.strip() != "":
                    for mul_ir in irreps.split("+"):
                        if "x" in mul_ir:
                            mul, ir = mul_ir.split("x")
                            mul = int(mul)
                            ir = Irrep(ir)
                        else:
                            mul = 1
                            ir = Irrep(mul_ir)

                        assert isinstance(mul, int) and mul >= 0
                        out.append(MulIrrep(mul, ir))
            except Exception:
                raise ValueError(f'Unable to convert string "{irreps}" into an Irreps')
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
                elif isinstance(mul_ir, MulIrrep):
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
                    raise ValueError(f'Unable to interpret "{mul_ir}" as an irrep.')

                out.append(MulIrrep(mul, ir))
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

    def randn(self, key, size, *, normalization=None):
        r"""Random tensor.

        Args:
            *size (list of int): size of the output tensor, needs to contains a ``-1``
            normalization : {'component', 'norm'}

        Returns:
            ``jnp.ndarray``: array of shape ``size`` where ``-1`` is replaced by ``self.dim``

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
        rsize = size[di + 1 :]

        if normalization is None:
            normalization = config("irrep_normalization")

        if normalization == "component":
            return jax.random.normal(key, lsize + (self.dim,) + rsize)
        elif normalization == "norm":
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

    def unify(self):
        r"""Regroup same irrep together.

        Returns:
            `Irreps`: new `Irreps` object

        Examples:
            >>> Irreps('0e + 1e').unify()
            1x0e+1x1e

            >>> Irreps('0e + 1e + 1e').unify()
            1x0e+2x1e

            >>> Irreps('0e + 0x1e + 0e').unify()
            1x0e+0x1e+1x0e
        """
        out = []
        for mul, ir in self:
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            else:
                out.append((mul, ir))
        return Irreps(out)

    def remove_zero_multiplicities(self):
        """Remove any irreps with multiplicities of zero.

        Examples:
            >>> Irreps("4x0e + 0x1o + 2x3e").remove_zero_multiplicities()
            4x0e+2x3e
        """
        return Irreps([(mul, ir) for mul, ir in self if mul > 0])

    def simplify(self):
        """Simplify the representations.

        Examples:
            Note that simplify does not sort the representations.

            >>> Irreps("1e + 1e + 0e").simplify()
            2x1e+1x0e

            Equivalent representations which are separated from each other are not combined.

            >>> Irreps("1e + 1e + 0e + 1e").simplify()
            2x1e+1x0e+1x1e

            Except if they are separated by an irrep with multiplicity of zero.

            >>> Irreps("1e + 0x0e + 1e").simplify().simplify()
            2x1e
        """
        return self.remove_zero_multiplicities().unify()

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

    def filter(self, ir_list):
        r"""Filter the irreps.

        Args:
            ir_list (list of `Irrep`): list of irrep to keep

        Returns:
            `Irreps`: filtered irreps

        Examples:
            >>> Irreps("1e + 2e + 0e").filter(["0e", "1e"])
            1x1e+1x0e
        """
        if isinstance(ir_list, (str, Irrep)):
            ir_list = [ir_list]
        ir_list = {Irrep(ir) for ir in ir_list}
        return Irreps([(mul, ir) for mul, ir in self if ir in ir_list])

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

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def transform_by_angles(self, array, alpha, beta, gamma, k=0):
        r"""Rotate the data by angles according to the irreps

        Args:
            array (``jnp.ndarray``): data compatible with the irreps.
            alpha (float): third rotation angle around the second axis (in radians)
            beta (float): second rotation angle around the first axis (in radians)
            gamma (float): first rotation angle around the second axis (in radians)
            k (int): parity operation

        Returns:
            ``jnp.ndarray``
        """
        assert array.shape[-1] == self.dim
        D = self.D_from_angles(alpha, beta, gamma, k)
        return jnp.einsum("ij,...j", D, array)

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def transform_by_quaternion(self, array, q, k=0):
        r"""Rotate data by a rotation given by a quaternion

        Args:
            array (``jnp.ndarray``): data compatible with the irreps.
            q (``jnp.ndarray``): quaternion
            k (int): parity operation

        Returns:
            ``jnp.ndarray``

        Examples:
            >>> irreps = Irreps("0e + 1o + 2e")
            >>> irreps.transform_by_quaternion(
            ...     jnp.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            ...     jnp.array([1.0, 0.0, 0.0, 0.0]),
            ...     1
            ... ) + 0.0
            DeviceArray([ 1.,  0., -1.,  0.,  0.,  0.,  1.,  0.,  0.], dtype=float32)
        """
        return self.transform_by_angles(array, *quaternion_to_angles(q), k)

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def transform_by_axis_angle(self, array, axis, angle, k=0):
        r"""Rotate data by an axis and an angle

        Args:
            array (``jnp.ndarray``): data compatible with the irreps.
            axis (``jnp.ndarray``): axis
            angle (float): angle (in radians)
            k (int): parity operation

        Returns:
            ``jnp.ndarray``
        """
        return self.transform_by_angles(array, *axis_angle_to_angles(axis, angle), k)

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def transform_by_matrix(self, array, R):
        r"""Rotate data by a rotation given by a matrix

        Args:
            array (``jnp.ndarray``): data compatible with the irreps.
            R (``jnp.ndarray``): rotation matrix

        Returns:
            ``jnp.ndarray``
        """
        d = jnp.sign(jnp.linalg.det(R))
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.transform_by_angles(array, *matrix_to_angles(R), k)

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def D_from_angles(self, alpha, beta, gamma, k=0):
        r"""Compute the D matrix from the angles

        Args:
            alpha (float): third rotation angle around the second axis (in radians)
            beta (float): second rotation angle around the first axis (in radians)
            gamma (float): first rotation angle around the second axis (in radians)
            k (int): parity operation

        Returns:
            ``jnp.ndarray``: array of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return jax.scipy.linalg.block_diag(*[ir.D_from_angles(alpha, beta, gamma, k) for mul, ir in self for _ in range(mul)])

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def D_from_quaternion(self, q, k=0):
        r"""Matrix of the representation

        Args:
            q (``jnp.ndarray``): array of shape :math:`(..., 4)`
            k (``jnp.ndarray``, optional): array of shape :math:`(...)`

        Returns:
            ``jnp.ndarray``: array of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return self.D_from_angles(*quaternion_to_angles(q), k)

    @partial(jax.jit, static_argnums=(0,), inline=True)
    def D_from_matrix(self, R):
        r"""Matrix of the representation

        Args:
            R (``jnp.ndarray``): array of shape :math:`(..., 3, 3)`

        Returns:
            ``jnp.ndarray``: array of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        d = jnp.sign(jnp.linalg.det(R))
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.D_from_angles(*matrix_to_angles(R), k)


jax.tree_util.register_pytree_node(Irreps, lambda irreps: ((), irreps), lambda irreps, _: irreps)
