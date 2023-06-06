import collections
import dataclasses
import itertools
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.scipy

from e3nn_jax import (
    axis_angle_to_log_coordinates,
    generators,
    matrix_to_angles,
    perm,
    quaternion_to_angles,
)

from .J import Jd

IntoIrrep = Union[int, "Irrep", "MulIrrep", Tuple[int, int]]


@dataclasses.dataclass(init=False, frozen=True)
class Irrep:
    r"""Irreducible representation of :math:`O(3)`.

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

    def __init__(self, l: IntoIrrep, p=None):
        """Initialize an Irrep."""
        if p is None:
            if isinstance(l, Irrep):
                p = l.p
                l = l.l

            if isinstance(l, MulIrrep):
                p = l.ir.p
                l = l.ir.l

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
        """Representation of the Irrep."""
        p = {+1: "e", -1: "o"}[self.p]
        return f"{self.l}{p}"

    @classmethod
    def iterator(cls, lmax=None):
        r"""Iterator through all the irreps of :math:`O(3)`.

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

    def D_from_log_coordinates(self, log_coordinates, k=0):
        r"""Matrix :math:`p^k D^l(\alpha)`.

        (matrix) Representation of :math:`O(3)`. :math:`D` is the representation of :math:`SO(3)`.

        Args:
            log_coordinates (`jax.numpy.ndarray`): of shape :math:`(..., 3)`
            k (optional `jax.numpy.ndarray`): of shape :math:`(...)`
                How many times the parity is applied.

        Returns:
            `jax.numpy.ndarray`: of shape :math:`(..., 2l+1, 2l+1)`

        See Also:
            Irreps.D_from_log_coordinates
        """
        k = jnp.asarray(k)
        shape = jnp.broadcast_shapes(log_coordinates.shape[:-1], k.shape)
        log_coordinates = jnp.broadcast_to(log_coordinates, shape + (3,))
        k = jnp.broadcast_to(k, shape)
        return (
            _wigner_D_from_log_coordinates(self.l, log_coordinates)
            * self.p ** k[..., None, None]
        )

    def D_from_angles(self, alpha, beta, gamma, k=0):
        r"""Matrix :math:`p^k D^l(\alpha, \beta, \gamma)`.

        (matrix) Representation of :math:`O(3)`. :math:`D` is the representation of :math:`SO(3)`.

        Args:
            alpha (`jax.numpy.ndarray`): of shape :math:`(...)`
                Rotation :math:`\alpha` around Y axis, applied third.
            beta (`jax.numpy.ndarray`): of shape :math:`(...)`
                Rotation :math:`\beta` around X axis, applied second.
            gamma (`jax.numpy.ndarray`): of shape :math:`(...)`
                Rotation :math:`\gamma` around Y axis, applied first.
            k (optional `jax.numpy.ndarray`): of shape :math:`(...)`
                How many times the parity is applied.

        Returns:
            `jax.numpy.ndarray`: of shape :math:`(..., 2l+1, 2l+1)`

        See Also:
            Irreps.D_from_angles
        """
        k = jnp.asarray(k)
        if isinstance(alpha, (int, float)) and alpha == 0:
            alpha = None
        else:
            alpha = jnp.asarray(alpha)

        if isinstance(beta, (int, float)) and beta == 0:
            beta = None
        else:
            beta = jnp.asarray(beta)

        if isinstance(gamma, (int, float)) and gamma == 0:
            gamma = None
        else:
            gamma = jnp.asarray(gamma)

        shape = jnp.broadcast_shapes(
            *[a.shape for a in [alpha, beta, gamma] if a is not None], k.shape
        )

        if alpha is not None:
            alpha = jnp.broadcast_to(alpha, shape)
        if beta is not None:
            beta = jnp.broadcast_to(beta, shape)
        if gamma is not None:
            gamma = jnp.broadcast_to(gamma, shape)
        k = jnp.broadcast_to(k, shape)

        return (
            _wigner_D_from_angles(self.l, alpha, beta, gamma)
            * self.p ** k[..., None, None]
        )

    def D_from_quaternion(self, q, k=0):
        r"""Matrix of the representation, see `Irrep.D_from_angles`.

        Args:
            q (`jax.numpy.ndarray`): shape :math:`(..., 4)`
            k (optional `jax.numpy.ndarray`): shape :math:`(...)`

        Returns:
            `jax.numpy.ndarray`: shape :math:`(..., 2l+1, 2l+1)`
        """
        return self.D_from_angles(*quaternion_to_angles(q), k)

    def D_from_matrix(self, R):
        r"""Matrix of the representation.

        Args:
            R (`jax.numpy.ndarray`): array of shape :math:`(..., 3, 3)`
            k (`jax.numpy.ndarray`, optional): array of shape :math:`(...)`

        Returns:
            `jax.numpy.ndarray`: array of shape :math:`(..., 2l+1, 2l+1)`

        Examples:
            >>> m = Irrep(1, -1).D_from_matrix(-jnp.eye(3))
            >>> m + 0.0
            Array([[-1.,  0.,  0.],
                   [ 0., -1.,  0.],
                   [ 0.,  0., -1.]], dtype=float32)

        See Also:
            `Irrep.D_from_angles`
        """
        d = jnp.sign(jnp.linalg.det(R))
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.D_from_angles(*matrix_to_angles(R), k)

    def D_from_axis_angle(self, axis, angle, k=0):
        return self.D_from_log_coordinates(
            axis_angle_to_log_coordinates(axis, angle), k
        )

    def generators(self):
        r"""Generators of the representation of :math:`SO(3)`.

        Returns:
            `jax.numpy.ndarray`: array of shape :math:`(3, 2l+1, 2l+1)`

        See Also:
            `generators`
        """
        return generators(self.l)

    @property
    def dim(self) -> int:
        """The dimension of the representation, :math:`2 l + 1`."""
        return 2 * self.l + 1

    def is_scalar(self) -> bool:
        """Equivalent to ``l == 0 and p == 1``."""
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
        r"""Integer times the irrep.

        >>> 3 * Irrep('1e')
        3x1e
        """
        assert isinstance(other, int)
        return Irreps([(other, self)])

    def __add__(self, other):
        r"""Sum of two irreps."""
        return Irreps(self) + Irreps(other)

    def __radd__(self, other):
        r"""Sum of two irreps."""
        return Irreps(other) + Irreps(self)

    def __iter__(self):
        r"""Deconstruct the irrep into ``l`` and ``p``."""
        yield self.l
        yield self.p

    def __lt__(self, other):
        r"""Compare the order of two irreps."""
        return (self.l, -self.p * (-1) ** self.l) < (
            other.l,
            -other.p * (-1) ** other.l,
        )

    def __eq__(self, other: object) -> bool:
        """Compare two irreps."""
        other = Irrep(other)
        return (self.l, self.p) == (other.l, other.p)


jax.tree_util.register_pytree_node(Irrep, lambda ir: ((), ir), lambda ir, _: ir)


@dataclasses.dataclass(init=False, frozen=True)
class MulIrrep:
    r"""An Irrep with a multiplicity."""
    mul: int
    ir: Irrep

    def __init__(self, mul, ir=None):
        r"""An irrep with a multiplicity."""
        if ir is None:
            mul, ir = mul

        object.__setattr__(self, "mul", mul)
        object.__setattr__(self, "ir", ir)

    @property
    def dim(self) -> int:
        """The dimension of the representations."""
        return self.mul * self.ir.dim

    def __repr__(self):
        """Representation of the irrep."""
        return f"{self.mul}x{self.ir}"

    def __iter__(self):
        """Deconstruct the mulirrep into ``mul`` and ``ir``."""
        yield self.mul
        yield self.ir

    def __lt__(self, other):
        """Compare the order of two mulirreps."""
        return (self.ir, self.mul) < (other.ir, other.mul)


jax.tree_util.register_pytree_node(
    MulIrrep, lambda mulir: ((), mulir), lambda mulir, _: mulir
)

IntoIrreps = Union[
    None,
    Irrep,
    MulIrrep,
    str,
    "Irreps",
    List[
        Union[
            str,
            Irrep,
            MulIrrep,
            Tuple[int, IntoIrrep],
        ]
    ],
]


class Irreps(tuple):
    r"""Direct sum of irreducible representations of :math:`O(3)`.

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
        (Irreps(), Irreps())
    """

    def __new__(cls, irreps: IntoIrreps = None):
        r"""Create a new Irreps object."""
        if isinstance(irreps, Irreps):
            return super().__new__(cls, irreps)

        out: List[MulIrrep] = []
        if isinstance(irreps, Irrep):
            out.append(MulIrrep(1, Irrep(irreps)))
        elif irreps is None:
            pass
        elif isinstance(irreps, MulIrrep):
            out.append(irreps)
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
        r"""Representation of the spherical harmonics.

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

    def slices(self) -> List[slice]:
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

    def __getitem__(self, i) -> Union[MulIrrep, "Irreps"]:
        r"""Indexing."""
        x = super().__getitem__(i)
        if isinstance(i, slice):
            return Irreps(x)
        return x

    def __contains__(self, ir) -> bool:
        r"""Check if an irrep is in the representation."""
        ir = Irrep(ir)
        return ir in (irrep for _, irrep in self)

    def count(self, ir: IntoIrrep) -> int:
        r"""Multiplicity of ``ir``.

        Args:
            ir (`Irrep`):

        Returns:
            `int`: total multiplicity of ``ir``

        Examples:
            >>> Irreps("2x0e + 3x1o").count("1o")
            3
        """
        ir = Irrep(ir)
        return sum(mul for mul, irrep in self if ir == irrep)

    def is_scalar(self) -> bool:
        r"""Check if the representation is scalar.

        Returns:
            `bool`: ``True`` if the representation is scalar

        Examples:
            >>> Irreps("2x0e + 3x1o").is_scalar()
            False
            >>> Irreps("2x0e + 2x0e").is_scalar()
            True
            >>> Irreps("0o").is_scalar()
            False
        """
        return {ir for _, ir in self} == {Irrep("0e")}

    def index(self, _object):  # noqa: D102
        raise NotImplementedError

    def __add__(self, irreps):
        r"""Add two representations."""
        irreps = Irreps(irreps)
        return Irreps(super().__add__(irreps))

    def __radd__(self, irreps):
        r"""Add two representations."""
        return Irreps(irreps) + self

    def __mul__(self, other):
        r"""Multiply the multiplicities of the irreps.

        Examples:
            >>> Irreps('0e + 1e') * 2
            2x0e+2x1e
        """
        if isinstance(other, Irreps):
            raise NotImplementedError(
                "Use e3nn.tensor_product for this, see the documentation"
            )
        return Irreps([(mul * other, ir) for mul, ir in self])

    def __rmul__(self, other):
        r"""Multiply the multiplicities of the irreps.

        Examples:
            >>> 2 * Irreps('0e + 1e')
            2x0e+2x1e
        """
        return self * other

    def __floordiv__(self, other):
        r"""Divide the multiplicities of the irreps.

        Examples:
            >>> Irreps('12x0e + 14x1e') // 2
            6x0e+7x1e
        """
        return Irreps([(mul // other, ir) for mul, ir in self])

    def __eq__(self, other: object) -> bool:
        r"""Check if two representations are equal."""
        if isinstance(other, str):
            try:
                other = Irreps(other)
            except ValueError:
                return False
        return super().__eq__(other)

    def __hash__(self) -> int:
        r"""Hash of the representation."""
        return super().__hash__()

    def repeat(self, n: int) -> "Irreps":
        r"""Repeat the representation ``n`` times.

        Examples:
            >>> Irreps('0e + 1e').repeat(2)
            1x0e+1x1e+1x0e+1x1e
        """
        return Irreps([(mul, ir) for mul, ir in self] * n)

    def unify(self) -> "Irreps":
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

    def remove_zero_multiplicities(self) -> "Irreps":
        """Remove any irreps with multiplicities of zero.

        Examples:
            >>> Irreps("4x0e + 0x1o + 2x3e").remove_zero_multiplicities()
            4x0e+2x3e
        """
        return Irreps([(mul, ir) for mul, ir in self if mul > 0])

    def simplify(self) -> "Irreps":
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

    def sort(
        self,
    ) -> NamedTuple("Sort", irreps="Irreps", p=Tuple[int, ...], inv=Tuple[int, ...]):
        r"""Sort the representations.

        Returns:
            (tuple): tuple containing:

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

    def regroup(self) -> "Irreps":
        r"""Regroup the same irreps together.

        Equivalent to :meth:`sort` followed by :meth:`simplify`.

        Returns:
            `Irreps`: regrouped irreps

        Examples:
            >>> Irreps("1e + 0e + 1e + 0x2e").regroup()
            1x0e+2x1e
        """
        return self.sort().irreps.simplify()

    def set_mul(self, mul: int) -> "Irreps":
        r"""Set the multiplicities to one.

        Examples:
            >>> Irreps("2x0e + 1x1e").set_mul(1)
            1x0e+1x1e
        """
        return Irreps([(mul, ir) for _, ir in self])

    def filter(
        self,
        keep: Union["Irreps", List[Irrep], Callable[[MulIrrep], bool]] = None,
        *,
        drop: Union["Irreps", List[Irrep], Callable[[MulIrrep], bool]] = None,
        lmax: int = None,
    ) -> "Irreps":
        r"""Filter the irreps.

        Args:
            keep (`Irreps` or list of `Irrep` or function): list of irrep to keep
            drop (`Irreps` or list of `Irrep` or function): list of irrep to drop
            lmax (int): maximum :math:`l` value

        Returns:
            `Irreps`: filtered irreps

        Examples:
            >>> Irreps("1e + 2e + 0e").filter(keep=["0e", "1e"])
            1x1e+1x0e

            >>> Irreps("1e + 2e + 0e").filter(keep="2e + 2x1e")
            1x1e+1x2e

            >>> Irreps("1e + 2e + 0e").filter(drop="2e + 2x1e")
            1x0e

            >>> Irreps("1e + 2e + 0e").filter(lmax=1)
            1x1e+1x0e
        """
        if keep is None and drop is None and lmax is None:
            return self
        if keep is not None and drop is not None:
            raise ValueError("Cannot specify both keep and drop")
        if keep is not None and lmax is not None:
            raise ValueError("Cannot specify both keep and lmax")
        if drop is not None and lmax is not None:
            raise ValueError("Cannot specify both drop and lmax")

        if keep is not None:
            if isinstance(keep, str):
                keep = Irreps(keep)
            if isinstance(keep, Irrep):
                keep = [keep]
            if callable(keep):
                return Irreps([mul_ir for mul_ir in self if keep(mul_ir)])
            keep = {Irrep(ir) for ir in keep}
            return Irreps([(mul, ir) for mul, ir in self if ir in keep])

        if drop is not None:
            if isinstance(drop, str):
                drop = Irreps(drop)
            if isinstance(drop, Irrep):
                drop = [drop]
            if callable(drop):
                return Irreps([mul_ir for mul_ir in self if not drop(mul_ir)])
            drop = {Irrep(ir) for ir in drop}
            return Irreps([(mul, ir) for mul, ir in self if ir not in drop])

        if lmax is not None:
            return Irreps([(mul, ir) for mul, ir in self if ir.l <= lmax])

    @property
    def slice_by_mul(self):
        r"""Return the slice with respect to the multiplicities.

        Examples:
            >>> Irreps("2x1e + 2e").slice_by_mul[2:]
            1x2e

            >>> Irreps("1e + 2e + 3x0e").slice_by_mul[1:3]
            1x2e+1x0e

            >>> Irreps("1e + 2e + 3x0e").slice_by_mul[1:]
            1x2e+3x0e
        """
        return _MulIndexSliceHelper(self)

    @property
    def slice_by_dim(self):
        r"""Return the slice with respect to the dimensions.

        Examples:
            >>> Irreps("1e + 2e + 3x0e").slice_by_dim[:3]
            1x1e

            >>> Irreps("1e + 2e + 3x0e").slice_by_dim[3:8]
            1x2e
        """
        return _DimIndexSliceHelper(self)

    @property
    def slice_by_chunk(self):
        r"""Return the slice with respect to the chunks.

        Examples:
            >>> Irreps("2x1e + 2e + 3x0e").slice_by_chunk[:1]
            2x1e

            >>> Irreps("1e + 2e + 3x0e").slice_by_chunk[1:]
            1x2e+3x0e
        """
        return _ChunkIndexSliceHelper(self)

    @property
    def dim(self) -> int:
        r"""Dimension of the irreps.

        Examples:
            >>> Irreps("3x0e + 2x1e").dim
            9
        """
        return sum(mul * ir.dim for mul, ir in self)

    @property
    def num_irreps(self) -> int:
        """Sum of the multiplicities.

        Examples:
            >>> Irreps("3x0e + 2x1e").num_irreps
            5
        """
        return sum(mul for mul, _ in self)

    @property
    def ls(self) -> List[int]:
        """List of the l values.

        Examples:
            >>> Irreps("3x0e + 2x1e").ls
            [0, 0, 0, 1, 1]
        """
        return [l for mul, (l, p) in self for _ in range(mul)]

    @property
    def lmax(self) -> int:
        """Maximum l value.

        Examples:
            >>> Irreps("3x0e + 2x1e").lmax
            1
        """
        if len(self) == 0:
            raise ValueError("Cannot get lmax of empty Irreps")
        return max(self.ls)

    def __repr__(self):
        """Representation of the irreps."""
        if len(self) == 0:
            return "Irreps()"
        return "+".join(f"{mul_ir}" for mul_ir in self)

    def D_from_log_coordinates(self, log_coordinates, k=0):
        r"""Matrix of the representation.

        Args:
            log_coordinates (`jax.numpy.ndarray`): array of shape :math:`(..., 3)`
            k (`jax.numpy.ndarray`, optional): array of shape :math:`(...)`

        Returns:
            `jax.numpy.ndarray`: array of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return jax.scipy.linalg.block_diag(
            *[
                ir.D_from_log_coordinates(log_coordinates, k)
                for mul, ir in self
                for _ in range(mul)
            ]
        )

    def D_from_angles(self, alpha, beta, gamma, k=0):
        r"""Compute the D matrix from the angles.

        Args:
            alpha (float): third rotation angle around the second axis (in radians)
            beta (float): second rotation angle around the first axis (in radians)
            gamma (float): first rotation angle around the second axis (in radians)
            k (int): parity operation

        Returns:
            `jax.numpy.ndarray`: array of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return jax.scipy.linalg.block_diag(
            *[
                ir.D_from_angles(alpha, beta, gamma, k)
                for mul, ir in self
                for _ in range(mul)
            ]
        )

    def D_from_quaternion(self, q, k=0):
        r"""Matrix of the representation.

        Args:
            q (`jax.numpy.ndarray`): array of shape :math:`(..., 4)`
            k (`jax.numpy.ndarray`, optional): array of shape :math:`(...)`

        Returns:
            `jax.numpy.ndarray`: array of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return self.D_from_angles(*quaternion_to_angles(q), k)

    def D_from_matrix(self, R):
        r"""Matrix of the representation.

        Args:
            R (`jax.numpy.ndarray`): array of shape :math:`(..., 3, 3)`

        Returns:
            `jax.numpy.ndarray`: array of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        d = jnp.sign(jnp.linalg.det(R))
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.D_from_angles(*matrix_to_angles(R), k)

    def D_from_axis_angle(self, axis, angle, k=0):
        return self.D_from_log_coordinates(
            axis_angle_to_log_coordinates(axis, angle), k
        )

    def generators(self) -> jnp.ndarray:
        r"""Generators of the representation.

        Returns:
            `jax.numpy.ndarray`: array of shape :math:`(3, \mathrm{dim}, \mathrm{dim})`
        """
        return jax.vmap(jax.scipy.linalg.block_diag)(
            *[ir.generators() for mul, ir in self for _ in range(mul)]
        )


jax.tree_util.register_pytree_node(
    Irreps, lambda irreps: ((), irreps), lambda irreps, _: irreps
)


class _MulIndexSliceHelper:
    irreps: Irreps

    def __init__(self, irreps) -> None:
        self.irreps = irreps

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError("Irreps.slice_by_mul only supports slices.")

        start, stop, stride = index.indices(self.irreps.num_irreps)
        if stride != 1:
            raise NotImplementedError("Irreps.slice_by_mul does not support strides.")

        out = []
        i = 0
        for mul, ir in self.irreps:
            if start <= i and i + mul <= stop:
                out.append((mul, ir))
            elif start < i + mul and i < stop:
                out.append((min(stop, i + mul) - max(start, i), ir))
            i += mul
        return Irreps(out)


class _DimIndexSliceHelper:
    irreps: Irreps

    def __init__(self, irreps) -> None:
        self.irreps = irreps

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError("Irreps.slice_by_dim only supports slices.")

        start, stop, stride = index.indices(self.irreps.dim)
        if stride != 1:
            raise NotImplementedError("Irreps.slice_by_dim does not support strides.")

        out = []
        i = 0
        for mul, ir in self.irreps:
            if start <= i and i + mul * ir.dim <= stop:
                out.append((mul, ir))
            elif start < i + mul * ir.dim and i < stop:
                dim = min(stop, i + mul * ir.dim) - max(start, i)
                if dim % ir.dim != 0:
                    raise ValueError(
                        f"Error in Irreps.slice_by_dim: {start}:{stop} is not a valid slice for irreps {self.irreps} "
                        f"because it does not split {mul}x{ir} in an equivariant way."
                    )
                out.append((dim // ir.dim, ir))
            i += mul * ir.dim
        return Irreps(out)


class _ChunkIndexSliceHelper:
    irreps: Irreps

    def __init__(self, irreps) -> None:
        self.irreps = irreps

    def __getitem__(self, index: slice) -> Irreps:
        if not isinstance(index, slice):
            raise IndexError("Irreps.slice_by_chunk only supports slices.")

        return Irreps(self.irreps[index])


def _wigner_D_from_angles(
    l: int,
    alpha: Optional[jnp.ndarray],
    beta: Optional[jnp.ndarray],
    gamma: Optional[jnp.ndarray],
) -> jnp.ndarray:
    r"""The Wigner-D matrix of the real irreducible representations of :math:`SO(3)`.

    Args:
        l (int): the representation order of the irrep
        alpha (jnp.ndarray): the first Euler angle
        beta (jnp.ndarray): the second Euler angle
        gamma (jnp.ndarray): the third Euler angle

    Returns:
        jnp.ndarray: the Wigner-D matrix
    """
    shape = ()
    if alpha is not None:
        alpha = alpha % (2 * jnp.pi)
        shape = alpha.shape
    if beta is not None:
        beta = beta % (2 * jnp.pi)
        shape = beta.shape
    if gamma is not None:
        gamma = gamma % (2 * jnp.pi)
        shape = gamma.shape

    def f(a, b, c):
        def rot_y(phi):
            M = jnp.zeros((2 * l + 1, 2 * l + 1), dtype=phi.dtype)
            inds = jnp.arange(0, 2 * l + 1, 1)
            reversed_inds = jnp.arange(2 * l, -1, -1)
            frequencies = jnp.arange(l, -l - 1, -1.0, dtype=phi.dtype)
            M = M.at[inds, reversed_inds].set(jnp.sin(frequencies * phi))
            M = M.at[inds, inds].set(jnp.cos(frequencies * phi))
            return M

        R = []
        if l < len(Jd):
            if a is not None:
                R += [rot_y(a)]
            if b is not None:
                J = Jd[l]
                R += [J @ rot_y(b) @ J]
            if c is not None:
                R += [rot_y(c)]
        else:
            X = generators(l)
            exp = jax.scipy.linalg.expm

            if a is not None:
                R += [exp(a * X[1])]
            if b is not None:
                R += [exp(b * X[0])]
            if c is not None:
                R += [exp(c * X[1])]

        if len(R) == 0:
            return jnp.eye(2 * l + 1)

        r = R[0]
        for r_ in R[1:]:
            r = r @ r_
        return r

    f_vec = f
    for _ in range(len(shape)):
        f_vec = jax.vmap(f_vec)

    return f_vec(alpha, beta, gamma)


def _wigner_D_from_log_coordinates(l: int, log_coordinates: jnp.ndarray) -> jnp.ndarray:
    r"""The Wigner-D matrix of the real irreducible representations of :math:`SO(3)`.

    Args:
        l (int): the representation order of the irrep
        log_coordinates (jnp.ndarray): the log coordinates

    Returns:
        jnp.ndarray: the Wigner-D matrix
    """
    X = generators(l)

    def func(log_coordinates):
        return jax.scipy.linalg.expm(jnp.einsum("a,aij->ij", log_coordinates, X))

    f = func
    for _ in range(log_coordinates.ndim - 1):
        f = jax.vmap(f)

    return f(log_coordinates)
