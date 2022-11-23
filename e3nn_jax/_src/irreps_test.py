import pytest
from e3nn_jax import Irrep, Irreps, MulIrrep


def test_creation():
    Irrep(3, 1)
    ir = Irrep("3e")
    Irrep(ir)
    assert Irrep("10o") == Irrep(10, -1)
    assert Irrep("1y") == Irrep("1o")

    irreps = Irreps(ir)
    Irreps(irreps)
    Irreps([(32, (4, -1))])
    Irreps("11e")
    assert Irreps("16x1e + 32 x 2o") == Irreps([(16, (1, 1)), (32, (2, -1))])
    Irreps(["1e", "2o"])
    Irreps([(16, "3e"), "1e"])
    Irreps([(16, "3e"), "1e", (256, (1, -1))])


def test_properties():
    irrep = Irrep("3e")
    assert irrep.l == 3
    assert irrep.p == 1
    assert irrep.dim == 7

    assert Irrep(repr(irrep)) == irrep

    l, p = Irrep("5o")
    assert l == 5
    assert p == -1

    iterator = Irrep.iterator(5)
    assert len(list(iterator)) == 12

    iterator = Irrep.iterator()
    for x in range(100):
        irrep = next(iterator)
        assert irrep.l == x // 2
        assert irrep.p in (-1, 1)
        assert irrep.dim == 2 * (x // 2) + 1

    irreps = Irreps("4x1e + 6x2e + 12x2o")
    assert Irreps(repr(irreps)) == irreps


def test_arithmetic():
    assert 3 * Irrep("6o") == Irreps("3x6o")
    products = list(Irrep("1o") * Irrep("2e"))
    assert products == [Irrep("1o"), Irrep("2o"), Irrep("3o")]

    assert Irrep("4o") + Irrep("7e") == Irreps("4o + 7e")

    assert 2 * Irreps("2x2e + 4x1o") == Irreps("4x2e + 8x1o")
    assert Irreps("2x2e + 4x1o") * 2 == Irreps("4x2e + 8x1o")

    assert Irreps("1o + 4o") + Irreps("1o + 7e") == Irreps("1o + 4o + 1o + 7e")


def test_empty_irreps():
    assert Irreps() == Irreps("") == Irreps([])
    assert len(Irreps()) == 0
    assert Irreps().dim == 0
    assert Irreps().ls == []
    assert Irreps().num_irreps == 0


def test_getitem():
    irreps = Irreps("16x1e + 3e + 2e + 5o")
    assert irreps[0] == MulIrrep(16, Irrep("1e"))
    assert irreps[3] == MulIrrep(1, Irrep("5o"))
    assert irreps[-1] == MulIrrep(1, Irrep("5o"))

    sliced = irreps[2:]
    assert isinstance(sliced, Irreps)
    assert sliced == Irreps("2e + 5o")


def test_cat():
    irreps = Irreps("4x1e + 6x2e + 12x2o") + Irreps("1x1e + 2x2e + 12x4o")
    assert len(irreps) == 6
    assert irreps.ls == [1] * 4 + [2] * 6 + [2] * 12 + [1] * 1 + [2] * 2 + [4] * 12
    assert irreps.lmax == 4
    assert irreps.num_irreps == 4 + 6 + 12 + 1 + 2 + 12


def test_contains():
    assert Irrep("2e") in Irreps("3x0e + 2x2e + 1x3o")
    assert Irrep("2o") not in Irreps("3x0e + 2x2e + 1x3o")


def test_fail1():
    with pytest.raises(AssertionError):
        Irreps([(32, 1)])


def test_ordering():
    n_test = 100

    last = None
    for ir in Irrep.iterator():
        if last is not None:
            assert last < ir
        last = ir

        n_test -= 1
        if n_test == 0:
            break


def test_slice_by_mul():
    assert Irreps("10x0e").slice_by_mul[1:4] == Irreps("3x0e")
    assert Irreps("10x0e + 10x1e").slice_by_mul[5:15] == Irreps("5x0e + 5x1e")
    assert Irreps("10x0e + 2e + 10x1e").slice_by_mul[5:15] == Irreps("5x0e + 2e + 4x1e")


def test_slice_by_dim():
    assert Irreps("10x0e").slice_by_dim[1:4] == Irreps("3x0e")
    assert Irreps("10x0e + 10x1e").slice_by_dim[5:13] == Irreps("5x0e + 1e")
    assert Irreps("10x0e + 2e + 10x1e").slice_by_dim[5:18] == Irreps("5x0e + 2e + 1e")


def test_slice_by_chunk():
    assert Irreps("10x0e").slice_by_chunk[:1] == Irreps("10x0e")
    assert Irreps("10x0e + 10x1e").slice_by_chunk[1:2] == Irreps("10x1e")
    assert Irreps("10x0e + 5x1e + 5x1e").slice_by_chunk[1:2] == Irreps("5x1e")
