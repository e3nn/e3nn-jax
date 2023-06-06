import jax
import numpy as np
import pytest

import e3nn_jax as e3nn


def test_creation():
    e3nn.Irrep(3, 1)
    ir = e3nn.Irrep("3e")
    e3nn.Irrep(ir)
    assert e3nn.Irrep("10o") == e3nn.Irrep(10, -1)
    assert e3nn.Irrep("1y") == e3nn.Irrep("1o")

    irreps = e3nn.Irreps(ir)
    e3nn.Irreps(irreps)
    e3nn.Irreps([(32, (4, -1))])
    e3nn.Irreps("11e")
    assert e3nn.Irreps("16x1e + 32 x 2o") == e3nn.Irreps([(16, (1, 1)), (32, (2, -1))])
    e3nn.Irreps(["1e", "2o"])
    e3nn.Irreps([(16, "3e"), "1e"])
    e3nn.Irreps([(16, "3e"), "1e", (256, (1, -1))])


def test_properties():
    irrep = e3nn.Irrep("3e")
    assert irrep.l == 3
    assert irrep.p == 1
    assert irrep.dim == 7

    assert e3nn.Irrep(repr(irrep)) == irrep

    l, p = e3nn.Irrep("5o")
    assert l == 5
    assert p == -1

    iterator = e3nn.Irrep.iterator(5)
    assert len(list(iterator)) == 12

    iterator = e3nn.Irrep.iterator()
    for x in range(100):
        irrep = next(iterator)
        assert irrep.l == x // 2
        assert irrep.p in (-1, 1)
        assert irrep.dim == 2 * (x // 2) + 1

    irreps = e3nn.Irreps("4x1e + 6x2e + 12x2o")
    assert e3nn.Irreps(repr(irreps)) == irreps


def test_arithmetic():
    assert 3 * e3nn.Irrep("6o") == e3nn.Irreps("3x6o")
    products = list(e3nn.Irrep("1o") * e3nn.Irrep("2e"))
    assert products == [e3nn.Irrep("1o"), e3nn.Irrep("2o"), e3nn.Irrep("3o")]

    assert e3nn.Irrep("4o") + e3nn.Irrep("7e") == e3nn.Irreps("4o + 7e")

    assert 2 * e3nn.Irreps("2x2e + 4x1o") == e3nn.Irreps("4x2e + 8x1o")
    assert e3nn.Irreps("2x2e + 4x1o") * 2 == e3nn.Irreps("4x2e + 8x1o")

    assert e3nn.Irreps("1o + 4o") + e3nn.Irreps("1o + 7e") == e3nn.Irreps(
        "1o + 4o + 1o + 7e"
    )


def test_empty_irreps():
    assert e3nn.Irreps() == e3nn.Irreps("") == e3nn.Irreps([])
    assert len(e3nn.Irreps()) == 0
    assert e3nn.Irreps().dim == 0
    assert e3nn.Irreps().ls == []
    assert e3nn.Irreps().num_irreps == 0


def test_getitem():
    irreps = e3nn.Irreps("16x1e + 3e + 2e + 5o")
    assert irreps[0] == e3nn.MulIrrep(16, e3nn.Irrep("1e"))
    assert irreps[3] == e3nn.MulIrrep(1, e3nn.Irrep("5o"))
    assert irreps[-1] == e3nn.MulIrrep(1, e3nn.Irrep("5o"))

    sliced = irreps[2:]
    assert isinstance(sliced, e3nn.Irreps)
    assert sliced == e3nn.Irreps("2e + 5o")


def test_cat():
    irreps = e3nn.Irreps("4x1e + 6x2e + 12x2o") + e3nn.Irreps("1x1e + 2x2e + 12x4o")
    assert len(irreps) == 6
    assert irreps.ls == [1] * 4 + [2] * 6 + [2] * 12 + [1] * 1 + [2] * 2 + [4] * 12
    assert irreps.lmax == 4
    assert irreps.num_irreps == 4 + 6 + 12 + 1 + 2 + 12


def test_contains():
    assert e3nn.Irrep("2e") in e3nn.Irreps("3x0e + 2x2e + 1x3o")
    assert e3nn.Irrep("2o") not in e3nn.Irreps("3x0e + 2x2e + 1x3o")


def test_fail1():
    with pytest.raises(AssertionError):
        e3nn.Irreps([(32, 1)])


def test_ordering():
    n_test = 100

    last = None
    for ir in e3nn.Irrep.iterator():
        if last is not None:
            assert last < ir
        last = ir

        n_test -= 1
        if n_test == 0:
            break


def test_slice_by_mul():
    assert e3nn.Irreps("10x0e").slice_by_mul[1:4] == e3nn.Irreps("3x0e")
    assert e3nn.Irreps("10x0e + 10x1e").slice_by_mul[5:15] == e3nn.Irreps("5x0e + 5x1e")
    assert e3nn.Irreps("10x0e + 2e + 10x1e").slice_by_mul[5:15] == e3nn.Irreps(
        "5x0e + 2e + 4x1e"
    )


def test_slice_by_dim():
    assert e3nn.Irreps("10x0e").slice_by_dim[1:4] == e3nn.Irreps("3x0e")
    assert e3nn.Irreps("10x0e + 10x1e").slice_by_dim[5:13] == e3nn.Irreps("5x0e + 1e")
    assert e3nn.Irreps("10x0e + 2e + 10x1e").slice_by_dim[5:18] == e3nn.Irreps(
        "5x0e + 2e + 1e"
    )


def test_slice_by_chunk():
    assert e3nn.Irreps("10x0e").slice_by_chunk[:1] == e3nn.Irreps("10x0e")
    assert e3nn.Irreps("10x0e + 10x1e").slice_by_chunk[1:2] == e3nn.Irreps("10x1e")
    assert e3nn.Irreps("10x0e + 5x1e + 5x1e").slice_by_chunk[1:2] == e3nn.Irreps("5x1e")


@pytest.mark.parametrize("ir", ["0e", "1e", "2e", "3e", "4e", "12e"])
def test_D(keys, ir):
    jax.config.update("jax_enable_x64", True)

    ir = e3nn.Irrep(ir)
    angles = e3nn.rand_angles(keys[0])
    Da = ir.D_from_angles(*angles)
    w = e3nn.angles_to_log_coordinates(*angles)
    Dw = ir.D_from_log_coordinates(w)

    np.testing.assert_allclose(Da, Dw, atol=1e-6)
