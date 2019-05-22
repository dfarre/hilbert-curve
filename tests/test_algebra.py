import unittest

from hilbert import algebra
from hilbert import spaces


class PolarComplexTests(unittest.TestCase):
    o, mo, i, mi, z, w = algebra.PolarComplex.from_x(1, -1, 1j, -1j, 0.7 - 1.2j, -0.8 + 2j)

    def test_ones(self):
        assert (self.o*self.i**2, +self.mi**2, self.mo*self.i*self.mi, -self.mi*self.i
                ) == (-1,)*4

    def test_eq(self):
        assert (self.o, -1, self.i, -1j, self.z) == (1, self.mo, 1j, self.mi, 0.7 - 1.2j)

    def test_div_rsub_comb(self):
        assert 5 - 3.1/self.z + self.w/2 == 5 - 3.1/(0.7 - 1.2j) + (-0.8 + 2j)/2

    def test_bool(self):
        assert bool(self.o) is True
        assert bool(algebra.PolarComplex(0, 7.13)) is False

    def test_pow(self):
        x, y, z = 3.1j - 0.3, 1 + 1j, 0.7 - 1.2j

        assert (y**self.z, self.z**x) == (y**z, z**x)


class PolarOperatorTests(unittest.TestCase):
    def setUp(self):
        self.C2 = spaces.R1Field.range(spaces.LebesgueCurveSpace, 0, 1, 2)
        self.s2 = self.C2.operator([[0, -1j], [1j, 0]])
        self.ps2 = self.s2.toggle_polar()

    def test(self):
        assert self.s2.is_polar() is False
        assert self.ps2.is_polar() is True
        assert self.ps2.toggle_polar().is_polar() is False
        assert (self.ps2@self.ps2, self.s2@self.ps2,
                self.ps2@self.s2, self.s2@self.s2) == (self.C2.Id,)*4
        assert self.s2.is_hermitian() is True
        assert self.ps2.is_hermitian() is True

    def test_new_basis(self):
        U = self.C2.unitary_op(self.ps2).toggle_polar()

        assert U.is_polar() is True

        self.C2.map_basis(U)

        assert self.C2.is_orthonormal('new') is True
