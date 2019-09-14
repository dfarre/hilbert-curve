import unittest

import pandas

from hilbert import algebra
from hilbert import fields
from hilbert import spaces

C2 = fields.R1Field.range(spaces.LebesgueCurveSpace, 0, 1, 2)


class TensorProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.u, cls.v, cls.w, cls.x = map(lambda a: C2(series=pandas.Series(a)), (
            [1, -1j], [-2j, 3], [0, 1], [1, -1]))
        cls.uv = algebra.TensorProduct(cls.u, cls.v)
        cls.vu = algebra.TensorProduct(cls.v, cls.u)
        cls.uvw = algebra.TensorProduct(cls.u, cls.v, cls.w)
        cls.wvu = algebra.TensorProduct(cls.w, cls.v, cls.u)
        cls.uvw_wvu = algebra.TensorProductSum(cls.uvw, cls.wvu)

    def test_vector_mul_vector(self):
        assert 2.1*(self.u*self.v) == self.uv*2.1
        assert (((-1)*self.v)*((-1)*self.u))*0.3 == 0.3*self.vu

    def test_vector_mul_tensorproduct(self):
        assert (self.uv*self.w, self.w*self.vu) == (self.uvw, self.wvu)

    def test_tensorproductsum_eq(self):
        assert 3.14*self.uvw_wvu == (self.wvu + self.uvw)*3.14


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
        self.s2 = C2.operator([[0, -1j], [1j, 0]])
        self.ps2 = self.s2.toggle_polar()

    def test(self):
        assert self.s2.is_polar() is False
        assert self.ps2.is_polar() is True
        assert self.ps2.toggle_polar().is_polar() is False
        assert (self.ps2@self.ps2, self.s2@self.ps2,
                self.ps2@self.s2, self.s2@self.s2) == (C2.Id,)*4
        assert self.s2.is_hermitian() is True
        assert self.ps2.is_hermitian() is True

    def test_new_basis(self):
        U = C2.unitary_op(self.ps2).toggle_polar()

        assert U.is_polar() is True

        C2.map_basis(U)

        assert C2.is_orthonormal('new') is True
