import random
import unittest

import numpy

from hilbert import spaces

from hilbert.curves import lib

R0to1L2 = spaces.R1Field.range(spaces.LebesgueCurveSpace, 0, 1, 101)
SymRm1to1L2 = spaces.R1Field.range(spaces.LebesgueCurveSpace, -1, 1, 201)
C1L2 = spaces.C1Field.rectangle(spaces.LebesgueCurveSpace, -1 - 1j, 1 + 1j, 301)

PW = 2*R0to1L2(lib.PiecewiseCurve([0.13], [lib.XtoA(1/2, 6/5), -lib.Exp(1, -1/2)]))


class CurveLibTests(unittest.TestCase):
    def test_nonlinear_raises(self):
        self.assertRaisesRegex(
            lib.InvalidParameters, r'At least 2 parameter\(s\) required', lib.XtoA, 3)

    def test_base_kind(self):
        assert lib.Log(3.14/19, pole=-2.2).kind() == 'Log'

    def test_polynomial_kind(self):
        assert lib.Polynomial(3, 2, 1).kind() == 'Poly(2)'

    def test_inverse_polynomial_kind(self):
        assert lib.InverseXPolynomial(3, 2, 1).kind() == 'Poly(-3)'

    def test_piecewise_kind(self):
        assert PW.kind() == 'PW:XtoA[0.13]Exp'

    def test_piecewise_str(self):
        assert repr(PW) == '<Vector: (1.0)x^(1.2) | (-2)exp(-0.5)x>'

    def test_delta_format(self):
        z = complex(*random.choice(C1L2.bases.o.index))
        delta = C1L2(lib.Delta(1/numpy.sqrt(C1L2.bases.measure), pole=z))

        assert repr(delta) == f'<Vector: (150.0)δ(x - {z})>'


class CurveEqualityTests(unittest.TestCase):
    def test_linear(self):
        log0, log1 = lib.Log(1), lib.Log(1, pole=-1)

        assert {log1, lib.Log(1, pole=-1)} == {log1}
        assert log1 != log0

    def test_nonlinear(self):
        assert 2*lib.XtoA(1, 1.1, pole=-1) == lib.XtoA(2, 1.1, pole=-1)
        assert 2*lib.XtoA(1, 1.1, pole=-1) != lib.XtoA(1, 1.1, pole=-1)
        assert 2*lib.XtoA(1, 1.1, pole=-1) != lib.XtoA(2, 1.1, pole=3)

    def test_piecewise_curve(self):
        pw = PW.curves[0]

        assert pw == 2*lib.PiecewiseCurve([0.13], [lib.XtoA(1/2, 6/5), -lib.Exp(1, -1/2)])
        assert PW != 2*lib.PiecewiseCurve([0.12], [lib.XtoA(1/2, 6/5), -lib.Exp(1, -1/2)])
        assert PW != 2*lib.PiecewiseCurve([0.12], [lib.XtoA(1/2, 6/5), lib.Exp(1, -1/2)])

        assert PW != 'a'


class ComplexCurveAlgebraTests(unittest.TestCase):
    def test(self):
        z = complex(*random.choice(C1L2.bases.o.index))
        u = C1L2(lib.Exp(1, -1/2))
        value = C1L2[z, 'delta']@u/numpy.sqrt(C1L2.bases.measure)

        self.assertAlmostEqual(value, u(z), places=6)
        self.assertAlmostEqual(value, numpy.exp(-0.5*z), places=6)
        self.assertAlmostEqual(u@u, 4.7008, places=1)


class RealCurveAlgebraTests(unittest.TestCase):
    def test_radd(self):
        shifted = '<Vector: (0.5)(x + 2)log(x + 2) + (1.9)>'

        assert repr(1.9 + SymRm1to1L2(lib.Xlog(1/2, pole=-2))) == shifted

    def test_linear(self):
        u = numpy.array([3, -1, 2])
        v = numpy.array([
            SymRm1to1L2(lib.Xlog(1/2, pole=-2), lib.InverseXPolynomial(3.14, pole=-2)),
            +SymRm1to1L2(lib.Log(1, pole=-2)), 0])
        w = numpy.array([1, 0, SymRm1to1L2(lib.Polynomial(1, 1))])
        vector = numpy.dot(u - v, w)
        value = SymRm1to1L2[-0.11, 'delta']@vector/numpy.sqrt(SymRm1to1L2.bases.measure)

        assert repr(vector) == '<Vector: (-0.5)(x + 2)log(x + 2) + (-3.14)/(x + 2)' \
            ' + (3) + (2) + (2)x>'
        self.assertAlmostEqual(value, vector(-0.11), places=12)
        self.assertAlmostEqual(
            value, 3 - 0.5*(-0.11 + 2)*numpy.log(-0.11 + 2) - 3.14/(-0.11 + 2) + 2 - 2*0.11,
            places=12)

    def test_nonlinear(self):
        vector = +2*R0to1L2(lib.XtoA(1/2, 6/5))/5 - 1
        value = R0to1L2[0.8, 'delta'] @ vector / numpy.sqrt(SymRm1to1L2.bases.measure)

        assert repr(vector) == '<Vector: (0.2)x^(1.2) + (-1)>'
        self.assertAlmostEqual(value, vector(0.8), places=12)
        self.assertAlmostEqual(value, -1 + 0.2*0.8**1.2, places=12)

    def test_no_finite_norn(self):
        self.assertRaisesRegex(
            spaces.InvalidVectorMade,
            r'\(2\)log\(x - 0.24\) does not belong to the space - no finite norm!',
            SymRm1to1L2, lib.Log(2, pole=0.24))

    def test_braket(self):
        u, v = R0to1L2(lib.Polynomial(1, 1)), R0to1L2(lib.Polynomial(1, -1))
        self.assertAlmostEqual(u@v, 0.67, places=2)

    def test_braket__null(self):
        j = SymRm1to1L2(lib.Polynomial(0, 1/7))

        self.assertAlmostEqual(1@j, 0, places=12)
        assert 0@j == j@0 == 0

    def test_piecewise_product(self):
        value = R0to1L2[0.1, 'delta'] @ PW / numpy.sqrt(SymRm1to1L2.bases.measure)

        self.assertAlmostEqual(value, PW(0.1), places=12)
        self.assertAlmostEqual(value, 0.1**1.2, places=12)
