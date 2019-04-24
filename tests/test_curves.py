import unittest

import numpy

from hilbert import spaces

from hilbert.curves import base

from hilbert.curves import Exp, Log, Xlog, InverseXPolynomial, XtoA

R0to1sics = spaces.LebesgueCurveSpace(spaces.Reals.range(0, 1, 0.01))
SymRm1to1sics = spaces.LebesgueCurveSpace(spaces.Reals.range(-0.99, 1, 0.01))
C1rectSics = spaces.LebesgueCurveSpace(spaces.Complex.rectangle(-1 - 1j, 1 + 1j, 0.01))

PW = 2*R0to1sics(base.PiecewiseCurve([13], [
    R0to1sics(XtoA(1/2, 6/5)), -R0to1sics(Exp(1, -1/2))]))


class CurveLibTests(unittest.TestCase):
    def test_nonlinear_raises(self):
        self.assertRaisesRegex(
            base.InvalidParameters, r'At least 2 parameter\(s\) required', XtoA, 3)

    def test_base_kind(self):
        assert Log(3.14/19, pole=-2.2).kind() == 'Log'

    def test_polynomial_kind(self):
        assert base.Polynomial(3, 2, 1).kind() == 'Poly(2)'

    def test_inverse_polynomial_kind(self):
        assert InverseXPolynomial(3, 2, 1).kind() == 'Poly(-3)'

    def test_piecewise_kind(self):
        assert PW.kind() == 'PW:XtoA[13]Exp'

    def test_piecewise_str(self):
        assert repr(PW) == '<Vector: (1.0)x^(1.2) | (-2)exp[(-0.5)x]>'


class CurveEqualityTests(unittest.TestCase):
    def test_linear(self):
        log0, log1 = Log(1), Log(1, pole=-1)

        assert {log1, Log(1, pole=-1)} == {log1}
        assert log1 != log0

    def test_nonlinear(self):
        assert 2*XtoA(1, 1.1, pole=-1) == XtoA(2, 1.1, pole=-1)
        assert 2*XtoA(1, 1.1, pole=-1) != XtoA(1, 1.1, pole=-1)
        assert 2*XtoA(1, 1.1, pole=-1) != XtoA(2, 1.1, pole=3)

    def test_piecewise_curve(self):
        pw = 2*R0to1sics(base.PiecewiseCurve([13], [
            R0to1sics(XtoA(1/2, 6/5)), -R0to1sics(XtoA(1/2, 6/5))]))

        assert pw == 2*R0to1sics(base.PiecewiseCurve([13], [
            R0to1sics(XtoA(1/2, 6/5)), -R0to1sics(XtoA(1/2, 6/5))]))
        assert pw != 2*R0to1sics(base.PiecewiseCurve([14], [
            R0to1sics(XtoA(1/2, 6/5)), -R0to1sics(XtoA(1/2, 6/5))]))


class ComplexCurveAlgebraTests(unittest.TestCase):
    def test_braket(self):
        u = C1rectSics(Exp(1, -1/2))

        assert round(u @ u, 1) == 4.7


class RealCurveAlgebraTests(unittest.TestCase):
    def test_radd(self):
        shifted = '<Vector: (0.5)(x + 1)log(x + 1) + (1.9)>'

        assert repr(1.9 + SymRm1to1sics(Xlog(1/2, pole=-1))) == shifted

    def test_linear(self):
        u = numpy.array([3, -1, 2])
        v = numpy.array([SymRm1to1sics(Xlog(1/2, pole=-1), InverseXPolynomial(3.14, pole=-1)),
                         +SymRm1to1sics(Log(1, pole=-1)), 0])
        w = numpy.array([1, 0, SymRm1to1sics(base.Polynomial(1, 1))])
        curve = numpy.dot(u - v, w)

        assert repr(curve) == '<Vector: (-0.5)(x + 1)log(x + 1) + (-3.14)/(x + 1)' \
            ' + (3) + (2) + (2)x>'
        assert round(curve(numpy.array([2.34]))[0], 7) == round(numpy.array([
            3 - 0.5*(2.34 + 1)*numpy.log(2.34 + 1) - 3.14/(2.34 + 1) + 2 + 2*2.34])[0], 7)

    def test_nonlinear(self):
        curve = +2*R0to1sics(XtoA(1/2, 6/5))/5 - 1

        assert repr(curve) == '<Vector: (0.2)x^(1.2) + (-1)>'
        assert round(curve(numpy.array([0.8]))[0], 7) == round(numpy.array([
            -1 + 0.2*0.8**1.2])[0], 7)

    def test_no_finite_norn(self):
        self.assertRaisesRegex(
            spaces.InvalidVectorMade,
            r'\(2\)logx does not belong to the space - no finite norm!',
            SymRm1to1sics, Log(2))

    def test_braket(self):
        u, v = R0to1sics(base.Polynomial(1, 1)), R0to1sics(base.Polynomial(1, -1))

        assert round(u @ v, 2) == round(v @ u, 2) == 0.67

    def test_braket__null(self):
        j = SymRm1to1sics(base.Polynomial(0, 1/7))

        assert round(1 @ j, 12) == round(j @ 1, 12) == 0
        assert 0 @ j == j @ 0 == 0

    def test_piecewise_product(self):
        pw = 2*R0to1sics(base.PiecewiseCurve([13], [
            R0to1sics(XtoA(1/2, 6/5)), -R0to1sics(XtoA(1/2, 6/5))]))

        assert round(pw(numpy.array([10.1]))[0], 7) == round(
            numpy.array([10.1**1.2])[0], 7)
