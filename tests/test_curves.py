import unittest

import numpy

from hilbert import iround

from hilbert.curves import base

from hilbert.curves.lib import Log, Xlog, InverseXPolynomial, XtoA


class CurveAlgebraTests(unittest.TestCase):
    def test_linear(self):
        u = numpy.array([3, -1, 2])
        v = numpy.array([base.Curve(Xlog(1/2, pole=-1), InverseXPolynomial(3.14)),
                         +base.Curve(Log(1, pole=-1)), 0])
        w = numpy.array([1, 0, base.Curve(base.Polynomial(1, 1))])
        curve = numpy.dot(u - v, w)

        assert repr(curve) == '<Curve: (-0.5)(x + 1)log(x + 1) + (-3.14)/x + (3) + (2) + (2)x>'
        assert iround(curve(numpy.array([2.34]))[0], 7) == iround(numpy.array([
            3 - 0.5*(2.34 + 1)*numpy.log(2.34 + 1) - 3.14/2.34 + 2 + 2*2.34])[0], 7)

    def test_nonlinear(self):
        curve = +2*base.Curve(XtoA(1/2, 6/5))/5 - 1

        assert repr(curve) == '<Curve: (0.2)x^(1.2) + (-1)>'
        assert iround(curve(numpy.array([0.11]))[0], 7) == iround(numpy.array([
            -1 + 0.2*0.11**1.2])[0], 7)

    def test_braket(self):
        j = base.Curve(base.Polynomial(0, 1/2))

        assert 1 @ j == j @ 1 == 0
        assert 0 @ j == j @ 0 == 0

    def test_piecewise_product(self):
        pw = 2*base.Curve(base.Piecewise([13], [
            base.Curve(XtoA(1/2, 6/5)), -base.Curve(XtoA(1/2, 6/5))]))

        assert iround(pw(numpy.array([10.1]))[0], 7) == iround(
            numpy.array([10.1**1.2])[0], 7)
