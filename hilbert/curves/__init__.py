"""Curve library"""
import numpy

from numpy.polynomial import polynomial

from hilbert.curves import base
from hilbert.curves.base import Polynomial, PiecewiseCurve  # noqa: handy imports


class Log(base.LinearCurve):
    def evaluate(self, s: numpy.array):
        return self.parameters[0]*numpy.log(s)

    def format(self, *params):
        return f'({params[0]})log{self.svar()}'


class Xlog(base.LinearCurve):
    def evaluate(self, s: numpy.array):
        return self.parameters[0]*s*numpy.log(s)

    def format(self, *params):
        return f'({params[0]}){self.svar()}log{self.svar()}'


class InverseXPolynomial(base.LinearCurve):
    def evaluate(self, s: numpy.array):
        return polynomial.Polynomial([0] + list(reversed(self.parameters)))(1/s)

    def kind(self):
        return f'Poly(-{len(self.parameters)})'

    def format(self, *params):
        return ' + '.join(reversed([
            f'({param}){self.svar(-n - 1)}'
            for n, param in enumerate(reversed(params))]))


class Exp(base.NonLinearCurve):
    def evaluate_normal(self, s: numpy.array):
        return numpy.exp(self.parameters[1]*s)

    def format(self, *params):
        return f'({params[0]})exp[({params[1]}){self.svar()}]'


class XtoA(base.NonLinearCurve):
    def evaluate_normal(self, s: numpy.array):
        return s**self.parameters[1]

    def format(self, *params):
        return f'({params[0]}){self.svar(params[1])}'
