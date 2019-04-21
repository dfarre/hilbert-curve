import numpy

from numpy.polynomial import polynomial

from hilbert.curves import base


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


class XtoA(base.NonLinearCurve):
    def evaluate(self, s: numpy.array):
        return self._mul*self.parameters[0]*s**self.parameters[1]

    def format(self, *params):
        return f'({self._mul*params[0]}){self.svar(params[1])}'
