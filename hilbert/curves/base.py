import abc

import numpy

from numpy.polynomial import polynomial

from hilbert import algebra

from hilbert.stock import Repr, Eq

EXPONENTS = {0: '', 1: '', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹'}


class InvalidParameters(Exception):
    """Raised when curve input parameters do not suit the curve type"""


class Curve(Repr, Eq, algebra.Scale, metaclass=abc.ABCMeta):
    min_dof = None

    def __init__(self, *parameters, pole=0):
        self.parameters = self._clean_parameters(parameters)
        self.pole = pole

    def __call__(self, x: numpy.array):
        return self.evaluate(x - self.pole)

    def __str__(self):
        return self.format(*self.parameters)

    def _clean_parameters(self, params):
        if self.min_dof and len(params) < self.min_dof:
            raise InvalidParameters(f'At least {self.min_dof} parameter(s) required')

        return params

    @abc.abstractmethod
    def evaluate(self, s: numpy.array):
        """Normalized function"""

    def eqkey(self):
        return self.parameters, self.pole

    def kind(self):
        return self.__class__.__name__

    def svar(self, exponent=1):
        if exponent == 0:
            return ''

        exp = EXPONENTS.get(abs(exponent), f'^({abs(exponent)})')

        return ('/' if exponent < 0 else '') + (
            f'(x - {self.pole})'.replace('- -', '+ ') if self.pole else 'x') + exp


class CurveFormatted(Curve):
    @abc.abstractmethod
    def format(self, *params):
        """Text representation given parameters `params`"""


class LinearCurve(CurveFormatted, metaclass=abc.ABCMeta):
    def num_prod(self, number):
        return self.__class__(*(number*p for p in self.parameters), pole=self.pole)


class Polynomial(LinearCurve):
    def evaluate(self, s: numpy.array):
        return polynomial.Polynomial(self.parameters)(s)

    def kind(self):
        return f'Poly({len(self.parameters) - 1})'

    def format(self, *params):
        return ' + '.join([f'({param}){self.svar(d)}' for d, param in enumerate(params)])


class NonLinearCurve(CurveFormatted, metaclass=abc.ABCMeta):
    """Of the form c0*f(c1, ...) with parameters (c0, c1, ...)"""
    min_dof = 2

    @abc.abstractmethod
    def evaluate_normal(self, s: numpy.array):
        """Normalized function with c0 == 1"""

    def evaluate(self, s: numpy.array):
        return self.parameters[0]*self.evaluate_normal(s)

    def num_prod(self, number):
        return self.__class__(
            number*self.parameters[0], *self.parameters[1:], pole=self.pole)


class PiecewiseCurve(Curve):
    def __init__(self, jumps_at, curves):
        super().__init__(pole=0)
        self.jumps_at, self.piece_count = tuple(jumps_at), len(jumps_at) + 1
        self.curves = curves

    def eqkey(self):
        return self.jumps_at, self.curves

    def num_prod(self, number):
        return self.__class__(self.jumps_at, tuple(c*number for c in self.curves))

    def __str__(self):
        return ' | '.join(list(map(str, self.curves)))

    def kind(self):
        headed = zip(self.jumps_at, self.curves[1:])
        chain = ''.join([self.curves[0].kind()] + [
            f'[{head}]{curve.kind()}' for head, curve in headed])

        return f'PW:{chain}'

    def evaluate(self, s: numpy.array):
        functions = numpy.dot(self.conditions(s), self.curves)

        return numpy.array([f(v) for f, v in zip(functions, s)])

    def conditions(self, s: numpy.array):
        return numpy.where(numpy.array([s < self.jumps_at[0]] + [
            self.jumps_at[i] <= s < self.jumps_at[i+1] for i in range(self.piece_count - 2)
        ] + [s >= self.jumps_at[self.piece_count-2]]).transpose(), 1, 0)


class Vector(Repr, Eq, algebra.Vector):
    def __init__(self, domain, *curves):
        self.domain, self.curves = domain, curves

    def __call__(self, x: numpy.array):
        return sum([curve(x) for curve in self.curves])

    def eqkey(self):
        return self.domain, self.curves

    def eat(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(self.domain, Polynomial(other))

        return other

    def add_other(self, curve):
        return self.__class__(self.domain, *(self.curves + curve.curves))

    def num_prod(self, number):
        return self.__class__(self.domain, *(number*cu for cu in self.curves))

    def braket(self, other):
        if not (self.domain.support == other.domain.support).all():
            raise NotImplementedError(f'Curves should have the same support')

        return self.domain.measure*numpy.dot(
            numpy.conj(self(self.domain.support)).flatten(),
            other(self.domain.support).flatten())

    def __str__(self):
        return ' + '.join(list(map(str, self.curves)))

    def kind(self):
        return '+'.join(sorted(curve.kind() for curve in self.curves))
