import abc

import numpy

from numpy.polynomial import polynomial

from hilbert import Repr
from hilbert import algebra

EXPONENTS = {0: '', 1: '', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹'}


class AbstractCurve(Repr, algebra.Scale, metaclass=abc.ABCMeta):
    def __init__(self, *parameters, pole=0):
        self.parameters, self.pole = numpy.array(parameters), pole

    def __call__(self, x: numpy.array):
        return self.evaluate(x - self.pole)

    def __str__(self):
        return self.format(*self.parameters)

    @abc.abstractmethod
    def evaluate(self, s: numpy.array):
        """Normalized function"""

    def kind(self):
        return self.__class__.__name__

    def svar(self, exponent=1):
        if exponent == 0:
            return ''

        exp = EXPONENTS.get(abs(exponent), f'^({abs(exponent)})')

        return ('/' if exponent < 0 else '') + (
            f'(x - {self.pole})'.replace('- -', '+ ') if self.pole else 'x') + exp

    def format(self, *params):
        """Text representation given parameters `params`"""


class LinearCurve(AbstractCurve, metaclass=abc.ABCMeta):
    def num_prod(self, number):
        return self.__class__(*(number*self.parameters), pole=self.pole)


class NonLinearCurve(AbstractCurve, metaclass=abc.ABCMeta):
    def __init__(self, *parameters, pole=0, factor=1):
        super().__init__(*parameters, pole=pole)
        self._mul = factor

    def num_prod(self, number):
        return self.__class__(
            *self.parameters, pole=self.pole, factor=number*self._mul)


class Piecewise(NonLinearCurve):
    def __init__(self, jumps_at, curves, pole=0, factor=1):
        super().__init__(pole=pole, factor=factor)
        self.jumps_at, self.piece_count = jumps_at, len(jumps_at) + 1
        self.curves = curves

    def num_prod(self, number):
        return self.__class__(
            self.jumps_at, self.curves, pole=self.pole, factor=number*self._mul)

    def __str__(self):
        return ' | '.join(list(map(str, self.curves)))

    def kind(self):
        headed = zip(self.jumps_at, self.curves[1:])
        chain = '-'.join([self.curves[0].kind()] + [
            f'[{head}]{curve.kind()}' for head, curve in headed])

        return f'PW:{chain}'

    def evaluate(self, s: numpy.array):
        functions = numpy.dot(self.conditions(s), self.curves)

        return self._mul*numpy.array([f(v) for f, v in zip(functions, s)])

    def conditions(self, s: numpy.array):
        return numpy.where(numpy.array([s < self.jumps_at[0]] + [
            self.jumps_at[i] <= s < self.jumps_at[i+1] for i in range(self.piece_count - 2)
        ] + [s >= self.jumps_at[self.piece_count-2]]).transpose(), 1, 0)


class Polynomial(LinearCurve):
    def evaluate(self, s: numpy.array):
        return polynomial.Polynomial(self.parameters)(s)

    def kind(self):
        return f'Poly({len(self.parameters) - 1})'

    def format(self, *params):
        return ' + '.join([f'({param}){self.svar(d)}' for d, param in enumerate(params)])


class Curve(Repr, algebra.Vector):
    support = numpy.arange(-1, 1.01, 0.01)

    def __init__(self, *curves, shift=0):
        self.curves = curves
        self._add = shift

    def __call__(self, x: numpy.array):
        return self._add + sum([curve(x) for curve in self.curves])

    def eat(self, other):
        if isinstance(other, (int, float)):
            return self.__class__(Polynomial(other))

        return other

    def add_other(self, curve):
        return self.__class__(*(self.curves + curve.curves), shift=self._add+curve._add)

    def num_prod(self, number):
        return self.__class__(*(number*cu for cu in self.curves), shift=number*self._add)

    def braket(self, other):
        if (self.support == other.support).all():
            return numpy.dot(numpy.conj(self(self.support)), other(self.support))

        raise NotImplementedError(f'Curves should have the same support')

    def __str__(self):
        return ' + '.join(([str(self._add)] if self._add else [])
                          + list(map(str, self.curves)))

    def kind(self):
        return '+'.join(sorted(curve.kind() for curve in self.curves))
