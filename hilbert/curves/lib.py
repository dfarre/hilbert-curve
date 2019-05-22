"""Curve library"""
import abc

import numpy
from numpy.polynomial import polynomial

from hilbert import EQ_ROUND_TO

from hilbert import algebra
from hilbert import stock

EXPONENTS = {0: '', 1: '', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹'}


class InvalidParameters(Exception):
    """Raised when curve input parameters do not suit the curve type"""


# Abstract base curves #########################################################

@stock.FrozenLazyAttrs(('parameters', 'pole'))
class Curve(stock.Repr, stock.Hashable, algebra.Scale,
            metaclass=abc.ABCMeta):
    min_dof = None

    def __init__(self, *parameters, pole=0):
        self.parameters = self._clean_parameters(parameters)
        self.pole = pole

    def __call__(self, x: numpy.array):
        return self.evaluate(x - self.pole)

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
    def __str__(self):
        return self.format(*self.parameters)

    @abc.abstractmethod
    def format(self, *params):
        """Text representation given parameters `params`"""


class LinearCurve(CurveFormatted, metaclass=abc.ABCMeta):
    def num_prod(self, number):
        return self.__class__(*(number*p for p in self.parameters), pole=self.pole)


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


class BasisCurve(Curve):
    """Orthonormal basis vector with a first parameter 1/√measure"""

    def scale(self, k0, k):
        return self.__class__(
            k0*self.parameters[0], *self.parameters[1:], pole=self.pole*k)


# Piecewise curves #############################################################

@stock.FrozenLazyAttrs(('jumps_at', 'piece_count', 'curves'))
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
        functions = (self.curves[b.nonzero()[0][0]] for b in self.conditions(s))

        return numpy.array([f(v) for f, v in zip(functions, s)])

    def conditions(self, s: numpy.array):
        return numpy.array([s < self.jumps_at[0]] + [
            self.jumps_at[i] <= s < self.jumps_at[i+1] for i in range(self.piece_count - 2)
        ] + [s >= self.jumps_at[self.piece_count-2]]).transpose()


# Non-linear curves ############################################################

class Exp(NonLinearCurve, BasisCurve):
    """Orthonormal basis vector with a factor 1/(√measure√dimension)"""

    def evaluate_normal(self, s: numpy.array):
        return numpy.exp(self.parameters[1]*s)

    def format(self, coef, exp_coef):
        return f'({coef})exp({exp_coef}){self.svar()}'


class XtoA(NonLinearCurve):
    def evaluate_normal(self, s: numpy.array):
        return s**self.parameters[1]

    def format(self, coef, exp_coef):
        return f'({coef}){self.svar(exp_coef)}'


# Linear curves ################################################################

class Delta(LinearCurve, BasisCurve):
    def evaluate(self, s: numpy.array):
        return numpy.where(s == 0, self.parameters[0], 0)

    def format(self, inverse_sqrt_measure):
        return f'({inverse_sqrt_measure})δ{self.svar()}'


class Polynomial(LinearCurve):
    def evaluate(self, s: numpy.array):
        return polynomial.Polynomial(self.parameters)(s)

    def kind(self):
        return f'Poly({len(self.parameters) - 1})'

    def format(self, *params):
        return ' + '.join([f'({param}){self.svar(d)}' for d, param in enumerate(params)])


class Log(LinearCurve):
    def evaluate(self, s: numpy.array):
        return self.parameters[0]*numpy.log(s)

    def format(self, coefficient):
        return f'({coefficient})log{self.svar()}'


class Xlog(LinearCurve):
    def evaluate(self, s: numpy.array):
        return self.parameters[0]*s*numpy.log(s)

    def format(self, coefficient):
        return f'({coefficient}){self.svar()}log{self.svar()}'


class InverseXPolynomial(LinearCurve):
    def evaluate(self, s: numpy.array):
        return polynomial.Polynomial([0] + list(reversed(self.parameters)))(1/s)

    def kind(self):
        return f'Poly(-{len(self.parameters)})'

    def format(self, *params):
        return ' + '.join(reversed([f'({param}){self.svar(-n - 1)}'
                                    for n, param in enumerate(reversed(params))]))


@stock.FrozenLazyAttrs(('image',))
class ImageCurve(Curve):
    """General, non-analytical definition for vectors"""

    def __init__(self, image):
        super().__init__(pole=0)
        self.image = image

    def __str__(self):
        return str(self.image)

    def eqkey(self):
        return self.pole, tuple(self.image.i.round(EQ_ROUND_TO))

    def num_prod(self, number):
        return self.__class__(self.image.__class__(series=number*self.image.i))

    def evaluate(self, s: numpy.array):
        return numpy.array([self.image[x] for x in s])
