import abc
import cmath

from hilbert import EQ_ROUND_TO

from hilbert import stock

import numpy


class PolarComplex(stock.Repr):
    def __init__(self, norm, phase):
        self.norm, self.phase = norm.real, phase.real

    def __str__(self):
        return f'{self.norm}·ej{self.phase}'

    def __eq__(self, other):
        return not self.__ne__(other)

    def __ne__(self, other):
        return bool(round(abs(self.number - (
            other.number if isinstance(other, self.__class__) else other)
        ), EQ_ROUND_TO))

    @property
    def number(self):
        return cmath.rect(self.norm, self.phase)

    @classmethod
    def from_x(cls, *numbers):
        arr = numpy.array([cls(*cmath.polar(number)) for number in numbers])

        return arr[0] if arr.shape[0] == 1 else arr

    @staticmethod
    def ph(*phases):
        return (sum(map(lambda x: x.real, phases)) - cmath.pi) % (2*cmath.pi) - cmath.pi

    def eat(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__class__(*cmath.polar(other))

        if isinstance(other, self.__class__):
            return other

        raise NotImplementedError(f'Operation with {other}')

    def eatbin(self, other, norm_phase):
        pc = self.eat(other)

        return self.__class__(*norm_phase(pc.norm, pc.phase))

    def take(self, other):
        if isinstance(other, (int, float, complex)):
            return other

        if isinstance(other, self.__class__):
            return other.number

        raise NotImplementedError(f'Operation with {other}')

    def takebin(self, other, norm_phase):
        number = self.take(other)

        return self.__class__(*norm_phase(number))

    def conjugate(self):
        return self.__class__(self.norm, -self.phase)

    def __bool__(self):
        return bool(self.norm)

    def __pos__(self):
        return self

    def __neg__(self):
        return self.__class__(self.norm, self.ph(self.phase, cmath.pi))

    def __abs__(self):
        return self.norm

    def __round__(self, to=4):
        norm_to, phase_to = (to, to) if isinstance(to, int) else to

        return self.__class__(round(self.norm, norm_to), round(self.phase, phase_to))

    def __mul__(self, other):
        return self.eatbin(other, lambda on, op: (self.norm*on, self.ph(self.phase, op)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.eatbin(other, lambda on, op: (self.norm/on, self.ph(self.phase, -op)))

    def __rtruediv__(self, other):
        return self.eatbin(other, lambda on, op: (on/self.norm, self.ph(op, -self.phase)))

    def __rpow__(self, other):
        n = self.number
        return self.eatbin(other, lambda on, op: (
            cmath.exp(-op*n.imag)*on**n.real, n.imag*cmath.log(on) + op*n.real))

    def __pow__(self, other):
        return self.takebin(other, lambda n: (
            cmath.exp(-self.phase*n.imag)*self.norm**n.real,
            n.imag*cmath.log(self.norm) + self.phase*n.real))

    def __add__(self, other):
        return self.takebin(other, lambda num: cmath.polar(self.number + num))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.takebin(other, lambda num: cmath.polar(self.number - num))

    def __rsub__(self, other):
        return -self.__sub__(other)


class Scale(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def num_prod(self, number):
        """Return the product"""

    def __mul__(self, number):
        if not number:
            return 0
        elif number == 1:
            return self
        elif isinstance(number, (int, float, complex)):
            return self.num_prod(number)
        else:
            raise NotImplementedError(f'Product by {number}')

    def __rmul__(self, number):
        return self.__mul__(number)

    def __pos__(self):
        return self

    def __neg__(self):
        return self.__mul__(-1)

    def __truediv__(self, number):
        return self.__mul__(1/number)

    def __rtruediv__(self, number):
        raise NotImplementedError(f'{self} is not /-invertible')


class AbelianSumScale(Scale, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def eat(self, other):
        """Return equivalent of `other` - if any"""

    @abc.abstractmethod
    def add_other(self, other):
        """Add other instance"""

    def __add__(self, other):
        if not other:
            return self

        other = self.eat(other)

        if isinstance(other, self.__class__):
            return self.add_other(other)

        raise NotImplementedError(f'Addition to {repr(other)}')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return -self.__sub__(other)


class Vector(AbelianSumScale, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def braket(self, other):
        """Real scalar product"""

    def __matmul__(self, other):
        if not other:
            return 0

        other = self.eat(other)

        if isinstance(other, self.__class__):
            return self.braket(other)

        raise NotImplementedError(f'Braket with {repr(other)}')

    def __rmatmul__(self, other):
        return numpy.conj(self.__matmul__(other))
