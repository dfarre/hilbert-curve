import abc
import decimal

import numpy


class Repr(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __str__(self):
        """Object's text content"""

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self}>'


class Eq(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def eqkey(self):
        """Return hashable key property to compare to others"""

    def __eq__(self, other):
        return self.eqkey() == other.eqkey()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.eqkey())


def get_exponent(number):
    if number in {-numpy.inf, numpy.inf}:
        return numpy.inf

    return int(numpy.format_float_scientific(float(number)).split('e')[1])


def iround(number, to=1):
    exp = get_exponent(number)

    if exp == numpy.inf:
        return decimal.Decimal(number)

    return decimal.Decimal(str(number)).scaleb(-exp).quantize(
        decimal.Decimal('1.' + '0'*(to - 1)), rounding=decimal.ROUND_HALF_UP
    ).scaleb(exp)
