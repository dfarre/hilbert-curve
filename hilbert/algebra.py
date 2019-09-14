import abc
import cmath
import functools
import itertools
import operator

from hilbert import EQ_ROUND_TO

from hilbert import stock

import pandas
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

        raise NotImplementedError(f'Operation with {repr(other)}')

    def eatbin(self, other, norm_phase):
        pc = self.eat(other)

        return self.__class__(*norm_phase(pc.norm, pc.phase))

    def take(self, other):
        if isinstance(other, (int, float, complex)):
            return other

        if isinstance(other, self.__class__):
            return other.number

        raise NotImplementedError(f'Operation with {repr(other)}')

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


class Scalable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def num_prod(self, number):
        """Return the instance scaled by number"""

    def __mul__(self, other):
        if not other:
            return 0
        elif other == 1:
            return self
        elif isinstance(other, (int, float, complex)):
            return self.num_prod(other)
        else:
            raise NotImplementedError(f'Product with {repr(other)}')

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


class ScalableAbelianSum(Scalable, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add_other(self, other):
        """Add other instance"""

    def __add__(self, other):
        if not other:
            return self

        return self.add_other(self.eat(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def eat(self, other):
        """Override to return `other` as an instance of `self.__class__`"""
        if isinstance(other, self.__class__):
            return other

        raise NotImplementedError(
            f'Cannot interpret {repr(other)} as an instance of {self.__class__}')


@stock.FrozenLazyAttrs(('space',), ('image',))
class Vector(ScalableAbelianSum, stock.Repr, stock.Hashable, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def braket(self, other):
        """Complex scalar product"""

    def __matmul__(self, other):
        return self.braket(self.eat(other))

    def __rmatmul__(self, other):
        return numpy.conj(self.__matmul__(other))

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return TensorProduct(self, other)
        elif isinstance(other, TensorProduct):
            return other.__class__(self, *other.vectors)
        else:
            return super().__mul__(other)

    def eqkey(self):
        return self.space, tuple(self.image.i.round(EQ_ROUND_TO))


class TensorProduct(Vector):
    def __init__(self, *vectors):
        self.vectors = vectors

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(*(self.vectors + other.vectors))
        elif isinstance(other, Vector):
            return self.__class__(*(self.vectors + (other,)))
        elif isinstance(other, TensorProductSum):
            return TensorProductSum(*(self.__class__(*(self.vectors + (oth.vectors,)))
                                      for oth in other.products))
        else:
            return super(Vector, self).__mul__(other)

    def __matmul__(self, other):
        if isinstance(other, TensorProductSum):
            return sum(self.braket(oth) for oth in other.products)
        else:
            return super().__mul__(other)

    def __str__(self):
        return ' ⊗ '.join(map(str, self.vectors))

    def _make_image(self):
        return functools.reduce(operator.mul, (v.image for v in self.vectors))

    def braket(self, other):
        return functools.reduce(operator.mul, (
            u.braket(v) for u, v in zip(self.vectors, other.vectors)))

    def num_prod(self, number):
        return self.__class__(self.vectors[0].num_prod(number), *self.vectors[1:])

    def add_other(self, other):
        return TensorProductSum(self, other)

    def eat(self, other):
        if isinstance(other, Vector):
            return self.__class__(other)
        else:
            return super().eat(other)


class TensorProductSum(Vector):
    def __init__(self, *tensor_products):
        self.products = tensor_products

    def __str__(self):
        return ' + '.join(map(str, self.products))

    def _make_image(self):
        return functools.reduce(operator.add, (v.image for v in self.products))

    def braket(self, other):
        return sum(x.braket(y) for x, y in itertools.product(self.products, other.products))

    def num_prod(self, number):
        return self.__class__(*(pr.num_prod(number) for pr in self.products))

    def add_other(self, other):
        return self.__class__(*(self.products + other.products))

    def eat(self, other):
        if isinstance(other, TensorProduct):
            return self.__class__(other)
        elif isinstance(other, Vector):
            return self.__class__(TensorProduct(other))
        else:
            return super().eat(other)


class Image(stock.Repr, Scalable):
    def __init__(self, *args, series=None, **kwargs):
        self.i = pandas.Series(*args, **kwargs) if series is None else series

    def __str__(self):
        return f'\n{self.i}'

    def __getitem__(self, x):
        return self.i.loc[x]

    def __add__(self, other):
        return self.__class__(series=self.i + self.take(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        taken = self.take(other)

        if isinstance(taken, pandas.Series):
            return self.__class__(series=self.tensor_product(self.i, taken))

        return super().__mul__(taken)

    def __rmul__(self, other):
        taken = self.take(other)

        if isinstance(taken, pandas.Series):
            return self.__class__(series=self.tensor_product(taken, self.i))

        return super().__rmul__(taken)

    def num_prod(self, number):
        return self.__class__(series=number*self.i)

    def take(self, other):
        return other.i if isinstance(other, self.__class__) else other

    @staticmethod
    def tensor_product(lseries, rseries):
        series = pandas.concat(iter(lseries.map(lambda x: x*rseries)))
        llevels = (lseries.index.levels if isinstance(lseries.index, pandas.MultiIndex)
                   else [lseries.index])
        rlevels = (rseries.index.levels if isinstance(rseries.index, pandas.MultiIndex)
                   else [rseries.index])
        series.index = pandas.MultiIndex.from_product(llevels + rlevels)

        return series

    @property
    def real(self):
        return pandas.Series(self.i.real, index=self.i.index)

    @property
    def imag(self):
        return pandas.Series(self.i.imag, index=self.i.index)

    @property
    def density(self):
        return self.i.abs()**2
