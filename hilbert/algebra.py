import abc
import cmath
import functools
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

    def prod(self, other):
        raise NotImplementedError(f'Product {repr(self)} * {repr(other)}')

    def rprod(self, other):
        raise NotImplementedError(f'Product {repr(other)} * {repr(self)}')

    def _num_prod(self, number):
        return self if number == 1 else 0 if number == 0 else self.num_prod(number)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self._num_prod(other)

        return self.prod(other)

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self._num_prod(other)

        return self.rprod(other)

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

    def eat(self, other):
        """Override to return `other` as an instance of `self.__class__`"""
        if isinstance(other, self.__class__):
            return other

        raise NotImplementedError(
            f'Cannot interpret {repr(other)} as an instance of {self.__class__}')

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


@stock.FrozenLazyAttrs(lazy_keys=('image',))
class Vector(ScalableAbelianSum, stock.Repr, stock.Hashable, metaclass=abc.ABCMeta):
    def __call__(self, ix):
        return self.image[ix]

    def braket(self, other):
        return self.measure*(self.image@other.image)

    def __matmul__(self, other):
        return self.braket(self.eat(other))

    def __rmatmul__(self, other):
        return numpy.conj(self.__matmul__(other))

    def stack(self, *args, **kwargs):
        return self.type(*args, image=self.image, **kwargs)

    @property
    def type(self):
        return self.__class__

    @property
    def measure(self):
        return 1

    @property
    def norm(self):
        return numpy.sqrt(self@self)

    def prod(self, other):
        if isinstance(other, self.__class__):
            return TensorProduct(self, other)
        elif isinstance(other, TensorProduct):
            return other.__class__(self, *other.vectors)
        elif isinstance(other, TensorProductSum):
            return other.__class__(*(self.prod(pr) for pr in other.products))

        return super().prod(self, other)

    def rprod(self, other):
        if isinstance(other, self.__class__):
            return TensorProduct(other, self)
        elif isinstance(other, TensorProduct):
            return other.__class__(*(other.vectors + (self,)))
        elif isinstance(other, TensorProductSum):
            return other.__class__(*(self.rprod(pr) for pr in other.products))

        return super().rprod(self, other)

    def eqkey(self):
        return self.image


class TensorProduct(Vector):
    def __init__(self, *vectors):
        super().__init__()
        self.vectors = vectors

    @property
    def type(self):
        return self.vectors[0].type

    def prod(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(*(self.vectors + other.vectors))
        elif isinstance(other, Vector):
            return other.rprod(self)
        elif isinstance(other, TensorProductSum):
            return TensorProductSum(*(self.prod(oth) for oth in other.products))

        return super().prod(self, other)

    def rprod(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(*(other.vectors + self.vectors))
        elif isinstance(other, Vector):
            return other.prod(self)
        elif isinstance(other, TensorProductSum):
            return TensorProductSum(*(self.rprod(oth) for oth in other.products))

        return super().rprod(self, other)

    def __str__(self):
        return ' ⊗ '.join(map(str, self.vectors))

    def _make_image(self):
        return functools.reduce(operator.mul, (v.image for v in self.vectors))

    def add_other(self, other):
        return TensorProductSum(self, other)

    def num_prod(self, number):
        return self.__class__(self.vectors[0].num_prod(number), *self.vectors[1:])

    def eat(self, other):
        if isinstance(other, Vector):
            return self.__class__(other)
        else:
            return super().eat(other)


class TensorProductSum(Vector):
    def __init__(self, *tensor_products):
        super().__init__()
        self.products = tensor_products

    @property
    def type(self):
        return self.products[0].type

    def __str__(self):
        return ' + '.join(map(str, self.products))

    def prod(self, other):
        if isinstance(other, self.__class__):
            return sum(self.prod(pr) for pr in other.products)
        elif isinstance(other, (Vector, TensorProduct)):
            return other.rprod(self)

        return super().prod(self, other)

    def rprod(self, other):
        if isinstance(other, self.__class__):
            return sum(self.rprod(pr) for pr in other.products)
        elif isinstance(other, (Vector, TensorProduct)):
            return other.prod(self)

        return super().rprod(self, other)

    def _make_image(self):
        return functools.reduce(operator.add, (v.image for v in self.products))

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


class Image(Vector):
    def __init__(self, *args, series=None, **kwargs):
        self.i = pandas.Series(*args, **kwargs) if series is None else series
        super().__init__()

    def _make_image(self):
        return self

    def braket(self, other):
        return numpy.dot(numpy.conj(self.i), other.i)

    def eqkey(self):
        return tuple(self.i.round(EQ_ROUND_TO))

    def __str__(self):
        return f'\n{self.i}'

    def __getitem__(self, x):
        item = self.i.loc[x]

        return self.__class__(series=item) if isinstance(item, pandas.Series) else item

    def add_other(self, other):
        return self.__class__(series=self.i + other.i)

    def prod(self, other):
        return self.__class__(series=self.tensor_product(self.i, self.eat(other).i))

    def rprod(self, other):
        return self.__class__(series=self.tensor_product(self.eat(other).i, self.i))

    def num_prod(self, number):
        return self.__class__(series=number*self.i)

    def eat(self, other):
        if isinstance(other, Vector):
            return other.image

        if isinstance(other, pandas.Series):
            return self.__class__(series=other)

        return super().eat(other)

    @staticmethod
    def tensor_product(lseries, rseries):
        series = pandas.concat(iter(lseries.map(lambda x: x*rseries)))
        series.index = stock.index_from_product(lseries.index, rseries.index)

        return series

    @property
    def real(self):
        return pandas.Series(numpy.real(self.i.to_numpy()), index=self.i.index)

    @property
    def imag(self):
        return pandas.Series(numpy.imag(self.i.to_numpy()), index=self.i.index)

    @property
    def density(self):
        return self.i.abs()**2

    def full_plot(self, top_ax, bottom_ax, **kwargs):
        self.real.plot(ax=top_ax, **kwargs)
        self.imag.plot(ax=bottom_ax, **kwargs)

    def density_plot(self, axes, **kwargs):
        self.density.plot(ax=axes, **kwargs)

    @stock.PyplotShow(nrows=2, ncols=1)
    def show(self, **kwargs):
        top_ax, bottom_ax = kwargs.pop('axes')
        self.full_plot(top_ax, bottom_ax, **kwargs)
        top_ax.yaxis.set_label_text('Re')
        bottom_ax.yaxis.set_label_text('Im')
        top_ax.grid()
        bottom_ax.grid()

    @stock.PyplotShow()
    def show_density(self, **kwargs):
        axes = kwargs.pop('axes')
        self.density_plot(axes, **kwargs)
        axes.yaxis.set_label_text('abs²')
        axes.grid()
