import operator

import numpy

from hilbert import EQ_ROUND_TO

from hilbert import algebra
from hilbert import stock


@stock.FrozenLazyAttrs(('space', 'o'))
class Operator(stock.Eq, stock.WrappedDataFrame):
    def __init__(self, space, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.space = space

    def __bool__(self):
        return not self.where(self.o != 0).o.dropna(how='all').empty

    def eq(self, other):
        return self.space == other.space and not round(abs(self - other))

    def take(self, other):
        if not isinstance(other, self.__class__):
            return other

        if not other.space == self.space:
            raise NotImplementedError('Operators not based on the same space')

        return other.o

    def un(self, op):
        return self.__class__(self.space, op(self.o))

    def bin(self, other, op):
        return self.__class__(self.space, op(self.o, self.take(other)))

    def rbin(self, other, op):
        return self.__class__(self.space, op(self.take(other), self.o))

    def apply(self, function, *args, **kwargs):
        return self.un(lambda df: df.apply(function, *args, **kwargs))

    def rename(self, *args, **kwargs):
        return self.un(lambda df: df.rename(*args, inplace=False, **kwargs))

    def toggle_polar(self):
        if self.is_polar():
            return self.apply(lambda col: col.map(lambda pc: pc.number))

        return self.apply(lambda arr: algebra.PolarComplex.from_x(*arr), raw=True)

    def is_polar(self):
        return set(self.o.dtypes).pop() == numpy.object

    def where(self, condition, *args, **kwargs):
        return self.un(lambda df: df.where(condition, *args, **kwargs))

    def dagger(self):
        return self.un(lambda df: numpy.conj(df.transpose()))

    def __round__(self, to=EQ_ROUND_TO):
        if self.is_polar():
            return self.apply(lambda col: col.map(lambda x: round(x, to)))

        return self.apply(lambda col: col.round(to))

    def __pos__(self):
        return self.un(operator.pos)

    def __neg__(self):
        return self.un(operator.neg)

    def __abs__(self):
        return self.un(lambda df: df.abs())

    def __lt__(self, other):
        return self.bin(other, operator.lt).o

    def __gt__(self, other):
        return self.bin(other, operator.gt).o

    def __le__(self, other):
        return self.bin(other, operator.le).o

    def __ge__(self, other):
        return self.bin(other, operator.ge).o

    def __mul__(self, other):
        return self.bin(other, operator.mul)

    def __rmul__(self, other):
        return self.rbin(other, operator.mul)

    def __truediv__(self, other):
        return self.bin(other, operator.truediv)

    def __rtruediv__(self, other):
        return self.rbin(other, operator.truediv)

    def __add__(self, other):
        return self.bin(other, operator.add)

    def __radd__(self, other):
        return self.rbin(other, operator.add)

    def __sub__(self, other):
        return self.bin(other, operator.sub)

    def __rsub__(self, other):
        return self.rbin(other, operator.sub)

    def transform(self, vector):
        return vector.type(
            self.space, image=algebra.Image(series=self.o@vector.image.i))

    def __matmul__(self, other):
        if isinstance(other, algebra.Vector):
            return self.transform(other)

        return self.bin(other, operator.matmul)

    def __pow__(self, other):
        return self.bin(other, operator.pow)

    def is_hermitian(self):
        return self.dagger() == self

    def is_unitary(self):
        return self.dagger()@self == self.space.Id and self@self.dagger() == self.space.Id
