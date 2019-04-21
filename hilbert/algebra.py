import abc

import numpy


class Scale(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def num_prod(self, number):
        """Return the product"""

    def __mul__(self, number):
        if not number:
            return 0
        elif number == 1:
            return self
        elif isinstance(number, (int, float)):
            return self.num_prod(number)
        else:
            raise NotImplementedError(f'Product by {number}')

    def __rmul__(self, number):
        return self.__mul__(number)

    def __truediv__(self, number):
        return self.__mul__(1/number)

    def __rtruediv__(self, number):
        raise NotImplementedError(f'{self} is not /-invertible')


class Vector(Scale, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def eat(self, other):
        """Return vector equivalent of `other` - if any"""

    @abc.abstractmethod
    def add_other(self, other):
        """Add other vector"""

    @abc.abstractmethod
    def braket(self, other):
        """Real scalar product"""

    def __add__(self, other):
        if not other:
            return self

        other = self.eat(other)

        if isinstance(other, self.__class__):
            return self.add_other(other)

        raise NotImplementedError(f'Addition to {other}')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def __pos__(self):
        return self

    def __matmul__(self, other):
        if not other:
            return 0

        other = self.eat(other)

        if isinstance(other, self.__class__):
            return self.braket(other)

        raise NotImplementedError(f'Braket with {other}')

    def __rmatmul__(self, other):
        return numpy.conj(self.__matmul__(other))
