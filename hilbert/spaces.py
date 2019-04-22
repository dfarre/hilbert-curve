import abc

import pandas
import numpy

from hilbert.curves import base


class Domain:
    def __init__(self, support, measure):
        self.support, self.measure = support, measure


class Reals(Domain):
    @classmethod
    def range(cls, start, end, cell):
        return cls(numpy.arange(start, end, cell), cell)

    @classmethod
    def from_index(cls, index):
        step = (index._step if isinstance(index, pandas.RangeIndex)
                else abs((index[1:] - index[:-1]).array).mean())

        return cls(index.array, step)


class Complex(Domain):
    @classmethod
    def rectangle(cls, sw, ne, re_cell, im_cell=None):
        re_cell, im_cell = re_cell, im_cell or re_cell
        size = tuple(map(int, ((ne.real - sw.real)/re_cell, (ne.imag - sw.imag)/im_cell)))

        return cls(numpy.array([[
            (sw.real + x*re_cell) + (sw.imag + y*im_cell)*1j
            for x in range(size[0])] for y in range(size[1])]), re_cell*im_cell)


class NoFiniteVectorNorm(Exception):
    """Raised for wrong vector input"""


class Space(metaclass=abc.ABCMeta):
    def __init__(self, validate_norm=True):
        self.validate_norm = validate_norm

    def __call__(self, *args, **kwargs):
        vector = self.make_vector(*args, **kwargs)

        if self.validate_norm:
            self.validate(vector)

        return vector

    @abc.abstractmethod
    def make_vector(self, *args, **kwargs):
        """Vector instance constructor returning the new vector"""

    def validate(self, vector, validate_norm=True):
        braket = vector @ vector

        if not (braket > 0 and numpy.isfinite(braket)):
            raise NoFiniteVectorNorm(
                f'{vector} does not belong to the space - no finite norm!')


class LebesgueCurveSpace(Space):
    def __init__(self, curve_domain, validate_norm=True):
        super().__init__(validate_norm)
        self.domain = curve_domain

    def make_vector(self, *args, **kwargs):
        return base.Vector(self.domain, *args, **kwargs)
